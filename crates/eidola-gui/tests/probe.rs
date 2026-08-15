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

use eidola_app_core::error::AppError;
use eidola_app_core::{
    AttestationInfo, BalancesResult, ConfigState, ModelInfo, PostBlock, PostNode, PostParticipant,
    PriceInfo, RequestInfo, SpendTrailEntry,
};
use eidola_app_core::{
    ParticipantInfo, ParticipantReference, SpaceTemplateInfo, TemplateParticipantInfo,
};
use eidola_gui::agents_settings::AgentsSettingsView;
use eidola_gui::general::GeneralView;
use eidola_gui::library::LibraryView;
use eidola_gui::onboarding::{OnboardingView, Slide};
use eidola_gui::probe;
use eidola_gui::record::{RecordSection, RecordView};
use eidola_gui::space_view::SpaceView;
use eidola_gui::stores::{Stores, StoresStub};
use eidola_gui::templates_settings::TemplatesSettingsView;
use eidola_gui::wallet::WalletView;
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

// ---------------------------------------------------------------------------
// Landmarks — the containers that give the tree a shape.
//
// Before these, every affordance hung directly off the window root (role-less
// containers collapse, so children attach to the nearest role-bearing
// ancestor). Each test asserts the container's *role*, which is what the
// macOS adapter turns into an `AXLandmark*` / `AXList` a screen reader can
// navigate by.
// ---------------------------------------------------------------------------

#[gpui::test]
fn library_listing_is_a_named_list(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let stores = stub_stores(cx, |s| {
        s.spaces = vec![space_info("s1", Some("Tides and the moon"))];
    });
    let (window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| LibraryView::new(stores, window, cx))
    });

    let entries = fresh_entries(cx, window);
    assert_probe(&entries, "library/list", gpui::Role::List, "Spaces");
    assert_probe(
        &entries,
        "library/row/0",
        gpui::Role::ListItem,
        "Tides and the moon",
    );

    probe::set_probes_enabled(false);
}

/// The Library's one op-error banner: a refused create / rename / archive is an
/// `Alert` whose label is the sentence itself (there is no `aria_live` at this
/// pin, so an Alert is perceivable but silent — the message must therefore be
/// what a reader lands on), beside a real Dismiss button.
#[gpui::test]
fn library_op_error_banner_is_an_alert_with_a_dismiss(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let stores = stub_stores(cx, |s| {
        s.spaces = vec![space_info("s1", Some("Tides and the moon"))];
    });
    let refusal = "Couldn't rename this space: space not found: s1";
    stores.spaces.update(cx, |s, cx| {
        s.settle_for_test(
            Some("s1".into()),
            Ok(vec![space_info("s1", Some("Tides and the moon"))]),
            Some(refusal),
            cx,
        )
    });
    let stores_for_view = stores.clone();
    let (window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| LibraryView::new(stores_for_view, window, cx))
    });

    let entries = fresh_entries(cx, window);
    assert_probe(&entries, "library/op-error", gpui::Role::Alert, refusal);
    assert_probe(
        &entries,
        "library/op-error/dismiss",
        gpui::Role::Button,
        "Dismiss",
    );

    // Dismissing is the store's, so the banner leaves every window at once.
    stores.spaces.update(cx, |s, cx| s.clear_op_error(cx));
    let entries = fresh_entries(cx, window);
    assert!(
        !entries.iter().any(|(n, _)| n == "library/op-error"),
        "a dismissed refusal leaves no banner behind"
    );

    probe::set_probes_enabled(false);
}

/// **"Failed is not empty" for the Library.** A failed *initial* read leaves
/// `list()` answering `&[]` exactly as a genuinely empty Library does, so the
/// page used to render "Nothing here yet — ⌘N starts a new space": a read error
/// stated as a fact about the reader's spaces, with no error, no retry, and ⌘N
/// as the only way forward. The shared `load_error_panel` is the house answer.
#[gpui::test]
fn library_failed_initial_load_offers_a_retry_not_an_empty_page(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let stores = stub_stores(cx, |_| {});
    stores.spaces.update(cx, |s, cx| {
        s.settle_for_test(None, Err("database is locked"), None, cx)
    });
    let (window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| LibraryView::new(stores, window, cx))
    });

    let entries = fresh_entries(cx, window);
    assert_probe(&entries, "library/retry", gpui::Role::Button, "Retry");
    assert!(
        !entries.iter().any(|(n, _)| n == "library/list"),
        "a failed initial read renders no listing to be mistaken for an empty one"
    );

    probe::set_probes_enabled(false);
}

/// The other half: a failed *refresh* over a listing we still hold keeps the
/// rows (never a blank page over a page we had) and adds the quiet retry — the
/// state a write's unconditional re-list reaches when the write lands and the
/// read behind it does not.
#[gpui::test]
fn library_failed_refresh_keeps_its_rows_and_offers_a_retry(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let stores = stub_stores(cx, |s| {
        s.spaces = vec![space_info("s1", Some("Tides and the moon"))];
    });
    stores.spaces.update(cx, |s, cx| {
        s.settle_for_test(None, Err("database is locked"), None, cx)
    });
    let (window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| LibraryView::new(stores, window, cx))
    });

    let entries = fresh_entries(cx, window);
    assert_probe(
        &entries,
        "library/row/0",
        gpui::Role::ListItem,
        "Tides and the moon",
    );
    assert_probe(&entries, "library/retry", gpui::Role::Button, "Retry");

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn settings_nav_and_content_are_landmarks(cx: &mut TestAppContext) {
    use eidola_gui::settings::{SettingsPane, SettingsView};

    let _guard = probes_on();

    let stores = ready_stores(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SettingsView::new(stores, window, cx))
    });

    let entries = fresh_entries(cx, window);
    assert_probe(
        &entries,
        "settings/nav",
        gpui::Role::TabList,
        "Settings sections",
    );
    // The content landmark is named for the pane it holds, so switching panes
    // renames it rather than leaving one anonymous "Main".
    assert_probe(
        &entries,
        "settings/content",
        gpui::Role::Main,
        "General settings",
    );

    view.update(cx, |v, cx| v.select(SettingsPane::Backends, cx));
    let entries = fresh_entries(cx, window);
    assert_probe(
        &entries,
        "settings/content",
        gpui::Role::Main,
        "Backends settings",
    );
    assert_probe(
        &entries,
        "settings/backends/tabs",
        gpui::Role::TabList,
        "Backend kinds",
    );

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn record_strip_and_body_are_landmarks(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let stores = ready_stores(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| RecordView::new(stores, window, cx))
    });

    // Empty section: the body is a named region, not a list of nothing.
    let entries = fresh_entries(cx, window);
    assert_probe(
        &entries,
        "record/sections",
        gpui::Role::TabList,
        "Record sections",
    );
    assert_probe(&entries, "record/body", gpui::Role::Region, "Attestations");

    // Populated: the body becomes the `List` parent the rows belong to.
    view.update(cx, |v, _| {
        v.set_attestations_for_test(vec![stub_attestation("att-hash-1")], false);
    });
    let entries = fresh_entries(cx, window);
    assert_probe(&entries, "record/body", gpui::Role::List, "Attestations");

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn space_window_landmarks_name_conversation_and_map(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let stores = ready_stores(cx);
    let (window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, None, WindowInput::new(cx), window, cx))
    });

    let entries = fresh_entries(cx, window);
    assert_probe(
        &entries,
        "space/conversation",
        gpui::Role::Main,
        "Conversation",
    );
    // The minimap is the window's table of contents — a navigation landmark,
    // so its position at the end of the reading order is reachable rather
    // than a burial. (The literal child reorder is blocked by the composer's
    // paint-order dependency; see AGENTS.md → Accessibility.)
    assert_probe(
        &entries,
        "space/minimap",
        gpui::Role::Navigation,
        "Conversation map",
    );
    assert_probe(
        &entries,
        "space/composer",
        gpui::Role::TextInput,
        "Message composer",
    );

    probe::set_probes_enabled(false);
}

// ---------------------------------------------------------------------------
// Label quality — a probe's label is what a screen reader says, so these
// assert the *wording*, not just that a probe exists. Every case below is a
// finding from the task-12a audit (§S8).
// ---------------------------------------------------------------------------

#[gpui::test]
fn row_verbs_name_the_row_they_act_on(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let stores = stub_stores(cx, |s| {
        s.spaces = vec![
            space_info("s1", Some("Tides and the moon")),
            space_info("s2", Some("Borrow checker")),
        ];
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| LibraryView::new(stores, window, cx))
    });
    // The verbs are hover-revealed; force the hover the snapshot tests use.
    view.update(cx, |v, _| v.set_hovered_for_test(Some(1)));

    let entries = fresh_entries(cx, window);
    assert_probe(
        &entries,
        "library/row/1/rename",
        gpui::Role::Button,
        "Rename Borrow checker",
    );
    assert_probe(
        &entries,
        "library/row/1/archive",
        gpui::Role::Button,
        "Archive Borrow checker",
    );

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn participant_row_verbs_name_the_participant(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let (window, view) = open_participants_inspector(cx);

    // The row is the disclosure, so it is named by the member it opens.
    let entries = fresh_entries(cx, window);
    assert_probe(
        &entries,
        "space/inspector/participants/agent-1",
        gpui::Role::Button,
        "Assistant",
    );

    // Repeated "Remove" with nothing to distinguish it is exactly the audit's
    // context-free-label finding; the row's subject supplies it.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.inspector_toggle_participant("agent-1", window, cx)
        });
    })
    .unwrap();
    let entries = fresh_entries(cx, window);
    assert_probe(
        &entries,
        "space/inspector/participants/agent-1/remove",
        gpui::Role::Button,
        "Remove Assistant",
    );

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn appearance_and_text_size_chips_carry_their_group(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let stores = ready_stores(cx);
    let (window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| GeneralView::new(stores.config.clone(), window, cx))
    });

    let entries = fresh_entries(cx, window);
    // "Auto" / "System" / "Day" / "Night" say nothing on their own — the
    // group label they sit beside is a node-less `div`.
    assert_probe(
        &entries,
        "settings/general/appearance/auto",
        gpui::Role::Button,
        "Day & night: Auto",
    );
    assert_probe(
        &entries,
        "settings/general/time-of-day/on",
        gpui::Role::Button,
        "Time of day: On",
    );
    // The current scale is likewise invisible to AT, so the chips carry it.
    assert_probe(
        &entries,
        "settings/general/text-size/larger",
        gpui::Role::Button,
        "Larger text, currently 100%",
    );
    // The login-item switch is a `Switch` with no node of its own; the
    // probed wrapper is the control, and its name has to be self-contained
    // (the "Open at login" field label is a node-less `div`).
    assert_probe(
        &entries,
        "settings/general/login-item",
        gpui::Role::CheckBox,
        "Open Eidola at login",
    );

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn minimap_labels_read_as_prose_not_markdown(cx: &mut TestAppContext) {
    // The minimap cells are the only place a post's *text* reaches assistive
    // technology today, and they used to carry raw markdown and raw
    // `{{ embed N }}` wire syntax cut mid-word at 56 characters.
    let _guard = probes_on();

    let stores = ready_stores(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, None, WindowInput::new(cx), window, cx))
    });
    let space = view.read_with(cx, |v, _| v.space().clone());
    cx.update(|cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(
                vec![probe_post(
                    "a1",
                    "**This week: the shepherd's bargain.** In *Republic* I, the argument turns.",
                )],
                cx,
            )
        });
    });

    let entries = fresh_entries(cx, window);
    let (_, cell) = entries
        .iter()
        .find(|(n, _)| n == "space/minimap/cell/0/0")
        .unwrap_or_else(|| {
            let names: Vec<&String> = entries.iter().map(|(n, _)| n).collect();
            panic!("minimap cell probe missing; recorded: {names:?}");
        });
    assert!(
        !cell.label.contains('*'),
        "minimap label still carries markdown: {:?}",
        cell.label
    );
    assert!(
        cell.label.contains("This week: the shepherd's bargain."),
        "minimap label lost the prose: {:?}",
        cell.label
    );
    // Truncated on a word boundary, with an ellipsis — never mid-word.
    assert!(
        cell.label.ends_with('…'),
        "minimap label should be truncated here: {:?}",
        cell.label
    );

    probe::set_probes_enabled(false);
}

// ---------------------------------------------------------------------------
// Content exposure (wave C) — the tree stops being affordances-only.
//
// Everything here asserts the **value** channel as well as the role and name,
// because the content is the point: a post whose text is absent, a balance
// nobody can hear, an alert with no message. The audit's §4 rule bounds it —
// a value may only be bound to text that has settled.
// ---------------------------------------------------------------------------

#[gpui::test]
fn settled_posts_are_articles_carrying_their_text(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let stores = ready_stores(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, Some("s".into()), WindowInput::new(cx), window, cx))
    });
    let space = view.read_with(cx, |v, _| v.space().clone());

    let mut timed = probe_post("a2", "*Republic* I, where the argument turns.");
    timed.parent_action_id = Some("a1".into());
    // 9:05 AM UTC — `fmt_clock` is timezone-free, so the byline time is
    // deterministic on every machine.
    timed.created_at = 9 * 3_600_000 + 5 * 60_000;
    cx.update(|cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![probe_post("a1", "the quick brown fox"), timed], cx)
        });
    });

    let entries = fresh_entries(cx, window);
    // An untimed row names its author alone; the value is the whole post.
    assert_probe_value(
        &entries,
        "space/post/0",
        gpui::Role::Article,
        "You",
        "the quick brown fox",
    );
    // Byline and time on one line — the gutter's stacked, node-less text.
    // Markdown is punctuation to the ear, so the value is spoken prose.
    assert_probe_value(
        &entries,
        "space/post/1",
        gpui::Role::Article,
        "You · 9:05 AM",
        "Republic I, where the argument turns.",
    );

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn a_streaming_reply_is_not_an_article(cx: &mut TestAppContext) {
    // The §4 trap: a value bound to streaming text makes assistive technology
    // restart the whole reply on every token. A stream therefore contributes
    // no node at all until it finalizes into a settled post.
    let _guard = probes_on();

    let stores = ready_stores(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, Some("s".into()), WindowInput::new(cx), window, cx))
    });
    let space = view.read_with(cx, |v, _| v.space().clone());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![probe_post("a1", "the question")], cx);
            s.push_streaming_turn_for_test(
                Some("agent-b".into()),
                Some("a1".into()),
                Default::default(),
                cx,
            );
        });
    })
    .unwrap();
    cx.run_until_parked();
    space.read_with(cx, |s, _| {
        assert_eq!(s.streams().len(), 1, "a turn really is in flight");
    });

    let entries = fresh_entries(cx, window);
    let articles: Vec<&String> = entries
        .iter()
        .filter(|(_, e)| e.role == gpui::Role::Article)
        .map(|(n, _)| n)
        .collect();
    assert_eq!(
        articles,
        vec!["space/post/0"],
        "only the settled post is an article while a reply streams"
    );

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn space_failure_and_cascade_notices_are_alerts(cx: &mut TestAppContext) {
    // Both notices used to be three unexplained buttons — the message itself
    // was a node-less div. The message is the **value** (the channel the macOS
    // adapter announces from once `aria_live` exists upstream).
    let _guard = probes_on();

    let stores = ready_stores(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, Some("s".into()), WindowInput::new(cx), window, cx))
    });
    let space = view.read_with(cx, |v, _| v.space().clone());

    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![probe_post("a1", "the question")], cx);
            s.emit_cascade_paused_for_test(4, 4, "a1".into(), cx);
        });
    })
    .unwrap();
    cx.run_until_parked();
    let entries = fresh_entries(cx, window);
    assert_probe_value(
        &entries,
        "space/cascade",
        gpui::Role::Alert,
        "Replies paused",
        "Replies paused — the conversation reached its cascade limit (4). \
         Ask to continue.",
    );

    // A failure outranks the pause (both bottom-anchor; an error is more
    // urgent), so only the failure notice is in the tree afterwards.
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.push_streaming_turn_for_test(
                Some("agent-b".into()),
                Some("a1".into()),
                Default::default(),
                cx,
            );
            s.apply_turn_failure_for_test(
                "agent-b",
                "a1",
                eidola_app_core::error::AppError::Network {
                    message: "connection reset".into(),
                },
                cx,
            );
        });
    })
    .unwrap();
    cx.run_until_parked();
    let entries = fresh_entries(cx, window);
    assert_probe_value(
        &entries,
        "space/error",
        gpui::Role::Alert,
        "Request failed",
        "network error: connection reset",
    );
    assert!(
        !entries.iter().any(|(n, _)| n == "space/cascade"),
        "the failure notice replaces the pause notice"
    );

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn account_balance_pools_and_plan_sublines_carry_their_values(cx: &mut TestAppContext) {
    use eidola_gui::account::AccountView;

    let _guard = probes_on();

    let stores = stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        s.eidola_trust = Some(probe_eidola_trust());
        s.balances = Some(BalancesResult {
            available: 4_200_000,
            pools: vec![eidola_app_core::BalancePoolInfo {
                amount: 1_000_000,
                source: "purchase".into(),
                // No expiry: `humanize_expiry` reads the real clock, and its
                // buckets have their own unit tests.
                expires_at: None,
            }],
        });
        s.prices = vec![PriceInfo {
            id: "price_once".into(),
            product_name: "One-time".into(),
            product_description: None,
            amount_display: "$5".into(),
            recurrence: "".into(),
            credits: 5_000_000,
        }];
        s.backends = backends_fixture();
    });
    let (window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| AccountView::new(stores, window, cx))
    });

    let entries = fresh_entries(cx, window);
    assert_probe_value(
        &entries,
        "settings/account/balance",
        gpui::Role::Label,
        "Balance",
        "4,200,000 credits available",
    );
    assert_probe_value(
        &entries,
        "settings/account/pool/0",
        gpui::Role::Label,
        "Credit pool 1",
        "purchase — 1,000,000 credits",
    );
    // The plan row keeps its name/price name; the subline — credits and the
    // expiry disclosure that must stay visible at the point of purchase — is
    // the value.
    assert_probe_value(
        &entries,
        "settings/account/plan/0",
        gpui::Role::ListBoxOption,
        "One-time — $5",
        "5,000,000 credits, expire one year after purchase",
    );

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn the_build_version_is_readable_in_about_and_updates(cx: &mut TestAppContext) {
    use eidola_gui::about::AboutView;
    use eidola_gui::updates::UpdatesView;

    let _guard = probes_on();

    let version = env!("CARGO_PKG_VERSION");
    let (about, _view) = open_view(cx, |window, cx| cx.new(|cx| AboutView::new(window, cx)));
    let entries = fresh_entries(cx, about);
    assert_probe_value(
        &entries,
        "about/version",
        gpui::Role::Label,
        "Version",
        &format!("v{version}"),
    );

    let stores = ready_stores(cx);
    let (updates, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| UpdatesView::new(stores, window, cx))
    });
    let entries = fresh_entries(cx, updates);
    assert_probe_value(
        &entries,
        "updates/version",
        gpui::Role::Label,
        "This build",
        &format!("Eidola {version}"),
    );

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn hash_labels_are_short_enough_to_hear(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let stores = ready_stores(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| RecordView::new(stores, window, cx))
    });
    view.update(cx, |v, _| {
        v.set_attestations_for_test(
            vec![stub_attestation(
                "1122334455667788112233445566778811223344556677881122334455667788",
            )],
            false,
        );
    });

    let entries = fresh_entries(cx, window);
    assert_probe(
        &entries,
        "record/attestation/0",
        gpui::Role::ListItem,
        "Attestation 11223344…",
    );

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn disabled_probes_record_nothing(cx: &mut TestAppContext) {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    probe::set_probes_enabled(false);

    let stores = ready_stores(cx);
    let (window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, None, WindowInput::new(cx), window, cx))
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

/// Like [`fresh_names`], but keeping each recorded entry so a test can assert
/// the **role** and **label** a probe applied — not merely that it exists.
///
/// This is as deep as the regression gate reaches at the current gpui pin:
/// `Window`'s emitted AccessKit `TreeUpdate` is crate-private, so nothing here
/// (or in the driver) can observe tree *shape*, node parentage, or the aria
/// attributes that don't pass through `.probe()` / `.probe_value()` —
/// `aria_position_in_set`, `aria_size_of_set`, `aria_selected`. The probe
/// registry is a faithful enumeration of the node set with its roles, names and
/// (since wave C) **values**, and that is the whole seam. See `AGENTS.md` →
/// Accessibility for the removal trigger.
fn fresh_entries(
    cx: &mut TestAppContext,
    window: AnyWindowHandle,
) -> Vec<(String, probe::ProbeEntry)> {
    probe::clear_window(window.window_id().as_u64());
    draw(cx, window);
    probe::window_entries(window.window_id().as_u64())
}

/// Assert that `name` was recorded with the given role and label.
#[track_caller]
fn assert_probe(
    entries: &[(String, probe::ProbeEntry)],
    name: &str,
    role: gpui::Role,
    label: &str,
) {
    let Some((_, entry)) = entries.iter().find(|(n, _)| n == name) else {
        let names: Vec<&String> = entries.iter().map(|(n, _)| n).collect();
        panic!("probe {name:?} missing; recorded: {names:?}");
    };
    assert_eq!(entry.role, role, "probe {name:?} role");
    assert_eq!(entry.label.as_ref(), label, "probe {name:?} label");
}

/// Assert role, label **and** accessible value — the content channel wave C
/// added (`aria_value`). A `None` value here means the call site used the plain
/// `probe`, i.e. the content never reaches assistive technology.
#[track_caller]
fn assert_probe_value(
    entries: &[(String, probe::ProbeEntry)],
    name: &str,
    role: gpui::Role,
    label: &str,
    value: &str,
) {
    assert_probe(entries, name, role, label);
    let (_, entry) = entries.iter().find(|(n, _)| n == name).unwrap();
    assert_eq!(
        entry.value.as_ref().map(|v| v.as_ref()),
        Some(value),
        "probe {name:?} value"
    );
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

fn probe_config_state() -> ConfigState {
    ConfigState {
        default_template: "00000000-0000-7000-8000-000000000010".into(),
        has_account: true,
        has_account_secret: true,
        account_id: Some("00000000-0000-7000-8000-000000000111".into()),
        account_secret: Some("probe-account-secret".into()),
        domain_separator: "ACT-v1:eidola:inference:production:2026-03-05".into(),
        appearance: eidola_app_core::config::AppearanceSetting::System,
        time_of_day_tint: eidola_app_core::config::TimeOfDayTint::On,
        light_character: eidola_app_core::config::LightCharacter::Neutral,
        font_scale: 1.0,
        language: None,
    }
}

fn probe_eidola_trust() -> eidola_app_core::EidolaTrust {
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

fn ready_stores(cx: &mut TestAppContext) -> Stores {
    stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        s.eidola_trust = Some(probe_eidola_trust());
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
        s.prices = vec![
            PriceInfo {
                id: "price_month".into(),
                product_name: "Monthly".into(),
                product_description: Some("Recurring top-up".into()),
                amount_display: "$10".into(),
                recurrence: "/mo".into(),
                credits: 10_000_000,
            },
            PriceInfo {
                id: "price_once".into(),
                product_name: "One-time".into(),
                product_description: None,
                amount_display: "$5".into(),
                recurrence: "".into(),
                credits: 5_000_000,
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
        let handle: AnyWindowHandle = window.into();
        // The probe registry is a process-global keyed by numeric window id
        // (`probe::REGISTRY`), and fresh `TestAppContext`s recycle those ids —
        // so a *previous* probe test's window may have left entries under this
        // id (e.g. the ⌥-revealed `space/composer/post-quiet` recorded by
        // `space_composer_alt_reveals_post_quiet_probe`). Drop them here so each
        // test's window starts clean and `window_entries` reflects only what
        // *this* scene renders — otherwise a scene asserting an affordance is
        // *absent* could inherit it from an earlier test, a leak that surfaces
        // only under a particular test-scheduling order (green on macOS locally,
        // red on Linux CI). Every test draws (directly or via `fresh_names`)
        // before reading, so this window's real probes are re-recorded after.
        probe::clear_window(handle.window_id().as_u64());
        (handle, inner.expect("build closure produced a view"))
    })
}

#[gpui::test]
fn space_probes_record_composer_and_band(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let stores = ready_stores(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, None, WindowInput::new(cx), window, cx))
    });
    // Seed a post with a committed reply (a1 → a2) so a1's band carries the
    // fork "+" (a leaf has no "+" — its tail draft is the reply affordance). The
    // blank space's active root draft provides the floating composer probe.
    let space = view.read_with(cx, |v, _| v.space().clone());
    let mut a2 = probe_post("a2", "a committed reply");
    a2.parent_action_id = Some("a1".into());
    cx.update(|cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![probe_post("a1", "a seeded root post"), a2], cx)
        });
    });
    // Give the active draft content so the action gutter's Post verb reveals.
    let editor = view
        .read_with(cx, |v, _| v.composer_state_for_test())
        .expect("blank space opens with the composer");
    cx.update(|cx| {
        editor.update(cx, |e, cx| e.set_value("a draft".to_string(), cx));
    });
    draw(cx, window);

    let entries = probe::window_entries(window.window_id().as_u64());
    let names: Vec<&str> = entries.iter().map(|(n, _)| n.as_str()).collect();
    assert!(
        names.contains(&"space/composer"),
        "composer probe missing; recorded: {names:?}"
    );
    assert!(
        names.contains(&"space/band/add"),
        "band reply affordance probe missing; recorded: {names:?}"
    );
    // The action gutter: Post (the composer's one CTA — the model picker and
    // request panel are gone; who answers is Participants configuration).
    assert!(
        names.contains(&"space/composer/post"),
        "Post affordance probe missing; recorded: {names:?}"
    );
    assert!(
        !names.contains(&"space/composer/post-quiet"),
        "Post quietly is ⌥-revealed only; recorded: {names:?}"
    );
    assert!(
        !names
            .iter()
            .any(|n| n.starts_with("space/request-panel") || *n == "space/composer/model"),
        "the retired model picker must record no probes; recorded: {names:?}"
    );
    // The floating composer's separator doubles as its resize handle (this
    // scene's root draft floats off-path over the seeded branch).
    assert!(
        names.contains(&"space/composer/resize"),
        "floating composer resize-handle probe missing; recorded: {names:?}"
    );
    // The minimap is a navigable table of contents: a labelled Group of
    // per-node Buttons.
    assert!(
        names.contains(&"space/minimap"),
        "minimap group probe missing; recorded: {names:?}"
    );
    let map = entries.iter().find(|(n, _)| n == "space/minimap").unwrap();
    assert_eq!(map.1.label.as_ref(), "Conversation map");
    assert!(
        names.iter().any(|n| n.starts_with("space/minimap/cell/")),
        "minimap column probes missing; recorded: {names:?}"
    );

    let composer = &entries
        .iter()
        .find(|(n, _)| n == "space/composer")
        .unwrap()
        .1;
    assert_eq!(format!("{:?}", composer.role), "TextInput");
    assert_eq!(composer.label.as_ref(), "Message composer");

    probe::set_probes_enabled(false);
}

/// Task 28: the post context menu is a menu in the a11y tree, and every row
/// it offers is a probed, driver-clickable `MenuItem`.
#[gpui::test]
fn space_context_menu_probes_its_rows(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let stores = ready_stores(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| {
            SpaceView::new(
                stores,
                Some("demo".into()),
                WindowInput::new(cx),
                window,
                cx,
            )
        })
    });
    let space = view.read_with(cx, |v, _| v.space().clone());
    cx.update(|cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![probe_post("a1", "the quick brown fox")], cx)
        });
    });
    draw(cx, window);

    // Open the menu over the post with its whole body selected, so it offers
    // the read-only set in full.
    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| {
            let len = v
                .post_body_editor_for_test("a1")
                .map(|e| e.read(cx).value().len())
                .unwrap_or(0);
            v.select_in_post_for_test("a1", 0..len, cx);
            v.open_context_menu_for_test("a1", cx);
        });
    })
    .unwrap();
    cx.run_until_parked();

    let entries = fresh_entries(cx, window);
    assert_probe(
        &entries,
        "space/context-menu",
        gpui::Role::Menu,
        "Post menu",
    );
    for (slug, label) in [
        ("copy", "Copy"),
        ("quote", "Quote"),
        ("quote-in-reply", "Quote in Reply"),
        ("select-all", "Select All"),
    ] {
        assert_probe(
            &entries,
            &format!("space/context-menu/{slug}"),
            gpui::Role::MenuItem,
            label,
        );
    }

    probe::set_probes_enabled(false);
}

/// Task 29: while a turn waits for its **engine** to warm, the streaming leaf
/// leads with "Loading model…" — the same quiet line, in the same slot, as
/// "Thinking…". It is a readout (`Role::Label`), not a control, and it is
/// keyed on the *correlation*, not on "some model somewhere is loading": the
/// responding participant's effective model must be the one warming.
#[gpui::test]
fn space_streaming_turn_reads_loading_model_while_its_engine_warms(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let warming = |status: eidola_app_core::LocalModelStatus| eidola_app_core::LocalModelsState {
        engine_path: Some("/opt/eidola/llama-server".into()),
        external: Vec::new(),
        models: vec![eidola_app_core::LocalModelInfo {
            id: "gemma-4-E4B_q4_0-it@local".into(),
            slug: "gemma-4-E4B_q4_0-it".into(),
            display_name: "Gemma 4 E4B".into(),
            file_name: "gemma-4-E4B_q4_0-it.gguf".into(),
            size_bytes: Some(5_154_939_136),
            source_url: None,
            status,
            last_error: None,
            on_disk: true,
        }],
    };
    let scene = |cx: &mut TestAppContext, status: eidola_app_core::LocalModelStatus| {
        let stores = stub_stores(cx, |s| {
            s.config_state = Some(probe_config_state());
            s.local_models = Some(warming(status));
            let (space_id, mut people) = probe_participants();
            // The responding agent runs the engine-served model above.
            if let Some(agent) = people.iter_mut().find(|p| p.kind == "agent") {
                agent.model_ref = Some("gemma-4-E4B_q4_0-it@local".into());
            }
            s.participants = Some((space_id, people));
        });
        let (window, view) = open_view(cx, |window, cx| {
            cx.new(|cx| {
                SpaceView::new(
                    stores,
                    Some("demo".into()),
                    WindowInput::new(cx),
                    window,
                    cx,
                )
            })
        });
        let space = view.read_with(cx, |v, _| v.space().clone());
        cx.update(|cx| {
            space.update(cx, |s, cx| {
                s.set_post_tree_for_test(vec![probe_post("a1", "a seeded root post")], cx);
                s.push_streaming_turn_for_test(
                    Some("agent-1".into()),
                    Some("a1".into()),
                    Default::default(),
                    cx,
                );
            });
        });
        window
    };

    let window = scene(cx, eidola_app_core::LocalModelStatus::Loading);
    let entries = fresh_entries(cx, window);
    assert_probe(
        &entries,
        "space/streaming/1/loading",
        gpui::Role::Label,
        "Loading model…",
    );

    // Once the engine is serving, the readout is gone — the silence is the
    // model thinking, and saying otherwise would be a lie.
    let window = scene(
        cx,
        eidola_app_core::LocalModelStatus::Loaded {
            port: 51_432,
            context_tokens: 8192,
            pinned: false,
        },
    );
    let names = fresh_names(cx, window);
    assert!(
        !names.iter().any(|n| n.ends_with("/loading")),
        "a loaded engine must show no loading readout; recorded: {names:?}"
    );

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn space_composer_alt_reveals_post_quiet_probe(cx: &mut TestAppContext) {
    // Holding ⌥ reveals the quiet verb (post without notifying anyone) beside
    // Post — the "Option reveals power" expansion, now model-free.
    let _guard = probes_on();

    let stores = ready_stores(cx);
    let (window, _view) = open_view(cx, |window, cx| {
        let wi = WindowInput::new(cx);
        wi.update(cx, |w, cx| w.set_alt_for_test(true, cx));
        cx.new(|cx| SpaceView::new(stores, None, wi, window, cx))
    });
    draw(cx, window);

    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"space/composer/post".to_string()),
        "Post probe missing under ⌥; recorded: {names:?}"
    );
    assert!(
        names.contains(&"space/composer/post-quiet".to_string()),
        "⌥ must reveal Post quietly; recorded: {names:?}"
    );
    // This scene's blank-space composer is docked (page geometry, not a
    // floating pane), so it offers no resize handle.
    assert!(
        !names.contains(&"space/composer/resize".to_string()),
        "a docked composer must offer no resize handle; recorded: {names:?}"
    );

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn space_error_notice_exposes_recovery_probes(cx: &mut TestAppContext) {
    // After a failed ask, the recovery notice exposes its three affordances
    // (Retry / Copy / Dismiss) as probes — the driver click targets and the
    // AccessKit annotations.
    let _guard = probes_on();

    let stores = ready_stores(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, None, WindowInput::new(cx), window, cx))
    });

    // Seed a saved user post, then drive one turn's failure (which records
    // the failed turn — who was asked, about what — so Retry renders alongside
    // Copy / Dismiss).
    let space = view.read_with(cx, |v, _| v.space().clone());
    cx.update(|cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![probe_post("a1", "Hello, what is your name?")], cx)
        });
        space.update(cx, |s, cx| {
            s.apply_turn_failure_for_test(
                "agent-1",
                "a1",
                eidola_app_core::error::AppError::Network {
                    message: "dns error".into(),
                },
                cx,
            )
        });
    });
    draw(cx, window);

    let entries = probe::window_entries(window.window_id().as_u64());
    let names: Vec<&str> = entries.iter().map(|(n, _)| n.as_str()).collect();
    for expected in [
        "space/error/dismiss",
        "space/error/copy",
        "space/error/retry",
    ] {
        assert!(
            names.contains(&expected),
            "{expected} probe missing; recorded: {names:?}"
        );
    }

    probe::set_probes_enabled(false);
}

fn probe_post(action_id: &str, text: &str) -> PostNode {
    PostNode {
        action_id: action_id.into(),
        item_id: format!("item-{action_id}"),
        parent_action_id: None,
        participant: PostParticipant {
            kind: "human".into(),
            label: "user".into(),
        },
        action_type: "user_input".into(),
        generation: 0,
        generation_count: 1,
        is_current: true,
        model: None,
        credits_consumed: None,
        relation: None,
        depth: 0,
        is_branch: false,
        blocks: vec![PostBlock {
            id: String::new(),
            block_type: "text".into(),
            text: Some(text.into()),
            tool_name: None,
            tool_call_id: None,
            data: None,
        }],
        references: Vec::new(),
        created_at: 0,
    }
}

#[gpui::test]
fn onboarding_probes_record_ctas_and_inputs(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let stores = ready_stores(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| OnboardingView::new(stores, window, cx))
    });
    draw(cx, window);

    // The first slide's call-to-action is probed for AT + the driver.
    let entries = probe::window_entries(window.window_id().as_u64());
    let names: Vec<&str> = entries.iter().map(|(n, _)| n.as_str()).collect();
    assert!(
        names.contains(&"onboarding/cta/pause"),
        "first-slide CTA probe missing; recorded: {names:?}"
    );

    // Walk to the existing-account branch, which reveals the credential inputs
    // and the verify CTA.
    view.update(cx, |v, cx| {
        v.reveal(Slide::Pause, Slide::Tool, cx);
        v.reveal(Slide::Tool, Slide::Control, cx);
        v.reveal(Slide::Control, Slide::Responsibility, cx);
        v.reveal(Slide::Responsibility, Slide::GetStarted, cx);
        v.reveal(Slide::GetStarted, Slide::ExistingAccount, cx);
    });
    draw(cx, window);

    let entries = probe::window_entries(window.window_id().as_u64());
    let names: Vec<&str> = entries.iter().map(|(n, _)| n.as_str()).collect();
    for expected in [
        "onboarding/input/account-id",
        "onboarding/input/account-secret",
        "onboarding/cta/verify",
    ] {
        assert!(
            names.contains(&expected),
            "existing-account probe {expected:?} missing; recorded: {names:?}"
        );
    }
    // Every revealed slide past the first carries a "back" affordance (the
    // visible up-arrow alternative to the scroll-back gesture); the first slide
    // (index 0) does not.
    assert!(
        names.iter().any(|n| n.starts_with("onboarding/back/")),
        "back-arrow probe missing on a non-first slide; recorded: {names:?}"
    );
    assert!(
        !names.contains(&"onboarding/back/0"),
        "the first slide must not carry a back arrow; recorded: {names:?}"
    );

    // Re-choose the new-account branch: the create slide carries the required
    // agreement checkbox and the (checkbox-gated) create CTA.
    view.update(cx, |v, cx| {
        v.reveal(Slide::GetStarted, Slide::CreateAccount, cx);
    });
    draw(cx, window);

    let entries = probe::window_entries(window.window_id().as_u64());
    let names: Vec<&str> = entries.iter().map(|(n, _)| n.as_str()).collect();
    for expected in ["onboarding/agree", "onboarding/cta/create"] {
        assert!(
            names.contains(&expected),
            "create-account probe {expected:?} missing; recorded: {names:?}"
        );
    }
    // The Control slide's repository link is scoped per-label (the former
    // shared `onboarding/link` name collided across every link).
    assert!(
        names
            .iter()
            .any(|n| n.starts_with("onboarding/link/") && n.contains("repository")),
        "scoped repository link probe missing; recorded: {names:?}"
    );

    // Walk to the Purchase slide (via the existing-account branch) and assert
    // the shared plans component annotates each row under the onboarding scope.
    view.update(cx, |v, cx| {
        v.reveal(Slide::GetStarted, Slide::ExistingAccount, cx);
        v.reveal(Slide::ExistingAccount, Slide::Purchase, cx);
    });
    draw(cx, window);

    let entries = probe::window_entries(window.window_id().as_u64());
    let names: Vec<&str> = entries.iter().map(|(n, _)| n.as_str()).collect();
    assert!(
        names.contains(&"onboarding/plans"),
        "onboarding plans listbox probe missing; recorded: {names:?}"
    );
    assert!(
        names.contains(&"onboarding/plan/0"),
        "onboarding plan-row probe missing; recorded: {names:?}"
    );
}

// ---------------------------------------------------------------------------
// Record window — the raw local trail. Listing rows, section tabs, refresh,
// and the load-more affordance are all probed (indexed row names so a driver
// can address "the third attestation" precisely).
// ---------------------------------------------------------------------------

fn stub_attestation(hash: &str) -> AttestationInfo {
    AttestationInfo {
        hash: hash.into(),
        pcr_digest: Some("pcr-abc".into()),
        created_at: 1_000,
        doc_bytes: 2_048,
        connection_count: 3,
    }
}

fn stub_request(id: &str) -> RequestInfo {
    RequestInfo {
        id: id.into(),
        method: "POST".into(),
        path: "/v1/chat/completions".into(),
        response_status: Some(200),
        duration_ms: Some(742),
        request_at: 1_000,
        error: None,
        attempt_number: 1,
        credential_nonce: Some("nonce-1".into()),
        transport: Some("clearnet".into()),
        base_url: Some("https://eidola.example".into()),
        attestation_hash: Some("att-1".into()),
    }
}

fn stub_spend(request_id: &str) -> SpendTrailEntry {
    SpendTrailEntry {
        credential_nonce: "nonce-1".into(),
        spend_amount: Some(1_000),
        credential_state: "spent".into(),
        request_id: request_id.into(),
        method: "POST".into(),
        path: "/v1/chat/completions".into(),
        request_at: 1_000,
        duration_ms: Some(742),
        attempt_number: 1,
        action_id: Some("act-1".into()),
        action_type: Some("inference".into()),
        model: Some("gemma4-31b".into()),
        credits_consumed: Some(950),
        intent: Some("chat".into()),
        space_id: Some("space-1".into()),
        space_title: Some("A space".into()),
        linkability: None,
    }
}

#[gpui::test]
fn listing_rows_are_positioned_among_data_rows_only(cx: &mut TestAppContext) {
    // `aria_position_in_set` / `aria_size_of_set` don't pass through `.probe()`,
    // so the view exposes what it would report. The trap this guards: the
    // display model interleaves spending group headers and a trailing
    // load-more row, so positioning by *display* index announces the first
    // spending entry as "2 of N" and counts the button as an extra item.
    let _guard = probes_on();

    let stores = ready_stores(cx);
    let (_window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| RecordView::new(stores, window, cx))
    });

    // Attestations with `has_more`: the load-more row must not inflate the set.
    view.update(cx, |v, _| {
        v.set_attestations_for_test(
            vec![stub_attestation("att-1"), stub_attestation("att-2")],
            true,
        );
    });
    view.read_with(cx, |v, _| {
        assert_eq!(v.row_set_metadata_for_test(), vec![(1, 2), (2, 2)]);
    });

    // Spending groups by credential: two nonces means two interleaved headers,
    // and the data rows must still read 1..=N contiguously.
    view.update(cx, |v, cx| {
        v.select_section(RecordSection::Spending, cx);
        let mut second = stub_spend("req-2");
        second.credential_nonce = "nonce-2".into();
        let mut third = stub_spend("req-3");
        third.credential_nonce = "nonce-2".into();
        v.set_spending_for_test(vec![stub_spend("req-1"), second, third], true);
    });
    view.read_with(cx, |v, _| {
        assert_eq!(v.row_set_metadata_for_test(), vec![(1, 3), (2, 3), (3, 3)]);
    });

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn record_probes_cover_rows_tabs_and_chrome(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let stores = ready_stores(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| RecordView::new(stores, window, cx))
    });

    // Attestations section: rows have `has_more` so the load-more affordance
    // renders too.
    view.update(cx, |v, _| {
        v.set_attestations_for_test(vec![stub_attestation("att-hash-1")], true);
    });
    let names = fresh_names(cx, window);
    for expected in [
        "record/section/attestations",
        "record/section/requests",
        "record/section/spending",
        "record/refresh",
        "record/attestation/0",
        "record/load-more",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "record probe {expected:?} missing; recorded: {names:?}"
        );
    }

    // Requests section.
    view.update(cx, |v, cx| {
        v.select_section(RecordSection::Requests, cx);
        v.set_requests_for_test(vec![stub_request("req-1")], false);
    });
    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"record/request/0".to_string()),
        "record request-row probe missing; recorded: {names:?}"
    );

    // Spending section.
    view.update(cx, |v, cx| {
        v.select_section(RecordSection::Spending, cx);
        v.set_spending_for_test(vec![stub_spend("req-1")], false);
    });
    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"record/spend/0".to_string()),
        "record spend-row probe missing; recorded: {names:?}"
    );

    probe::set_probes_enabled(false);
}

// ---------------------------------------------------------------------------
// Settings cluster — the Account and Wallet panes and General's affordances.
// ---------------------------------------------------------------------------

#[gpui::test]
fn account_pane_probes_cover_controls_and_plans(cx: &mut TestAppContext) {
    use eidola_gui::account::AccountView;

    let _guard = probes_on();

    // Account is a top-level Settings pane again (shown while the eidola
    // backend is enabled). Its `settings/account/*` probes stay stable.
    let stores = account_backends_stores(cx);
    let (window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| AccountView::new(stores, window, cx))
    });

    let names = fresh_names(cx, window);
    for expected in [
        "settings/account/reset",
        "settings/account/refresh-balances",
        // The shared plans component, scoped to the Account host.
        "settings/account/plans",
        "settings/account/plan/0",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "account pane probe {expected:?} missing; recorded: {names:?}"
        );
    }

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn the_credential_fields_and_their_suffix_controls_are_each_one_node(cx: &mut TestAppContext) {
    use eidola_gui::account::AccountView;

    // The two credential fields follow the focus-bearing-editor regime — the
    // `Input` carries the label and is the node, its wrapper is bounds-only —
    // and the controls sitting in the secret field's suffix follow the hoist:
    // a probed wrapper is the control, so each is a named, reachable node
    // rather than a presentational widget nobody can get to. `.role(None)` on
    // the widgets alone would satisfy the source scan while deleting these
    // from the tree, which is precisely what this test refuses.
    let _guard = probes_on();

    let stores = account_backends_stores(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| AccountView::new(stores, window, cx))
    });

    let entries = fresh_entries(cx, window);
    assert_probe(
        &entries,
        "settings/account/id",
        gpui::Role::TextInput,
        "Account ID",
    );
    assert_probe(
        &entries,
        "settings/account/secret",
        gpui::Role::TextInput,
        "Account Secret",
    );
    assert_probe(
        &entries,
        "settings/account/id/copy",
        gpui::Role::Button,
        "Copy Account ID",
    );
    assert_probe(
        &entries,
        "settings/account/secret/copy",
        gpui::Role::Button,
        "Copy Account Secret",
    );
    assert_probe(
        &entries,
        "settings/account/secret/reveal",
        gpui::Role::Button,
        "Show account secret",
    );
    // The component paints twice and derives every name and element id from
    // its prefix, so the two copies are distinct nodes — and only the secret
    // has anything to reveal.
    assert!(
        !entries
            .iter()
            .any(|(n, _)| n == "settings/account/id/reveal"),
        "the account id is never masked; recorded: {:?}",
        entries.iter().map(|(n, _)| n).collect::<Vec<_>>()
    );

    // Revealed, the control says what it would now do.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.toggle_account_secret_revealed(window, cx));
    })
    .unwrap();
    let entries = fresh_entries(cx, window);
    assert_probe(
        &entries,
        "settings/account/secret/reveal",
        gpui::Role::Button,
        "Hide account secret",
    );

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn a_lapsed_account_is_offered_its_billing_and_every_plan(cx: &mut TestAppContext) {
    use eidola_gui::account::AccountView;

    // A customer with nothing in force still has a payment relationship —
    // a saved card, invoices, receipts — so the portal door stays, worded
    // for what is actually behind it rather than for a subscription that
    // does not exist. Every plan stays offered: there is nothing to
    // double-subscribe over.
    let _guard = probes_on();

    let stores = account_backends_stores(cx);
    stores.account.update(cx, |s, cx| {
        s.set_subscription_for_test(
            eidola_gui::loadable::Loadable::loaded(eidola_app_core::SubscriptionInfo {
                state: eidola_app_core::SubscriptionState::Inactive,
                status: None,
                current_period_end: None,
            }),
            cx,
        );
    });
    let (window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| AccountView::new(stores, window, cx))
    });

    let names = fresh_names(cx, window);
    for expected in [
        "settings/account/subscription",
        "settings/account/billing-portal",
        "settings/account/subscription-retry",
        "settings/account/plan/0",
        "settings/account/plan/1",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "lapsed account probe {expected:?} missing; recorded: {names:?}"
        );
    }
    assert!(
        !names.contains(&"settings/account/manage-subscription".to_string()),
        "there is no subscription to manage; recorded: {names:?}"
    );

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn an_account_that_never_transacted_is_shown_no_billing_door(cx: &mut TestAppContext) {
    use eidola_gui::account::AccountView;

    // No payment customer means no relationship to be let into — a portal
    // session minted for someone money has never moved for would be a door
    // onto an empty room. The *answer* is still owed, and so is the way to
    // ask again: a reader who completes their first checkout is in exactly
    // this state with this pane already open in front of them.
    let _guard = probes_on();

    let stores = account_backends_stores(cx);
    stores.account.update(cx, |s, cx| {
        s.set_subscription_for_test(
            eidola_gui::loadable::Loadable::loaded(eidola_app_core::SubscriptionInfo {
                state: eidola_app_core::SubscriptionState::NoCustomer,
                status: None,
                current_period_end: None,
            }),
            cx,
        );
    });
    let (window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| AccountView::new(stores, window, cx))
    });

    let names = fresh_names(cx, window);
    // The answer, and the door back to it.
    for expected in [
        "settings/account/subscription",
        "settings/account/subscription-retry",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "an account with no payment customer is still owed {expected:?}; \
             recorded: {names:?}"
        );
    }
    // But no billing door of either wording.
    for absent in [
        "settings/account/billing-portal",
        "settings/account/manage-subscription",
    ] {
        assert!(
            !names.contains(&absent.to_string()),
            "an account with no payment customer must show no {absent:?}; \
             recorded: {names:?}"
        );
    }
    // The plans are the whole surface for someone who has never transacted.
    for expected in ["settings/account/plan/0", "settings/account/plan/1"] {
        assert!(
            names.contains(&expected.to_string()),
            "plan probe {expected:?} missing; recorded: {names:?}"
        );
    }

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn a_known_subscription_can_always_be_re_checked(cx: &mut TestAppContext) {
    use eidola_gui::account::AccountView;

    // The reader changes their subscription in a browser window this app
    // never hears about and comes back to a Settings window that never
    // closed. Nothing on the bus can invalidate the cell, so a re-check
    // offered only after a *failed* read is a door that opens only once it
    // is already too late.
    let _guard = probes_on();

    let stores = account_backends_stores(cx);
    stores.account.update(cx, |s, cx| {
        s.set_subscription_for_test(
            eidola_gui::loadable::Loadable::loaded(eidola_app_core::SubscriptionInfo {
                state: eidola_app_core::SubscriptionState::Active,
                status: Some("active".into()),
                current_period_end: None,
            }),
            cx,
        );
    });
    let (window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| AccountView::new(stores, window, cx))
    });

    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"settings/account/subscription-retry".to_string()),
        "a successfully read subscription must still offer a re-check; \
         recorded: {names:?}"
    );

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn a_subscribed_account_is_offered_management_and_only_one_time_plans(cx: &mut TestAppContext) {
    use eidola_gui::account::AccountView;

    let _guard = probes_on();

    let stores = account_backends_stores(cx);
    stores.account.update(cx, |s, cx| {
        s.set_subscription_for_test(
            eidola_gui::loadable::Loadable::loaded(eidola_app_core::SubscriptionInfo {
                state: eidola_app_core::SubscriptionState::Active,
                status: Some("active".into()),
                current_period_end: None,
            }),
            cx,
        );
    });
    let (window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| AccountView::new(stores, window, cx))
    });

    let names = fresh_names(cx, window);
    for expected in [
        "settings/account/subscription",
        "settings/account/manage-subscription",
        // One-time top-ups stay purchasable alongside the subscription.
        "settings/account/plan/0",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "subscribed account probe {expected:?} missing; recorded: {names:?}"
        );
    }
    assert!(
        !names.contains(&"settings/account/plan/1".to_string()),
        "the recurring plan should be gone — the server refuses a second \
         subscription; recorded: {names:?}"
    );

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn a_failed_subscription_read_offers_a_retry_and_withholds_no_plans(cx: &mut TestAppContext) {
    use eidola_gui::account::AccountView;

    let _guard = probes_on();

    let stores = account_backends_stores(cx);
    stores.account.update(cx, |s, cx| {
        s.set_subscription_for_test(
            eidola_gui::loadable::Loadable::Failed {
                error: AppError::Network {
                    message: "connection reset".into(),
                },
                prior: None,
            },
            cx,
        );
    });
    let (window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| AccountView::new(stores, window, cx))
    });

    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"settings/account/subscription-retry".to_string()),
        "a failed read must offer its way back; recorded: {names:?}"
    );
    // Not knowing is not knowing: every plan stays offered, and the server
    // refuses honestly if one of them is not allowed.
    for expected in ["settings/account/plan/0", "settings/account/plan/1"] {
        assert!(
            names.contains(&expected.to_string()),
            "plan probe {expected:?} missing; recorded: {names:?}"
        );
    }

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn a_failed_re_read_keeps_the_subscription_it_already_knew(cx: &mut TestAppContext) {
    use eidola_gui::account::AccountView;

    let _guard = probes_on();

    let stores = account_backends_stores(cx);
    stores.account.update(cx, |s, cx| {
        s.set_subscription_for_test(
            eidola_gui::loadable::Loadable::Failed {
                error: AppError::Network {
                    message: "connection reset".into(),
                },
                prior: Some(eidola_app_core::SubscriptionInfo {
                    state: eidola_app_core::SubscriptionState::Active,
                    status: Some("active".into()),
                    current_period_end: None,
                }),
            },
            cx,
        );
    });
    let (window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| AccountView::new(stores, window, cx))
    });

    let names = fresh_names(cx, window);
    // The known answer stays on screen — a failed refresh never blanks a
    // cell that has data — with the quiet way to ask again beside it.
    for expected in [
        "settings/account/subscription",
        "settings/account/manage-subscription",
        "settings/account/subscription-retry",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "stale-subscription probe {expected:?} missing; recorded: {names:?}"
        );
    }
    assert!(
        !names.contains(&"settings/account/plan/1".to_string()),
        "the stale answer still governs what is offered; recorded: {names:?}"
    );

    probe::set_probes_enabled(false);
}

/// Stores with a linked account (balance + plans) *and* an eidola-enabled
/// registry.
fn account_backends_stores(cx: &mut TestAppContext) -> Stores {
    stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        s.eidola_trust = Some(probe_eidola_trust());
        s.balances = Some(BalancesResult {
            available: 4_200_000,
            pools: Vec::new(),
        });
        s.prices = vec![
            PriceInfo {
                id: "price_month".into(),
                product_name: "Monthly".into(),
                product_description: Some("Recurring top-up".into()),
                amount_display: "$10".into(),
                recurrence: "/mo".into(),
                credits: 10_000_000,
            },
            PriceInfo {
                id: "price_once".into(),
                product_name: "One-time".into(),
                product_description: None,
                amount_display: "$5".into(),
                recurrence: "".into(),
                credits: 5_000_000,
            },
        ];
        s.backends = backends_fixture();
    })
}

#[gpui::test]
fn wallet_pane_probes_cover_refresh(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let stores = ready_stores(cx);
    let (window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| WalletView::new(stores, window, cx))
    });

    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"settings/wallet/refresh".to_string()),
        "wallet refresh probe missing; recorded: {names:?}"
    );

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn general_pane_probes_cover_appearance_and_startup(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let stores = ready_stores(cx);
    let (window, _view) = open_view(cx, move |window, cx| {
        cx.new(|cx| GeneralView::new(stores.config.clone(), window, cx))
    });

    // The pane is the appearance chips plus the one startup row — every
    // trust / connection affordance lives in Backends → Eidola now.
    let names = fresh_names(cx, window);
    for expected in [
        "settings/general/appearance/system",
        "settings/general/time-of-day/on",
        "settings/general/login-item",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "general probe {expected:?} missing; recorded: {names:?}"
        );
    }
    for absent in [
        // The base-URL editor and the trust summaries moved to Backends →
        // Eidola; the Advanced disclosure is gone entirely.
        "settings/general/change",
        "settings/general/advanced/toggle",
        "settings/general/open-record",
    ] {
        assert!(
            !names.contains(&absent.to_string()),
            "General must carry no trust/connection affordance: {absent:?} in {names:?}"
        );
    }

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn band_menu_probes_appear_on_open_and_clear_on_dismiss(cx: &mut TestAppContext) {
    // The separator band's Reply-or-Ask menu (the request panel's successor):
    // opening a band's "+" mounts the menu probes — Reply (a post with a
    // committed reply) and one Ask per agent participant — and dismissing
    // clears them (stale entries would be ghost click targets for the driver).
    let _guard = probes_on();

    let stores = stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        s.eidola_trust = Some(probe_eidola_trust());
        s.participants = Some(probe_participants());
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| {
            SpaceView::new(
                stores,
                Some("demo".into()),
                WindowInput::new(cx),
                window,
                cx,
            )
        })
    });
    // a1 has a committed reply (a2), so its band offers Reply *and* Ask.
    let space = view.read_with(cx, |v, _| v.space().clone());
    let mut a2 = probe_post("a2", "a committed reply");
    a2.parent_action_id = Some("a1".into());
    cx.update(|cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![probe_post("a1", "a seeded root post"), a2], cx)
        });
    });

    // Closed: no menu probes; the band's "+" advertises both verbs.
    draw(cx, window);
    let names = fresh_names(cx, window);
    assert!(
        !names.iter().any(|n| n.starts_with("space/band/menu")),
        "menu probes before opening: {names:?}"
    );
    assert!(
        names.contains(&"space/band/add".to_string()),
        "band + affordance missing: {names:?}"
    );

    // Open a1's menu.
    cx.update(|cx| {
        view.update(cx, |v, cx| v.set_band_menu_for_test(Some("a1"), cx));
    });
    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"space/band/menu".to_string()),
        "menu probe missing after open: {names:?}"
    );
    assert!(
        names.contains(&"space/band/menu/reply".to_string()),
        "Reply probe missing (a1 has a committed reply): {names:?}"
    );
    assert!(
        names.contains(&"space/band/menu/ask/0".to_string()),
        "per-agent Ask probe missing: {names:?}"
    );

    // Dismiss: the clear-then-redraw dance must drop the unmounted menu.
    cx.update(|cx| {
        view.update(cx, |v, cx| v.set_band_menu_for_test(None, cx));
    });
    let names = fresh_names(cx, window);
    assert!(
        !names.iter().any(|n| n.starts_with("space/band/menu")),
        "menu probes must clear after dismiss: {names:?}"
    );
    assert!(
        names.contains(&"space/band/add".to_string()),
        "still-mounted probes must survive the refresh: {names:?}"
    );
}

#[gpui::test]
fn cascade_notice_probes_expose_ask_and_dismiss(cx: &mut TestAppContext) {
    // The cascade-paused notice: quiet, dismissible, with one "Ask <agent>"
    // chip per agent participant — all probed for the driver + AccessKit.
    let _guard = probes_on();

    let stores = stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        s.eidola_trust = Some(probe_eidola_trust());
        s.participants = Some(probe_participants());
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| {
            SpaceView::new(
                stores,
                Some("demo".into()),
                WindowInput::new(cx),
                window,
                cx,
            )
        })
    });
    let space = view.read_with(cx, |v, _| v.space().clone());
    cx.update(|cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![probe_post("a1", "the question")], cx);
            s.emit_cascade_paused_for_test(4, 4, "a1".into(), cx);
        });
    });
    draw(cx, window);

    let entries = probe::window_entries(window.window_id().as_u64());
    let names: Vec<&str> = entries.iter().map(|(n, _)| n.as_str()).collect();
    assert!(
        names.contains(&"space/cascade/dismiss"),
        "cascade dismiss probe missing; recorded: {names:?}"
    );
    assert!(
        names.contains(&"space/cascade/ask/0"),
        "cascade per-agent ask probe missing; recorded: {names:?}"
    );
    let ask = entries
        .iter()
        .find(|(n, _)| n == "space/cascade/ask/0")
        .unwrap();
    assert_eq!(ask.1.label.as_ref(), "Ask Assistant to continue");

    probe::set_probes_enabled(false);
}

/// Linux window chrome: `ChromeRoot` wraps every production window and hosts
/// the primary menu (the macOS app/Space-menu replacement). The wordmark
/// affordance must probe at rest; toggling the menu (the F10 action path)
/// must mount the panel + item probes; toggling again must clear them (no
/// ghost targets). macOS is excluded because `ChromeRoot::wrap` is an
/// identity there — there is no chrome layer to probe.
#[cfg(not(target_os = "macos"))]
#[gpui::test]
fn chrome_menu_probes_and_toggle(cx: &mut TestAppContext) {
    use eidola_gui::chrome::{ChromeRoot, TogglePrimaryMenu};

    let _guard = probes_on();

    let stores = ready_stores(cx);
    let mut inner: Option<Entity<SpaceView>> = None;
    let window: AnyWindowHandle = cx.update(|cx| {
        gpui_component::init(cx);
        eidola_gui::theme::install(cx);
        cx.open_window(WindowOptions::default(), |window, cx| {
            let view =
                cx.new(|cx| SpaceView::new(stores.clone(), None, WindowInput::new(cx), window, cx));
            inner = Some(view.clone());
            let chrome = ChromeRoot::wrap(view.into(), cx);
            cx.new(|cx| Root::new(chrome, window, cx))
        })
        .expect("open test window")
        .into()
    });
    let view = inner.expect("built view");

    // At rest: the wordmark affordance probes; the panel does not.
    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"chrome/menu".to_string()),
        "menu wordmark probe missing: {names:?}"
    );
    assert!(
        !names.iter().any(|n| n.starts_with("chrome/menu/")),
        "menu panel probes before opening: {names:?}"
    );

    // Toggle open via the real action dispatch path (what F10 does). The
    // handler lives on ChromeRoot's wrapper div — an ancestor of the focused
    // view — so dispatching through the view's focus handle exercises the
    // production dispatch route.
    let focus = view.read_with(cx, |v, _| v.focus_handle());
    cx.update_window(window, |_, window, cx| {
        focus.dispatch_action(&TogglePrimaryMenu, window, cx);
    })
    .unwrap();
    cx.run_until_parked();

    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"chrome/menu/panel".to_string()),
        "menu panel probe missing after toggle: {names:?}"
    );
    for item in [
        "chrome/menu/new-space",
        "chrome/menu/library",
        "chrome/menu/record",
        "chrome/menu/settings",
        "chrome/menu/updates",
        "chrome/menu/about",
        "chrome/menu/quit",
    ] {
        assert!(
            names.contains(&item.to_string()),
            "menu item probe {item} missing: {names:?}"
        );
    }

    // Toggle closed: panel probes must clear.
    cx.update_window(window, |_, window, cx| {
        focus.dispatch_action(&TogglePrimaryMenu, window, cx);
    })
    .unwrap();
    cx.run_until_parked();

    let names = fresh_names(cx, window);
    assert!(
        !names.iter().any(|n| n.starts_with("chrome/menu/")),
        "menu panel probes must clear after dismiss: {names:?}"
    );
}

/// The F10 *keystroke* (not just the action) must open the primary menu:
/// exercises the global `f10 → TogglePrimaryMenu` binding through the real
/// keymap + dispatch path with the composer focused, as a user would hit it.
#[cfg(not(target_os = "macos"))]
#[gpui::test]
fn chrome_menu_opens_on_f10_keystroke(cx: &mut TestAppContext) {
    use eidola_gui::chrome::ChromeRoot;

    let _guard = probes_on();

    let stores = ready_stores(cx);
    let window: AnyWindowHandle = cx.update(|cx| {
        gpui_component::init(cx);
        eidola_gui::theme::install(cx);
        eidola_gui::install_keybindings(cx);
        cx.open_window(WindowOptions::default(), |window, cx| {
            let view =
                cx.new(|cx| SpaceView::new(stores.clone(), None, WindowInput::new(cx), window, cx));
            let chrome = ChromeRoot::wrap(view.into(), cx);
            cx.new(|cx| Root::new(chrome, window, cx))
        })
        .expect("open test window")
        .into()
    });
    draw(cx, window);

    cx.simulate_keystrokes(window, "f10");
    cx.run_until_parked();

    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"chrome/menu/panel".to_string()),
        "F10 keystroke must open the primary menu: {names:?}"
    );
}

/// The View → zoom *keystrokes* must resolve through the production keymap to
/// their actions with the composer focused, as a user would hit them. Guards
/// the `secondary-0 / secondary-= / secondary-+ / secondary--` binding strings
/// end to end: an invalid keystroke string would either panic
/// `install_keybindings` (run in setup) or silently bind nothing (the
/// assertions below would then fail). macOS-only because the chords are ⌘-based.
#[cfg(target_os = "macos")]
#[gpui::test]
fn zoom_keystrokes_resolve_to_actions(cx: &mut TestAppContext) {
    use eidola_gui::actions::{ActualSize, ZoomIn, ZoomOut};
    use std::cell::RefCell;
    use std::rc::Rc;

    let _guard = probes_on();

    let fired: Rc<RefCell<Vec<&'static str>>> = Rc::new(RefCell::new(Vec::new()));
    let stores = ready_stores(cx);
    let window: AnyWindowHandle = cx.update(|cx| {
        gpui_component::init(cx);
        eidola_gui::theme::install(cx);
        eidola_gui::install_keybindings(cx);
        // Global test handlers standing in for the production ones (which need
        // `AppGlobal`); they only record which action each keystroke dispatched.
        {
            let f = fired.clone();
            cx.on_action(move |_: &ActualSize, _| f.borrow_mut().push("actual"));
        }
        {
            let f = fired.clone();
            cx.on_action(move |_: &ZoomIn, _| f.borrow_mut().push("in"));
        }
        {
            let f = fired.clone();
            cx.on_action(move |_: &ZoomOut, _| f.borrow_mut().push("out"));
        }
        cx.open_window(WindowOptions::default(), |window, cx| {
            let view =
                cx.new(|cx| SpaceView::new(stores.clone(), None, WindowInput::new(cx), window, cx));
            cx.new(|cx| Root::new(view, window, cx))
        })
        .expect("open test window")
        .into()
    });
    draw(cx, window);

    cx.simulate_keystrokes(window, "cmd-0");
    cx.simulate_keystrokes(window, "cmd-=");
    cx.simulate_keystrokes(window, "cmd-+");
    cx.simulate_keystrokes(window, "cmd--");
    cx.run_until_parked();

    assert_eq!(
        *fired.borrow(),
        vec!["actual", "in", "in", "out"],
        "⌘0/⌘=/⌘+/⌘- must dispatch Actual Size / Zoom In (×2) / Zoom Out"
    );
}

/// A real pointer click on the wordmark (down+up at its painted bounds) must
/// toggle the primary menu — guards the deferred-hitbox + press-swallow
/// interplay that action-level dispatch can't see.
#[cfg(not(target_os = "macos"))]
#[gpui::test]
fn chrome_menu_opens_on_wordmark_click(cx: &mut TestAppContext) {
    use eidola_gui::chrome::ChromeRoot;
    use gpui::{Modifiers, VisualTestContext, px};

    let _guard = probes_on();

    let stores = ready_stores(cx);
    let window: AnyWindowHandle = cx.update(|cx| {
        gpui_component::init(cx);
        eidola_gui::theme::install(cx);
        cx.open_window(WindowOptions::default(), |window, cx| {
            let view =
                cx.new(|cx| SpaceView::new(stores.clone(), None, WindowInput::new(cx), window, cx));
            let chrome = ChromeRoot::wrap(view.into(), cx);
            cx.new(|cx| Root::new(chrome, window, cx))
        })
        .expect("open test window")
        .into()
    });

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(800.), px(600.)));
    vcx.run_until_parked();

    let entries = probe::window_entries(window.window_id().as_u64());
    let (_, wordmark) = entries
        .iter()
        .find(|(n, _)| n == "chrome/menu")
        .expect("wordmark probe painted");
    let center = wordmark.bounds.center();

    vcx.simulate_click(center, Modifiers::default());
    vcx.run_until_parked();

    let names: Vec<String> = {
        probe::clear_window(window.window_id().as_u64());
        vcx.update(|window, _| window.refresh());
        vcx.run_until_parked();
        probe::window_entries(window.window_id().as_u64())
            .into_iter()
            .map(|(n, _)| n)
            .collect()
    };
    assert!(
        names.contains(&"chrome/menu/panel".to_string()),
        "clicking the wordmark must open the primary menu: {names:?}"
    );
}

// ---------------------------------------------------------------------------
// Local models — the Settings Models pane and the request panel's
// on-this-device group.
// ---------------------------------------------------------------------------

fn local_models_fixture() -> eidola_app_core::LocalModelsState {
    use eidola_app_core::{
        ExternalEngineBackend, LocalModelInfo, LocalModelStatus, LocalModelsState,
    };
    LocalModelsState {
        engine_path: Some("/opt/homebrew/bin/llama-server".into()),
        external: vec![ExternalEngineBackend {
            backend_id: "my-box".into(),
            display_name: "My box".into(),
            enabled: true,
            models_dir: "/Users/me/models".into(),
            engine_path: Some("/opt/homebrew/bin/llama-server".into()),
            auto_start: true,
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
                id: "tiny-a@local".into(),
                slug: "tiny-a".into(),
                display_name: "Tiny A".into(),
                file_name: "tiny-a.gguf".into(),
                size_bytes: Some(3_000_000_000),
                source_url: None,
                status: LocalModelStatus::Available,
                last_error: None,
                on_disk: true,
            },
            LocalModelInfo {
                id: "tiny-b@local".into(),
                slug: "tiny-b".into(),
                display_name: "Tiny B".into(),
                file_name: "tiny-b.gguf".into(),
                size_bytes: Some(5_000_000_000),
                source_url: None,
                status: LocalModelStatus::Loaded {
                    port: 4242,
                    context_tokens: 8192,
                    pinned: false,
                },
                last_error: None,
                on_disk: true,
            },
        ],
    }
}

/// A registry fixture: the two singletons plus one llamacpp external.
fn backends_fixture() -> Vec<eidola_app_core::BackendInfo> {
    use eidola_app_core::{BackendInfo, BackendKind};
    vec![
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
            auto_start: true,
            created_at: 1,
        },
    ]
}

#[gpui::test]
fn backends_pane_probes_cover_installed_catalog_and_url(cx: &mut TestAppContext) {
    use eidola_gui::backends_settings::{BackendsSettingsView, BackendsTab};

    let _guard = probes_on();

    let stores = stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        s.eidola_trust = Some(probe_eidola_trust());
        s.backends = backends_fixture();
        s.local_models = Some(local_models_fixture());
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| BackendsSettingsView::new(stores, window, cx))
    });

    // The tab strip is present regardless of the selected tab. The Eidola tab
    // (default) carries the eidola singleton's enable/disable toggle plus the
    // connection + trust surface (the base-URL editor moved here from General).
    let names = fresh_names(cx, window);
    for expected in [
        "settings/backends/tab/eidola",
        "settings/backends/tab/local",
        "settings/backends/tab/external",
        "settings/backends/eidola/toggle",
        "settings/backends/eidola/url/change",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "eidola-tab probe {expected:?} missing; recorded: {names:?}"
        );
    }
    // No overrides in this fixture: the danger warning band must be absent.
    assert!(
        !names.contains(&"settings/backends/eidola/trust-warning".to_string()),
        "no override → no warning band: {names:?}"
    );

    // The Local tab: the singleton toggle, installed-model verbs, catalog,
    // and the paste-a-URL row.
    view.update(cx, |v, cx| v.select_tab(BackendsTab::Local, cx));
    let names = fresh_names(cx, window);
    for expected in [
        "settings/backends/local/toggle",
        // The Available model affords load + delete; the Loaded one,
        // pin + unload.
        "settings/backends/local/installed/0/load",
        "settings/backends/local/installed/0/delete",
        "settings/backends/local/installed/1/pin",
        "settings/backends/local/installed/1/unload",
        // No fixture file matches a catalog entry, so every catalog row
        // affords download.
        "settings/backends/local/catalog/0/download",
        // The paste-a-URL affordances.
        "settings/backends/local/url",
        "settings/backends/local/url/download",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "local-tab probe {expected:?} missing; recorded: {names:?}"
        );
    }

    // The External tab: the llamacpp backend's toggle/remove/autostart, its
    // scanned model's load verb, and the add-a-backend affordances.
    view.update(cx, |v, cx| v.select_tab(BackendsTab::External, cx));
    let names = fresh_names(cx, window);
    for expected in [
        "settings/backends/my-box/toggle",
        "settings/backends/my-box/remove",
        "settings/backends/my-box/autostart",
        // The llamacpp backend's scanned model affords load (never delete —
        // the file is the user's).
        "settings/backends/my-box/model/0/load",
        // The add-a-backend affordances.
        "settings/backends/add/openai",
        "settings/backends/add/llamacpp",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "external-tab probe {expected:?} missing; recorded: {names:?}"
        );
    }
    // The user-owned file must never afford deletion through Eidola.
    assert!(
        !names.contains(&"settings/backends/my-box/model/0/delete".to_string()),
        "a llamacpp backend's files are the user's: {names:?}"
    );

    probe::set_probes_enabled(false);
}

/// A catalog entry whose download failed leaves a row named exactly like the
/// entry, with no file behind it. "Installed" *replaces* the Download verb,
/// so reading that row as installed would take away the retry the row's own
/// error line is asking for.
#[gpui::test]
fn a_failed_catalog_download_still_affords_downloading(cx: &mut TestAppContext) {
    use eidola_gui::backends_settings::{BackendsSettingsView, BackendsTab};

    let _guard = probes_on();

    let entry = &eidola_app_core::LOCAL_MODEL_CATALOG[0];
    let mut state = local_models_fixture();
    state.models = vec![eidola_app_core::LocalModelInfo {
        id: format!("{}@local", entry.id),
        slug: entry.id.into(),
        display_name: entry.display_name.into(),
        file_name: entry.file_name.into(),
        size_bytes: None,
        source_url: None,
        status: eidola_app_core::LocalModelStatus::Available,
        last_error: Some("connection reset".into()),
        on_disk: false,
    }];

    let stores = stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        s.eidola_trust = Some(probe_eidola_trust());
        s.backends = backends_fixture();
        s.local_models = Some(state);
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| BackendsSettingsView::new(stores, window, cx))
    });
    view.update(cx, |v, cx| v.select_tab(BackendsTab::Local, cx));

    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"settings/backends/local/catalog/0/download".to_string()),
        "a row with no file behind it must keep its Download verb: {names:?}"
    );

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn eidola_trust_surface_probes_cover_editor_and_overrides(cx: &mut TestAppContext) {
    use eidola_gui::backends_settings::BackendsSettingsView;

    let _guard = probes_on();

    // An overridden trust bundle so the revert affordances render.
    let mut trust = probe_eidola_trust();
    trust.base_url = "https://staging.eidola.example/v1".into();
    trust.base_url_is_override = true;
    trust.trusted_measurements = vec![eidola_app_core::MeasurementInfo {
        snp: "9d2bb3ef58af1e7c0c12f3b4a5d6e7f8901a2b3c4d5e6f708192a3b4c5d6e7f8".into(),
        tdx_rtmr1: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into(),
        tdx_rtmr2: "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210".into(),
    }];
    trust.trusted_measurements_are_override = true;
    // Root CA overridden (with a viewable PEM) so the view/copy + Clear verbs
    // render; intermediate left at the pin (only the Set textarea shows there).
    trust.has_hardware_root_ca = true;
    trust.hardware_root_ca_pem =
        Some("-----BEGIN CERTIFICATE-----\nMIIBcustomroot\n-----END CERTIFICATE-----".into());

    let stores = stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        s.eidola_trust = Some(trust);
        s.backends = backends_fixture();
    });
    let (window, view) = open_view(cx, move |window, cx| {
        cx.new(|cx| BackendsSettingsView::new(stores, window, cx))
    });

    // At rest with an active override: the warning band, the base-URL Change
    // affordance, both revert-to-pin verbs, and every trust row's resting
    // affordances — nothing hides behind a disclosure. The *inputs* (add
    // field, CA textareas) stay out of the tree until revealed on demand.
    let names = fresh_names(cx, window);
    for expected in [
        "settings/backends/eidola/trust-warning",
        "settings/backends/eidola/url/change",
        "settings/backends/eidola/url/revert-to-pin",
        "settings/backends/eidola/measurements/revert",
        // The override measurement line: copyable (full triple) + untrustable.
        "settings/backends/eidola/measurements/0/copy",
        "settings/backends/eidola/measurements/0/untrust",
        // The build pin (dropped by the override): auditable + re-addable.
        "settings/backends/eidola/measurements/pin/copy",
        "settings/backends/eidola/measurements/pin/trust",
        // The add-input reveal + the Record cross-link.
        "settings/backends/eidola/measurements/trust-new",
        "settings/backends/eidola/open-record",
        // Root CA override: copyable, replaceable, clearable.
        "settings/backends/eidola/ca/root/copy",
        "settings/backends/eidola/ca/root/change",
        "settings/backends/eidola/ca/root/clear",
        // Intermediate CA at the pin: just the set-custom reveal.
        "settings/backends/eidola/ca/intermediate/change",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "eidola trust-surface probe {expected:?} missing: {names:?}"
        );
    }
    for absent in [
        // The inputs reveal on demand — not in the tree at rest.
        "settings/backends/eidola/measurements/add",
        "settings/backends/eidola/measurements/add/submit",
        "settings/backends/eidola/ca/root/input",
        "settings/backends/eidola/ca/root/set",
        "settings/backends/eidola/ca/intermediate/input",
        // Pinned intermediate CA: nothing to copy or clear.
        "settings/backends/eidola/ca/intermediate/copy",
        "settings/backends/eidola/ca/intermediate/clear",
        // No raw PEM dump in Settings (the Copy verb carries it).
        "settings/backends/eidola/ca/root/current",
        // The disclosure is gone.
        "settings/backends/eidola/advanced/toggle",
    ] {
        assert!(
            !names.contains(&absent.to_string()),
            "{absent:?} must not render at rest: {names:?}"
        );
    }

    // The warning band (role Alert) names exactly which values are overridden.
    let entries = probe::window_entries(window.window_id().as_u64());
    let band = entries
        .iter()
        .find(|(n, _)| n == "settings/backends/eidola/trust-warning")
        .expect("trust warning band recorded");
    assert_eq!(band.1.role, gpui::Role::Alert);
    let label = band.1.label.to_string();
    for named in ["base URL", "trusted measurements", "hardware root CA"] {
        assert!(
            label.contains(named),
            "warning band must name {named:?}; label: {label:?}"
        );
    }

    // Revealing the add input swaps in the field + submit/cancel and retires
    // the reveal verb (edit-in-place).
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.begin_add_measurement(window, cx));
    })
    .unwrap();
    let names = fresh_names(cx, window);
    for expected in [
        "settings/backends/eidola/measurements/add",
        "settings/backends/eidola/measurements/add/submit",
        "settings/backends/eidola/measurements/add/cancel",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "revealed add-measurement probe {expected:?} missing: {names:?}"
        );
    }
    assert!(
        !names.contains(&"settings/backends/eidola/measurements/trust-new".to_string()),
        "the reveal verb retires while the input is open: {names:?}"
    );
    view.update(cx, |v, cx| v.cancel_add_measurement(cx));

    // Revealing a CA editor swaps in that row's textarea + Set/Cancel; the
    // other CA row is untouched.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.begin_edit_ca(eidola_gui::backends_settings::CaKind::Root, window, cx)
        });
    })
    .unwrap();
    let names = fresh_names(cx, window);
    for expected in [
        "settings/backends/eidola/ca/root/input",
        "settings/backends/eidola/ca/root/set",
        "settings/backends/eidola/ca/root/cancel",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "revealed CA editor probe {expected:?} missing: {names:?}"
        );
    }
    assert!(
        !names.contains(&"settings/backends/eidola/ca/intermediate/input".to_string()),
        "editing one CA must not reveal the other's textarea: {names:?}"
    );
    view.update(cx, |v, cx| v.cancel_edit_ca(cx));

    // Entering the base-URL editor swaps in the input + save/cancel.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.begin_edit_base_url(window, cx));
    })
    .unwrap();
    let names = fresh_names(cx, window);
    for expected in [
        "settings/backends/eidola/url/base-url",
        "settings/backends/eidola/url/save",
        "settings/backends/eidola/url/cancel",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "eidola url-editor probe {expected:?} missing: {names:?}"
        );
    }

    probe::set_probes_enabled(false);
}

/// The pinned (no-override) measurement is auditable — its full value is
/// copyable — but **not** untrustable (removing the build root is meaningless;
/// this was the no-op "Untrust" bug). The pin-not-trusted line's Trust verb
/// only appears when an override has actually dropped the pin.
#[gpui::test]
fn eidola_pinned_measurement_copyable_not_untrustable(cx: &mut TestAppContext) {
    use eidola_gui::backends_settings::BackendsSettingsView;

    let _guard = probes_on();

    // Non-override trust: the resolved set is the single build pin.
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        s.eidola_trust = Some(probe_eidola_trust());
        s.backends = backends_fixture();
    });
    let (window, _view) = open_view(cx, move |window, cx| {
        cx.new(|cx| BackendsSettingsView::new(stores, window, cx))
    });

    // The measurement lines are always visible — no disclosure to expand.
    let names = fresh_names(cx, window);

    assert!(
        names.contains(&"settings/backends/eidola/measurements/pin/copy".to_string()),
        "the pinned measurement must be copyable for audit: {names:?}"
    );
    for absent in [
        // No Untrust anywhere — not on the pin card, not on a phantom index.
        "settings/backends/eidola/measurements/pin/untrust",
        "settings/backends/eidola/measurements/0/untrust",
        // Trust (re-add) only shows when an override dropped the pin.
        "settings/backends/eidola/measurements/pin/trust",
    ] {
        assert!(
            !names.contains(&absent.to_string()),
            "pinned measurement must not carry {absent:?}: {names:?}"
        );
    }

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn backends_pane_add_form_probes_appear_per_kind(cx: &mut TestAppContext) {
    use eidola_gui::backends_settings::{AddKind, BackendsSettingsView, BackendsTab};

    let _guard = probes_on();

    let stores = stub_stores(cx, |s| {
        s.backends = backends_fixture();
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| BackendsSettingsView::new(stores, window, cx))
    });

    // The add form lives in the External tab.
    view.update(cx, |v, cx| v.select_tab(BackendsTab::External, cx));

    // Open the OpenAI form: id + url + key inputs plus submit/cancel.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.begin_add(AddKind::OpenAi, window, cx));
    })
    .unwrap();
    let names = fresh_names(cx, window);
    for expected in [
        "settings/backends/add/id",
        "settings/backends/add/url",
        "settings/backends/add/key",
        "settings/backends/add/submit",
        "settings/backends/add/cancel",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "openai add-form probe {expected:?} missing: {names:?}"
        );
    }

    // Switching to the System llama.cpp form swaps url/key for the directory,
    // and adds the optional engine-path input + auto-start checkbox.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.begin_add(AddKind::LlamaCpp, window, cx));
    })
    .unwrap();
    let names = fresh_names(cx, window);
    for expected in [
        "settings/backends/add/dir",
        "settings/backends/add/engine-path",
        "settings/backends/add/autostart",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "llamacpp add-form probe {expected:?} missing: {names:?}"
        );
    }
    assert!(
        !names.contains(&"settings/backends/add/url".to_string()),
        "url input must not linger on the llamacpp form: {names:?}"
    );

    probe::set_probes_enabled(false);
}

// ---------------------------------------------------------------------------
// Participants view + Space Templates pane probes
// ---------------------------------------------------------------------------

fn probe_participants() -> (String, Vec<ParticipantInfo>) {
    let you = ParticipantInfo {
        id: eidola_app_core::HUMAN_PARTICIPANT_ID.into(),
        scope: "global".into(),
        source: "referenced".into(),
        kind: "human".into(),
        // The **wire** label task 64 seeds — every surface that shows this row
        // must map it back to "You" through the shared presentation rule.
        label: "User".into(),
        model_ref: None,
        system_prompt: None,
        notify_policy: "explicit".into(),
        role: "member".into(),
        reference: Some(ParticipantReference {
            base_label: "User".into(),
            base_model_ref: None,
            base_system_prompt: None,
            base_notify_policy: "explicit".into(),
            override_label: None,
            override_model_ref: None,
            override_system_prompt: None,
            override_notify_policy: None,
        }),
    };
    let agent = ParticipantInfo {
        id: "agent-1".into(),
        scope: "space".into(),
        source: "owned".into(),
        kind: "agent".into(),
        label: "Assistant".into(),
        model_ref: Some("gemma4-31b".into()),
        system_prompt: Some("Be concise.".into()),
        notify_policy: "human".into(),
        role: "member".into(),
        reference: None,
    };
    ("demo".into(), vec![you, agent])
}

/// Open a space window with the inspector open over a space whose membership is
/// the fixture above — the Participants section's home since wave 26.3.
fn open_participants_inspector(
    cx: &mut TestAppContext,
) -> (AnyWindowHandle, Entity<eidola_gui::space_view::SpaceView>) {
    let stores = participants_inspector_stores(cx);
    open_participants_inspector_with(cx, stores)
}

/// **A notebook's owner carries no Remove** (task 36; Codex review, PR #279).
///
/// A notebook is a real space, so opening one renders its owner as an ordinary
/// referenced participant. Its membership is structural, though — the space
/// exists only for that agent and is where its `core` memory lives — so
/// app-core refuses to end it, and an affordance that could only be refused has
/// no business on the row. The space's own settings are what say which
/// participant that is.
#[gpui::test]
fn space_inspector_a_notebooks_owner_is_not_removable(cx: &mut TestAppContext) {
    let _guard = probes_on();
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        s.participants = Some(probe_participants());
        s.space_settings = Some((
            "demo".into(),
            eidola_app_core::SpaceSettings {
                notebook_participant_id: Some("agent-1".into()),
                ..Default::default()
            },
        ));
    });
    let (window, view) = open_participants_inspector_with(cx, stores);

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.inspector_toggle_participant("agent-1", window, cx)
        });
    })
    .unwrap();
    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"space/inspector/participants/editor/save".to_string()),
        "the row is still an editor: {names:?}"
    );
    assert!(
        !names.contains(&"space/inspector/participants/agent-1/remove".to_string()),
        "a notebook's owner must not be offered removal from its own notebook: {names:?}"
    );
}

fn participants_inspector_stores(cx: &mut TestAppContext) -> Stores {
    stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        s.participants = Some(probe_participants());
        s.space_settings = Some(("demo".into(), eidola_app_core::SpaceSettings::default()));
    })
}

fn open_participants_inspector_with(
    cx: &mut TestAppContext,
    stores: Stores,
) -> (AnyWindowHandle, Entity<eidola_gui::space_view::SpaceView>) {
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| {
            SpaceView::new(
                stores,
                Some("demo".into()),
                WindowInput::new(cx),
                window,
                cx,
            )
        })
    });
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.set_inspector_open_for_test(true, window, cx))
    })
    .unwrap();
    cx.run_until_parked();
    (window, view)
}

#[gpui::test]
fn space_inspector_participants_probes_cover_rows_editor_and_add(cx: &mut TestAppContext) {
    let _guard = probes_on();
    let (window, view) = open_participants_inspector(cx);

    // The resting rows are the disclosures, plus the two links below them.
    let names = fresh_names(cx, window);
    let you = eidola_app_core::HUMAN_PARTICIPANT_ID;
    for expected in [
        format!("space/inspector/participants/{you}"),
        "space/inspector/participants/agent-1".to_string(),
        "space/inspector/participants/add".to_string(),
        "space/inspector/participants/template".to_string(),
    ] {
        assert!(
            names.contains(&expected),
            "row probe {expected:?} missing: {names:?}"
        );
    }
    // A closed row shows no verbs — Remove lives inside the disclosure.
    assert!(
        !names.contains(&"space/inspector/participants/agent-1/remove".to_string()),
        "a resting row carries no verbs: {names:?}"
    );

    // **The roster names the human the way every other surface does.** Task 64
    // renamed the seeded human's stored label to `User` so a model is never
    // told the other participant is "you"; that is a wire representation, and
    // a surface printing it raw would show the rename to the reader (Codex
    // review, PR #294).
    assert_probe(
        &fresh_entries(cx, window),
        &format!("space/inspector/participants/{you}"),
        gpui::Role::Button,
        "You",
    );

    // Opening You surfaces the edit-everywhere-vs-override-here fork, and the
    // shared human is not removable.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_toggle_participant(you, window, cx));
    })
    .unwrap();
    let names = fresh_names(cx, window);
    for expected in [
        "space/inspector/participants/editor/mode/everywhere",
        "space/inspector/participants/editor/mode/override",
        "space/inspector/participants/editor/name",
        "space/inspector/participants/editor/cancel",
        "space/inspector/participants/editor/save",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "editor probe {expected:?} missing: {names:?}"
        );
    }
    assert!(
        !names.contains(&format!("space/inspector/participants/{you}/remove")),
        "the shared human must not be removable: {names:?}"
    );
    // The human editor shows only the mode toggle + name (no model/prompt).
    assert!(!names.contains(&"space/inspector/participants/editor/model".to_string()));
    // Nothing offers to share what is already shared — promotion is one-way, so
    // no surface here may imply its inverse either.
    assert!(
        !names.contains(&format!("space/inspector/participants/{you}/share")),
        "a referenced global carries no share verb: {names:?}"
    );

    // An agent's disclosure adds the model field, the prompt and notify chips.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.inspector_toggle_participant("agent-1", window, cx)
        });
    })
    .unwrap();
    let names = fresh_names(cx, window);
    for expected in [
        "space/inspector/participants/editor/model",
        "space/inspector/participants/editor/system-prompt",
        "space/inspector/participants/editor/notify/human",
        "space/inspector/participants/agent-1/remove",
        // A space-owned agent is the one that can be shared (task 36).
        "space/inspector/participants/agent-1/share",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "agent editor probe {expected:?} missing: {names:?}"
        );
    }

    // The share asks first — the verb is replaced by its confirmation, which
    // carries the reassurance as a readable node rather than only as pixels.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_begin_promote(window, cx));
    })
    .unwrap();
    let names = fresh_names(cx, window);
    for expected in [
        "space/inspector/participants/agent-1/share/note",
        "space/inspector/participants/agent-1/share/confirm",
        "space/inspector/participants/agent-1/share/cancel",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "share-confirm probe {expected:?} missing: {names:?}"
        );
    }
    assert!(
        !names.contains(&"space/inspector/participants/agent-1/share".to_string()),
        "the armed confirmation replaces the verb: {names:?}"
    );
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_cancel_promote(window, cx));
    })
    .unwrap();

    // The add form.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_begin_add_participant(window, cx));
    })
    .unwrap();
    let names = fresh_names(cx, window);
    for expected in [
        "space/inspector/participants/add/name",
        "space/inspector/participants/add/model",
        "space/inspector/participants/add/system-prompt",
        "space/inspector/participants/add/notify/human",
        "space/inspector/participants/add/submit",
        "space/inspector/participants/add/cancel",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "add-form probe {expected:?} missing: {names:?}"
        );
    }

    // The save-as-template form.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_begin_template(window, cx));
    })
    .unwrap();
    let names = fresh_names(cx, window);
    for expected in [
        "space/inspector/participants/template/title",
        "space/inspector/participants/template/save",
        "space/inspector/participants/template/cancel",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "template-form probe {expected:?} missing: {names:?}"
        );
    }

    probe::set_probes_enabled(false);
}

fn probe_templates() -> Vec<SpaceTemplateInfo> {
    vec![
        SpaceTemplateInfo {
            id: eidola_app_core::DEFAULT_TEMPLATE_ID.into(),
            title: "Default".into(),
            cascade_limit: 4,
            // A remote router — the case whose backend the resting row has to
            // name (it bills per post).
            router_model: Some("gemma4-31b".into()),
            participants: vec![TemplateParticipantInfo {
                id: "t-1".into(),
                label: "Assistant".into(),
                model_ref: Some("gemma4-31b".into()),
                system_prompt: None,
                notify_policy: "human".into(),
            }],
            referenced: Vec::new(),
        },
        SpaceTemplateInfo {
            id: "tmpl-research".into(),
            title: "Research".into(),
            cascade_limit: 6,
            router_model: None,
            participants: Vec::new(),
            // A template saved from a space carries the shared "You" by
            // reference — the read-only half of the editor's participant list.
            referenced: vec![eidola_app_core::TemplateReferencedParticipant {
                id: eidola_app_core::HUMAN_PARTICIPANT_ID.into(),
                kind: "human".into(),
                // The wire label (task 64); the row must read "You".
                label: "User".into(),
                model_ref: None,
                system_prompt: Some("Keep me honest.".into()),
                notify_policy: "explicit".into(),
            }],
        },
    ]
}

#[gpui::test]
fn templates_pane_probes_cover_rows_and_editor(cx: &mut TestAppContext) {
    let _guard = probes_on();
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        s.templates = probe_templates();
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| TemplatesSettingsView::new(stores.clone(), window, cx))
    });

    let names = fresh_names(cx, window);
    for expected in [
        "settings/templates/new",
        "settings/templates/tmpl-research/edit",
        "settings/templates/tmpl-research/set-default",
        "settings/templates/tmpl-research/remove",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "template row probe {expected:?} missing: {names:?}"
        );
    }
    // The built-in Default is the default → no set-default; it's built-in → no remove.
    let default_id = eidola_app_core::DEFAULT_TEMPLATE_ID;
    assert!(!names.contains(&format!("settings/templates/{default_id}/set-default")));
    assert!(!names.contains(&format!("settings/templates/{default_id}/remove")));

    // The create editor.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.begin_create(window, cx));
    })
    .unwrap();
    let names = fresh_names(cx, window);
    for expected in [
        "settings/templates/editor/title",
        "settings/templates/editor/cascade/inc",
        "settings/templates/editor/cascade/dec",
        "settings/templates/editor/add-participant",
        "settings/templates/editor/save",
        "settings/templates/editor/cancel",
        "settings/templates/participant/0/name",
        "settings/templates/participant/0/model",
        "settings/templates/participant/0/system-prompt",
        "settings/templates/participant/0/notify/human",
        "settings/templates/participant/0/remove",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "template editor probe {expected:?} missing: {names:?}"
        );
    }

    probe::set_probes_enabled(false);
}

fn probe_agents() -> Vec<eidola_app_core::GlobalAgentInfo> {
    vec![
        eidola_app_core::GlobalAgentInfo {
            id: "agent-ada".into(),
            label: "Ada".into(),
            model_ref: Some("gemma4-31b".into()),
            system_prompt: Some("Be concise.".into()),
            notify_policy: "human".into(),
            notebook_space_id: Some("nb-ada".into()),
        },
        // A shared agent with no notebook is representable (only promotion
        // makes one), and the row must simply not offer the door.
        eidola_app_core::GlobalAgentInfo {
            id: "agent-bo".into(),
            label: "Bo".into(),
            model_ref: None,
            system_prompt: None,
            notify_policy: "explicit".into(),
            notebook_space_id: None,
        },
    ]
}

#[gpui::test]
fn agents_pane_probes_cover_rows_editor_and_retire(cx: &mut TestAppContext) {
    let _guard = probes_on();
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        s.agents = Some(probe_agents());
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| AgentsSettingsView::new(stores.clone(), window, cx))
    });

    // A resting row: its content is what it answers with and when.
    let entries = fresh_entries(cx, window);
    assert_probe_value(
        &entries,
        "settings/agents/agent-ada",
        gpui::Role::ListItem,
        "Ada",
        "gemma4-31b · Eidola · responds to people",
    );
    let names: Vec<String> = entries.iter().map(|(name, _)| name.clone()).collect();
    for expected in [
        "settings/agents/agent-ada/notebook",
        "settings/agents/agent-ada/edit",
        "settings/agents/agent-ada/retire",
        "settings/agents/agent-bo/edit",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "agent row probe {expected:?} missing: {names:?}"
        );
    }
    // No notebook, no door.
    assert!(
        !names.contains(&"settings/agents/agent-bo/notebook".to_string()),
        "an agent without a notebook offers no notebook verb: {names:?}"
    );

    // The editor.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.toggle_edit("agent-ada", window, cx));
    })
    .unwrap();
    let names = fresh_names(cx, window);
    for expected in [
        "settings/agents/editor/name",
        "settings/agents/editor/model",
        "settings/agents/editor/system-prompt",
        "settings/agents/editor/notify/human",
        "settings/agents/editor/save",
        "settings/agents/editor/cancel",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "agent editor probe {expected:?} missing: {names:?}"
        );
    }

    // A refused write stands **under its own row**, keyed like the store's slot,
    // so two of them can be told apart — and its accessible name carries the
    // agent's, which the row above it cannot supply to a screen reader.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.cancel_edit(window, cx));
    })
    .unwrap();
    let refusal = "Couldn't save: notify policy must be explicit, human or all";
    stores
        .agents
        .update(cx, |s, _| s.set_op_error_for_test("agent-ada", refusal));
    let entries = fresh_entries(cx, window);
    assert_probe(
        &entries,
        "settings/agents/agent-ada/error",
        gpui::Role::Alert,
        &format!("Ada: {refusal}"),
    );
    assert_probe(
        &entries,
        "settings/agents/agent-ada/error/dismiss",
        gpui::Role::Button,
        "Dismiss the message about Ada",
    );
    let names: Vec<String> = entries.iter().map(|(name, _)| name.clone()).collect();
    assert!(
        !names.contains(&"settings/agents/agent-bo/error".to_string()),
        "another agent's row carries no band: {names:?}"
    );

    // The retire confirmation — its note is a readable node, not only pixels.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.arm_retire("agent-ada", window, cx));
    })
    .unwrap();
    let names = fresh_names(cx, window);
    for expected in [
        "settings/agents/agent-ada/retire/note",
        "settings/agents/agent-ada/retire/confirm",
        "settings/agents/agent-ada/retire/cancel",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "retire-confirm probe {expected:?} missing: {names:?}"
        );
    }

    probe::set_probes_enabled(false);
}

/// The third state the other two are measured against: a cell that **has not
/// answered**. `Loadable`'s rule is that it says nothing — so rendering the
/// "share one from a space" invitation over it tells a cold-opening reader their
/// shared agents are gone (Codex review, PR #279).
#[gpui::test]
fn agents_pane_unread_library_is_not_an_empty_one(cx: &mut TestAppContext) {
    let _guard = probes_on();
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        // No fixture: the stub store stays `NotLoaded` and never answers.
        s.agents = None;
    });
    let (window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| AgentsSettingsView::new(stores.clone(), window, cx))
    });
    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"settings/agents/loading".to_string()),
        "an unread library says it is unread: {names:?}"
    );
    assert!(
        !names.contains(&"settings/agents/empty".to_string()),
        "and must not read as an empty one: {names:?}"
    );

    probe::set_probes_enabled(false);
}

/// An empty library says so, and a failed *initial* read says something else —
/// "Failed is not empty", over the pane that would otherwise invite the reader
/// to share an agent they may already have.
#[gpui::test]
fn agents_pane_failed_load_shows_retry_not_an_empty_library(cx: &mut TestAppContext) {
    let _guard = probes_on();
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        s.agents = Some(Vec::new());
    });
    let (window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| AgentsSettingsView::new(stores.clone(), window, cx))
    });
    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"settings/agents/empty".to_string()),
        "an empty library names its door: {names:?}"
    );

    stores
        .agents
        .update(cx, |s, _| s.set_failed_for_test("boom"));
    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"settings/agents/retry".to_string()),
        "a failed load must offer Retry: {names:?}"
    );
    assert!(
        !names.contains(&"settings/agents/empty".to_string()),
        "a failed load must not read as an empty library: {names:?}"
    );

    probe::set_probes_enabled(false);
}

/// The router row's probes, and the cost note's exact condition: it is present
/// for a **remote** (eidola) reference and absent for Off and for a local one.
/// Both directions, because a cost warning that never disappears teaches
/// nothing and one that never appears is the whole failure this copy exists to
/// prevent.
#[gpui::test]
fn templates_pane_router_probes_and_remote_cost_note(cx: &mut TestAppContext) {
    let _guard = probes_on();
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        s.backends = backends_fixture();
        s.templates = probe_templates();
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| TemplatesSettingsView::new(stores.clone(), window, cx))
    });
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.begin_edit("tmpl-research", window, cx));
    })
    .unwrap();

    // Off (the default): the picker is there, the cost note is not.
    let entries = fresh_entries(cx, window);
    assert_probe(
        &entries,
        "settings/templates/editor/router",
        gpui::Role::Button,
        "Router model",
    );
    // The referenced global this template carries is listed read-only — named,
    // with its config as the content (its effective system prompt included, so
    // a real charter is neither hidden nor silently dropped), and no verbs.
    assert_probe_value(
        &entries,
        "settings/templates/editor/referenced/0",
        gpui::Role::Label,
        "You — shared participant",
        "Responds when asked · Keep me honest.",
    );
    assert!(
        !entries
            .iter()
            .any(|(n, _)| n == "settings/templates/editor/router/cost"),
        "Off must carry no cost note"
    );

    // Opening the picker offers Off as a first-class option.
    view.update(cx, |v, cx| v.toggle_router_picker(cx));
    let entries = fresh_entries(cx, window);
    assert_probe(
        &entries,
        "settings/templates/editor/router/menu",
        gpui::Role::ListBox,
        "Router models",
    );
    assert_probe(
        &entries,
        "settings/templates/editor/router/option/off",
        gpui::Role::Button,
        "Off",
    );

    // A remote (eidola) reference: the note is visible, in full, always.
    view.update(cx, |v, cx| v.set_router_model(Some("gemma4-31b"), cx));
    let entries = fresh_entries(cx, window);
    assert_probe(
        &entries,
        "settings/templates/editor/router/cost",
        gpui::Role::Label,
        eidola_gui::templates_settings::ROUTER_REMOTE_COST_NOTE,
    );

    // A local reference routes free — no note.
    view.update(cx, |v, cx| {
        v.set_router_model(Some("gemma-4-e2b@local"), cx)
    });
    let entries = fresh_entries(cx, window);
    assert!(
        !entries
            .iter()
            .any(|(n, _)| n == "settings/templates/editor/router/cost"),
        "a local router routes free — no per-call cost note"
    );

    probe::set_probes_enabled(false);
}

/// A resting row summarizes its template, and the router is named there because
/// it bills per post — which makes the **backend** the load-bearing half:
/// `gemma4-31b@eidola` charges an inference on every post in every space the
/// template makes, where an identically-named on-device model is free, and the
/// model name alone cannot tell those apart. The summary is also the row's
/// spoken content (settled — it moves only when the registry is refetched), so
/// a screen reader hears the same distinction.
#[gpui::test]
fn a_template_row_names_its_routers_backend(cx: &mut TestAppContext) {
    let _guard = probes_on();
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        s.backends = backends_fixture();
        s.templates = probe_templates();
    });
    let (window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| TemplatesSettingsView::new(stores.clone(), window, cx))
    });

    let entries = fresh_entries(cx, window);
    assert_probe_value(
        &entries,
        &format!(
            "settings/templates/{}",
            eidola_app_core::DEFAULT_TEMPLATE_ID
        ),
        gpui::Role::ListItem,
        "Default — the default template",
        "cascade 4 · router gemma4-31b · Eidola · Assistant",
    );
    // Off is the default and says nothing; the row is otherwise the same shape.
    assert_probe_value(
        &entries,
        "settings/templates/tmpl-research",
        gpui::Role::ListItem,
        "Research",
        "cascade 6 · You",
    );

    probe::set_probes_enabled(false);
}

/// The editor's referenced-globals list is read-only display of config that
/// lives on the **shared** participant, so an "edit everywhere" landing while
/// the editor is open — from the Participants window, another window, or the
/// CLI — must show through. The draft carries only what the editor edits;
/// these rows resolve against the live registry every frame.
#[gpui::test]
fn templates_editor_referenced_rows_follow_the_live_registry(cx: &mut TestAppContext) {
    let _guard = probes_on();
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        s.templates = probe_templates();
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| TemplatesSettingsView::new(stores.clone(), window, cx))
    });
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.begin_edit("tmpl-research", window, cx));
    })
    .unwrap();
    let entries = fresh_entries(cx, window);
    assert_probe_value(
        &entries,
        "settings/templates/editor/referenced/0",
        gpui::Role::Label,
        "You — shared participant",
        "Responds when asked · Keep me honest.",
    );

    // The shared global is edited everywhere: `Change::Participants` refreshes
    // this store (a `SpaceTemplateInfo` embeds its referenced globals'
    // effective config), and the snapshot it hands back carries the new values.
    let mut edited = probe_templates();
    let referenced = &mut edited[1].referenced[0];
    referenced.label = "Myself".into();
    referenced.system_prompt = Some("Keep me sharp.".into());
    referenced.notify_policy = "all".into();
    stores
        .templates
        .update(cx, |s, cx| s.set_templates_for_test(edited, cx));

    // The editor is still open, and it shows the edit — not the values it
    // opened with.
    let entries = fresh_entries(cx, window);
    assert!(
        view.read_with(cx, |v, _| v.is_editing()),
        "the editor must still be open — otherwise this proves nothing"
    );
    assert_probe_value(
        &entries,
        "settings/templates/editor/referenced/0",
        gpui::Role::Label,
        "Myself — shared participant",
        "Responds to everything · Keep me sharp.",
    );

    probe::set_probes_enabled(false);
}

/// Two enabled backends serving a model of the same name — the managed local
/// store and a user's own llama.cpp install, which is exactly what happens when
/// someone downloads a model Eidola also curates.
fn same_name_on_two_backends() -> eidola_app_core::LocalModelsState {
    use eidola_app_core::{
        ExternalEngineBackend, LocalModelInfo, LocalModelStatus, LocalModelsState,
    };
    let model = |id: &str, slug: &str| LocalModelInfo {
        id: id.into(),
        slug: slug.into(),
        display_name: "Gemma 4 E2B".into(),
        file_name: "gemma-4-e2b.gguf".into(),
        size_bytes: Some(3_000_000_000),
        source_url: None,
        status: LocalModelStatus::Available,
        last_error: None,
        on_disk: true,
    };
    LocalModelsState {
        engine_path: Some("/opt/homebrew/bin/llama-server".into()),
        external: vec![ExternalEngineBackend {
            backend_id: "my-box".into(),
            display_name: "My box".into(),
            enabled: true,
            models_dir: "/Users/me/models".into(),
            engine_path: Some("/opt/homebrew/bin/llama-server".into()),
            auto_start: true,
            models: vec![model("gemma-4-e2b@my-box", "gemma-4-e2b")],
        }],
        models: vec![model("gemma-4-e2b@local", "gemma-4-e2b")],
    }
}

/// **A dropdown option has to name its own backend.** The visible row shows
/// only the model name because the group header above it supplies the backend —
/// but that header is a role-less `div`, so it never reaches the accessibility
/// tree, and two backends serving the same model name give a screen reader two
/// rows it cannot tell apart. On the router picker that ambiguity is also the
/// billing difference. One helper serves both pickers, so this pins both.
#[gpui::test]
fn model_picker_options_name_their_backend(cx: &mut TestAppContext) {
    let _guard = probes_on();
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        s.backends = backends_fixture();
        s.local_models = Some(same_name_on_two_backends());
        s.templates = probe_templates();
        s.participants = Some(probe_participants());
    });

    // The Templates pane's router picker.
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| TemplatesSettingsView::new(stores.clone(), window, cx))
    });
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.begin_edit("tmpl-research", window, cx));
    })
    .unwrap();
    view.update(cx, |v, cx| v.toggle_router_picker(cx));
    let entries = fresh_entries(cx, window);
    assert_probe(
        &entries,
        "settings/templates/editor/router/option/0/0",
        gpui::Role::Button,
        "Gemma 4 E2B · Local",
    );
    assert_probe(
        &entries,
        "settings/templates/editor/router/option/1/0",
        gpui::Role::Button,
        "Gemma 4 E2B · My box",
    );

    // The participant model field — the same shared widget, the same shape.
    let (p_window, p_view) = open_participants_inspector_with(cx, stores.clone());
    cx.update_window(p_window, |_, window, cx| {
        p_view.update(cx, |v, cx| {
            v.inspector_begin_add_participant(window, cx);
            v.inspector_open_add_picker_for_test(cx);
        });
    })
    .unwrap();
    let entries = fresh_entries(cx, p_window);
    assert_probe(
        &entries,
        "space/inspector/participants/add/model/option/0/0",
        gpui::Role::Button,
        "Gemma 4 E2B · Local",
    );
    assert_probe(
        &entries,
        "space/inspector/participants/add/model/option/1/0",
        gpui::Role::Button,
        "Gemma 4 E2B · My box",
    );

    probe::set_probes_enabled(false);
}

/// A picker's label names it ("Model", "Router model"); *which* model is chosen
/// is its content, and rides `aria_value` — settled by construction, since it
/// moves only on a click. Both pickers take the same shape from one helper: a
/// screen reader otherwise hears the same word whether a space routes on-device,
/// bills per post, or has no model selected at all.
#[gpui::test]
fn model_pickers_announce_their_selection(cx: &mut TestAppContext) {
    let _guard = probes_on();
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        s.backends = backends_fixture();
        s.templates = probe_templates();
        s.participants = Some(probe_participants());
    });

    // The Templates pane's router picker: Off (the default) is a real choice
    // and says so; a chosen reference names itself and its backend.
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| TemplatesSettingsView::new(stores.clone(), window, cx))
    });
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.begin_edit("tmpl-research", window, cx));
    })
    .unwrap();
    let entries = fresh_entries(cx, window);
    assert_probe_value(
        &entries,
        "settings/templates/editor/router",
        gpui::Role::Button,
        "Router model",
        "Off",
    );
    view.update(cx, |v, cx| v.set_router_model(Some("gemma4-31b"), cx));
    let entries = fresh_entries(cx, window);
    assert_probe_value(
        &entries,
        "settings/templates/editor/router",
        gpui::Role::Button,
        "Router model",
        "gemma4-31b · Eidola",
    );

    // The participant model field — the shared widget the Templates pane's own
    // agent rows use too — takes the identical shape.
    let (p_window, p_view) = open_participants_inspector_with(cx, stores.clone());
    cx.update_window(p_window, |_, window, cx| {
        p_view.update(cx, |v, cx| {
            v.inspector_toggle_participant("agent-1", window, cx)
        });
    })
    .unwrap();
    let entries = fresh_entries(cx, p_window);
    assert_probe_value(
        &entries,
        "space/inspector/participants/editor/model",
        gpui::Role::Button,
        "Model",
        "gemma4-31b · Eidola",
    );

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn space_inspector_participants_failed_load_shows_retry_not_controls(cx: &mut TestAppContext) {
    let _guard = probes_on();
    let stores = participants_inspector_stores(cx);
    let (window, _view) = open_participants_inspector_with(cx, stores.clone());
    // A failed *initial* load: no prior data, no live roster.
    stores
        .participants
        .update(cx, |s, _| s.set_failed_for_test("demo", "boom"));

    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"space/inspector/participants/retry".to_string()),
        "failed load must offer Retry: {names:?}"
    );
    for absent in [
        "space/inspector/participants/add",
        "space/inspector/participants/template",
    ] {
        assert!(
            !names.contains(&absent.to_string()),
            "a failed load must not show live controls ({absent:?}): {names:?}"
        );
    }
    probe::set_probes_enabled(false);
}

#[gpui::test]
fn templates_pane_failed_load_shows_retry(cx: &mut TestAppContext) {
    let _guard = probes_on();
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| TemplatesSettingsView::new(stores.clone(), window, cx))
    });
    stores
        .templates
        .update(cx, |s, _| s.set_failed_for_test("boom"));
    let _ = view;

    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"settings/templates/retry".to_string()),
        "failed template load must offer Retry: {names:?}"
    );
    assert!(
        !names.contains(&"settings/templates/new".to_string()),
        "a failed registry load must not read as an empty registry with New: {names:?}"
    );
    probe::set_probes_enabled(false);
}

#[gpui::test]
fn space_probes_record_footnote_rail_and_highlight_picker(cx: &mut TestAppContext) {
    // Every quoted-reference affordance is a probe target: the footnote rows
    // on a post and on a draft, their removal chips, and the multi-referencer
    // picker. (The highlight *wash* itself is a decoration, not an
    // affordance — the editor owns its hit-test, so it carries no probe.)
    let _guard = probes_on();

    let stores = ready_stores(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, Some("s".into()), WindowInput::new(cx), window, cx))
    });
    let space = view.read_with(cx, |v, _| v.space().clone());

    let mut post = probe_post("a1", "the quick brown fox");
    post.blocks[0].id = "b1".into();
    post.references = vec![
        eidola_app_core::PostReference {
            antecedent_action_id: "x1".into(),
            ordinal: 1,
            content_block_id: Some("bx".into()),
            range_start: Some(0),
            range_end: Some(4),
            annotation: None,
            snippet: Some("an earlier passage".into()),
            antecedent_author_label: "Ada".into(),
            antecedent_author_kind: "agent".into(),
        },
        eidola_app_core::PostReference {
            antecedent_action_id: "x2".into(),
            ordinal: 2,
            content_block_id: Some("by".into()),
            range_start: Some(0),
            range_end: Some(4),
            annotation: None,
            snippet: None, // the honest "quoted an earlier version" row
            antecedent_author_label: "Ada".into(),
            antecedent_author_kind: "agent".into(),
        },
    ];
    cx.update(|cx| {
        space.update(cx, |s, cx| s.set_post_tree_for_test(vec![post], cx));
    });
    draw(cx, window);

    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"space/post/0/footnote/1".to_string()),
        "the post's footnote rows are probed: {names:?}"
    );
    assert!(
        names.contains(&"space/post/0/footnote/2".to_string()),
        "including the unresolvable-range row: {names:?}"
    );
    assert!(
        !names.contains(&"space/post/0/footnote/1/remove".to_string()),
        "removal chips appear only inside an Edit session: {names:?}"
    );

    // Inside an Edit session each row grows its removal chip.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.begin_edit("a1".into(), window, cx));
    })
    .unwrap();
    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"space/post/0/footnote/1/remove".to_string()),
        "the edit session reveals removal chips: {names:?}"
    );

    // Cancel out, then quote into the tail draft: the draft rail probes.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.cancel_edit(window, cx));
        view.update(cx, |v, cx| v.select_in_post_for_test("a1", 4..15, cx));
        view.update(cx, |v, cx| v.quote(&eidola_gui::actions::Quote, window, cx));
    })
    .unwrap();
    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"space/draft/footnote/1".to_string()),
        "the draft's pending quote is a footnote row: {names:?}"
    );
    assert!(
        names.contains(&"space/draft/footnote/1/remove".to_string()),
        "a pending quote is always removable: {names:?}"
    );
    assert!(
        !names.contains(&"space/draft/footnote/1/embed".to_string()),
        "the marker is in the body, so there is nothing to re-embed: {names:?}"
    );

    // Drop the marker (as a Backspace over the quote block would) and the
    // rail grows its "embed" affordance — the way back.
    let composer = view
        .read_with(cx, |v, _| v.composer_state_for_test())
        .expect("the draft's editor");
    cx.update_window(window, |_, _, cx| {
        composer.update(cx, |e, cx| e.set_value("prose only", cx));
    })
    .unwrap();
    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"space/draft/footnote/1/embed".to_string()),
        "a quote with no marker offers to re-embed: {names:?}"
    );

    // The multi-referencer picker.
    let incoming = |action: &str| eidola_app_core::IncomingReference {
        action_id: action.into(),
        space_id: "s".into(),
        ordinal: 1,
        content_block_id: Some("b1".into()),
        range_start: Some(4),
        range_end: Some(15),
        annotation: None,
        created_at: 0,
    };
    cx.update(|cx| {
        space.update(cx, |s, _| {
            s.seed_incoming_references_for_test("a1", vec![incoming("z1"), incoming("z2")]);
        });
    });
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.click_highlight_for_test("a1", &[0, 1], window, cx)
        });
    })
    .unwrap();
    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"space/highlight/picker".to_string()),
        "the picker group is probed: {names:?}"
    );
    assert!(
        names.contains(&"space/highlight/picker/0".to_string())
            && names.contains(&"space/highlight/picker/1".to_string()),
        "each candidate is a probed choice: {names:?}"
    );
}

/// REGRESSION (task 68 + Codex review, PR #292): a footnote row's byline was
/// resolved *only* against this space's posts, so a cross-space quote — the one
/// case a reference exists for — lost its author the moment the post persisted.
/// The draft rail had the name (the picker handed it over), and submitting
/// replaced it with the literal "another space".
///
/// The rail reads the author from the first source that has one, and this
/// asserts every arm at once on one post:
///
/// 1. the in-space post's own **gutter byline**, which is *not* the effective
///    label the edge carries — a human participant labelled `user` reads "You"
///    in the gutter, and two names for one person inside one window would be
///    worse than the bug;
/// 2. the edge's carried identity, **by name**, for a conversation this window
///    never loaded;
/// 3. the edge's carried identity where the label is blank (the schema's
///    "override to empty"): the *kind* still names them — "You" for the one
///    human, "Eidola" for an unnamed agent — which is the second half of the
///    same defect. Composing that quote showed the source window's rendered
///    byline, so a raw label would have flipped "You" to "another space" at the
///    durability boundary, exactly as before, one layer down;
/// 4. `ELSEWHERE` only where nothing names anyone at all.
#[gpui::test]
fn space_footnote_rail_names_a_cross_space_author(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let stores = ready_stores(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, Some("s".into()), WindowInput::new(cx), window, cx))
    });
    let space = view.read_with(cx, |v, _| v.space().clone());

    let reference = |antecedent: &str, ordinal: i64, kind: &str, label: &str, snippet: &str| {
        eidola_app_core::PostReference {
            antecedent_action_id: antecedent.into(),
            ordinal,
            content_block_id: Some("bx".into()),
            range_start: Some(0),
            range_end: Some(4),
            annotation: None,
            snippet: Some(snippet.into()),
            antecedent_author_label: label.into(),
            antecedent_author_kind: kind.into(),
        }
    };

    // The quoted post that *is* in this space — a human participant, so its
    // gutter byline is "You" while the edge carries the label "user".
    let quoted = probe_post("a1", "the sentence everything else hangs off");
    let mut post = probe_post("a2", "one post, every way of naming a quoted author");
    post.parent_action_id = Some("a1".into());
    post.relation = Some("reply".into());
    post.references = vec![
        reference("a1", 1, "human", "user", "everything else hangs off"),
        // Quoted out of a conversation this window never loaded: only the
        // edge's own identity can name Sofia.
        reference(
            "x1",
            2,
            "agent",
            "Sofia",
            "a passage from another conversation",
        ),
        // …and from a space that overrode the label to empty. The kind is what
        // is left, and it is enough: this is the reader's own passage.
        reference("x2", 3, "human", "", "a passage of my own from elsewhere"),
        // The same, for an agent nobody named.
        reference("x3", 4, "agent", "  ", "a passage by an agent nobody named"),
        // Nothing names anyone: no label, and a kind with no name of its own.
        reference(
            "x4",
            5,
            "tool",
            "",
            "a passage from a space that names no one",
        ),
    ];
    cx.update(|cx| {
        space.update(cx, |s, cx| s.set_post_tree_for_test(vec![quoted, post], cx));
    });

    let entries = fresh_entries(cx, window);
    for (index, expected) in [
        (1, "You — everything else hangs off"),
        (2, "Sofia — a passage from another conversation"),
        (3, "You — a passage of my own from elsewhere"),
        (4, "Eidola — a passage by an agent nobody named"),
        (
            5,
            "another space — a passage from a space that names no one",
        ),
    ] {
        assert_probe(
            &entries,
            &format!("space/post/1/footnote/{index}"),
            gpui::Role::Link,
            &format!("Reference {index}: {expected}"),
        );
    }

    probe::set_probes_enabled(false);
}

/// REGRESSION (Codex review, PR #292, round 2): the rail's cross-space
/// fallback renders the carried identity through the gutter's **whole**
/// rendering, not only its first pass.
///
/// The gutter is two passes — `byline_for_participant` resolves the identity
/// pair, then an assistant row's byline is resolved *again* through
/// `SpaceView::model_display` — and both the post gutter and the **draft** rail
/// read the same `PostData::byline`, so both already show the second pass. A
/// participant label that parses as a model selector therefore composed as
/// "Gemma 4 E4B" and persisted as `gemma-4-E4B_q4_0-it@local`: the attribution
/// changing at the durability boundary, which is the defect this whole PR is
/// about, one layer down. Nothing mints such a label — `db::default_agent_label`
/// strips the `@backend` suffix and title-cases — but a reader can type one.
#[gpui::test]
fn space_footnote_rail_names_a_cross_space_author_the_way_the_gutter_would(
    cx: &mut TestAppContext,
) {
    let _guard = probes_on();

    let stores = stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        s.local_models = Some(eidola_app_core::LocalModelsState {
            engine_path: Some("/opt/eidola/llama-server".into()),
            external: Vec::new(),
            models: vec![eidola_app_core::LocalModelInfo {
                id: "gemma-4-E4B_q4_0-it@local".into(),
                slug: "gemma-4-E4B_q4_0-it".into(),
                display_name: "Gemma 4 E4B".into(),
                file_name: "gemma-4-E4B_q4_0-it.gguf".into(),
                size_bytes: Some(5_154_939_136),
                source_url: None,
                status: eidola_app_core::LocalModelStatus::Available,
                last_error: None,
                on_disk: true,
            }],
        });
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, Some("s".into()), WindowInput::new(cx), window, cx))
    });
    let space = view.read_with(cx, |v, _| v.space().clone());

    // A passage quoted out of a conversation this window never loaded, whose
    // author a reader named after the model it answers with.
    let mut post = probe_post("a1", "the post that carries the quote");
    post.references = vec![eidola_app_core::PostReference {
        antecedent_action_id: "x1".into(),
        ordinal: 1,
        content_block_id: Some("bx".into()),
        range_start: Some(0),
        range_end: Some(4),
        annotation: None,
        snippet: Some("a passage from another conversation".into()),
        antecedent_author_label: "gemma-4-E4B_q4_0-it@local".into(),
        antecedent_author_kind: "agent".into(),
    }];
    cx.update(|cx| {
        space.update(cx, |s, cx| s.set_post_tree_for_test(vec![post], cx));
    });

    let entries = fresh_entries(cx, window);
    assert_probe(
        &entries,
        "space/post/0/footnote/1",
        gpui::Role::Link,
        "Reference 1: Gemma 4 E4B — a passage from another conversation",
    );

    probe::set_probes_enabled(false);
}

/// REGRESSION (Codex review, PR #292): a participant label has no maximum
/// length and the rail's byline sits `flex_none` beside the passage, so a long
/// one could squeeze the quoted text out of the row it is there to attribute.
/// It is now bounded — and the bound is **visual only**: the row's accessible
/// name is built from the whole byline, so a screen reader still hears who
/// wrote the passage. That is the half a test can see; the elision itself is
/// the driver scene `space_cross_space_quote`.
#[gpui::test]
fn space_footnote_rail_speaks_a_long_byline_in_full(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let stores = ready_stores(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, Some("s".into()), WindowInput::new(cx), window, cx))
    });
    let space = view.read_with(cx, |v, _| v.space().clone());

    let long = "Republic Book I Close-Reading Group (Tuesdays, in the long room)";
    let mut post = probe_post("a1", "a quote from a very long-named colleague");
    post.references = vec![eidola_app_core::PostReference {
        antecedent_action_id: "x1".into(),
        ordinal: 1,
        content_block_id: Some("bx".into()),
        range_start: Some(0),
        range_end: Some(4),
        annotation: None,
        snippet: Some("the passage itself".into()),
        antecedent_author_label: long.into(),
        antecedent_author_kind: "agent".into(),
    }];
    cx.update(|cx| {
        space.update(cx, |s, cx| s.set_post_tree_for_test(vec![post], cx));
    });

    let entries = fresh_entries(cx, window);
    assert_probe(
        &entries,
        "space/post/0/footnote/1",
        gpui::Role::Link,
        &format!("Reference 1: {long} — the passage itself"),
    );

    probe::set_probes_enabled(false);
}

/// REGRESSION: the Settings window's drag band reserved 44px while every other
/// window used 36 — purely to fake the top padding the Backends pane lacked.
/// Once the band began *blocking* the mouse (task 32), that extra strip sat
/// over the pane's tab strip and the Eidola/Local/External tabs stopped being
/// clickable. The band is now the shared `DRAG_BAND_HEIGHT` everywhere and the
/// pane carries its own "Backends" title, so every tab paints clear of it.
#[gpui::test]
fn settings_backends_tabs_paint_clear_of_the_drag_band(cx: &mut TestAppContext) {
    use eidola_gui::settings::{SettingsPane, SettingsView};

    let _guard = probes_on();

    let stores = ready_stores(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SettingsView::new(stores, window, cx))
    });
    view.update(cx, |v, cx| v.select(SettingsPane::Backends, cx));

    let entries = fresh_entries(cx, window);
    // The band's own hitbox spans the window's full width from y=0, so a tab
    // whose top is inside it is a tab the band swallows.
    let band = eidola_gui::titlebar::DRAG_BAND_HEIGHT.as_f32();
    for slug in ["eidola", "local", "external"] {
        let name = format!("settings/backends/tab/{slug}");
        let (_, entry) = entries
            .iter()
            .find(|(n, _)| *n == name)
            .unwrap_or_else(|| panic!("probe {name:?} missing"));
        let top = entry.bounds.origin.y.as_f32();
        assert!(
            top >= band,
            "tab {slug:?} paints at y={top}, inside the {band}px drag band that blocks the mouse"
        );
    }

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn space_probes_record_the_trace_disclosure(cx: &mut TestAppContext) {
    // The trace disclosure is an affordance, so it is a probe target — and the
    // annotation is also the driver's selector, so the labels have to say what
    // the click does and what each round was.
    let _guard = probes_on();

    let stores = ready_stores(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, Some("s".into()), WindowInput::new(cx), window, cx))
    });
    let space = view.read_with(cx, |v, _| v.space().clone());

    let ask = probe_post("a1", "which branch settled it?");
    cx.update(|cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![ask], cx);
            s.seed_traces_for_test(vec![eidola_app_core::PostTrace {
                id: "t1".into(),
                anchor_action_id: "a1".into(),
                participant_label: "Mara".into(),
                unanswered: true,
                entries: vec![
                    eidola_app_core::TraceEntry::Tool {
                        action_id: "tc1".into(),
                        request_id: Some("req-1".into()),
                        call_id: "c1".into(),
                        name: "read_thread".into(),
                        arguments: "{\"handle\":\"h1\"}".into(),
                        result: Some("8 posts".into()),
                    },
                    eidola_app_core::TraceEntry::Declined {
                        action_id: "d1".into(),
                        reason: Some("nothing to add".into()),
                    },
                ],
            }]);
        });
    });

    // Collapsed: the toggle is there, its rows are not.
    let entries = fresh_entries(cx, window);
    assert_probe_value(
        &entries,
        "space/post/0/trace/0",
        gpui::Role::Button,
        "Show what this turn did",
        "Mara — declined to respond · 1 tool call",
    );
    assert!(
        !entries
            .iter()
            .any(|(n, _)| n == "space/post/0/trace/0/round/1"),
        "a collapsed disclosure reveals no rows"
    );

    // Expansion is keyed on the turn, not on the post it hangs under.
    cx.update(|cx| {
        space.update(cx, |s, cx| s.toggle_trace("t1", cx));
    });
    let entries = fresh_entries(cx, window);
    assert_probe(
        &entries,
        "space/post/0/trace/0",
        gpui::Role::Button,
        "Hide what this turn did",
    );
    // A round with a recorded exchange is a Link (it opens the Record); a
    // decision has no exchange of its own, so it is a plain list item.
    assert_probe(
        &entries,
        "space/post/0/trace/0/round/1",
        gpui::Role::Link,
        "Round 1: read_thread — {\"handle\":\"h1\"} → 8 posts",
    );
    assert_probe(
        &entries,
        "space/post/0/trace/0/round/2",
        gpui::Role::ListItem,
        "Round 2: declined — nothing to add",
    );

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn space_probes_record_a_disclosure_per_turn(cx: &mut TestAppContext) {
    // A fan-out puts several turns under one post, and a post can be declined
    // twice by one agent. Each turn is its own quiet line, named for the
    // participant that ran it — one aggregated line would credit everybody's
    // activity to whichever turn happened to sort first.
    let _guard = probes_on();

    let stores = ready_stores(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, Some("s".into()), WindowInput::new(cx), window, cx))
    });
    let space = view.read_with(cx, |v, _| v.space().clone());

    let declined = |id: &str, label: &str, reason: &str| eidola_app_core::PostTrace {
        id: id.into(),
        anchor_action_id: "a1".into(),
        participant_label: label.into(),
        unanswered: true,
        entries: vec![
            eidola_app_core::TraceEntry::Tool {
                action_id: format!("{id}-tc"),
                request_id: Some(format!("{id}-req")),
                call_id: "c1".into(),
                name: "decline".into(),
                arguments: "{}".into(),
                result: Some("Declined.".into()),
            },
            eidola_app_core::TraceEntry::Declined {
                action_id: format!("{id}-d"),
                reason: Some(reason.into()),
            },
        ],
    };

    cx.update(|cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![probe_post("a1", "which branch settled it?")], cx);
            s.seed_traces_for_test(vec![
                declined("t-mara", "Mara", "not my area"),
                declined("t-ferris", "Ferris", "Mara covered it"),
            ]);
        });
    });

    let entries = fresh_entries(cx, window);
    assert_probe_value(
        &entries,
        "space/post/0/trace/0",
        gpui::Role::Button,
        "Show what this turn did",
        "Mara — declined to respond · 1 tool call",
    );
    assert_probe_value(
        &entries,
        "space/post/0/trace/1",
        gpui::Role::Button,
        "Show what this turn did",
        "Ferris — declined to respond · 1 tool call",
    );

    // Each opens on its own, and its rounds are its own.
    cx.update(|cx| {
        space.update(cx, |s, cx| s.toggle_trace("t-ferris", cx));
    });
    let entries = fresh_entries(cx, window);
    assert_probe(
        &entries,
        "space/post/0/trace/0",
        gpui::Role::Button,
        "Show what this turn did",
    );
    assert_probe(
        &entries,
        "space/post/0/trace/1",
        gpui::Role::Button,
        "Hide what this turn did",
    );
    assert!(
        !entries
            .iter()
            .any(|(n, _)| n == "space/post/0/trace/0/round/1"),
        "the turn that was not opened reveals nothing"
    );
    assert_probe(
        &entries,
        "space/post/0/trace/1/round/2",
        gpui::Role::ListItem,
        "Round 2: declined — Mara covered it",
    );

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn keyboard_focus_reveals_a_posts_hover_gated_verbs(cx: &mut TestAppContext) {
    // Wave B / audit S7. The Edit and Regenerate verbs are hover-gated, and
    // gpui suppresses hover entirely while the input modality is keyboard — so
    // without a focus-within reveal they are unreachable by exactly the user
    // the keyboard model exists for. Asserted through the probe registry: the
    // verb is *rendered*, not merely conceptually revealed.
    let _guard = probes_on();

    let stores = ready_stores(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, Some("s".into()), WindowInput::new(cx), window, cx))
    });
    let space = view.read_with(cx, |v, _| v.space().clone());
    cx.update(|cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![probe_post("a1", "the quick brown fox")], cx)
        });
    });

    let entries = fresh_entries(cx, window);
    assert!(
        !entries.iter().any(|(n, _)| n == "space/post/0/edit"),
        "the verb is hidden at rest"
    );

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.focus_post("a1".into(), window, cx));
    })
    .unwrap();
    let entries = fresh_entries(cx, window);
    assert_probe(
        &entries,
        "space/post/0/edit",
        gpui::Role::Button,
        "Edit this post",
    );
}

#[gpui::test]
fn compact_actionable_post_keeps_its_height_when_actions_reveal(cx: &mut TestAppContext) {
    use gpui::{VisualTestContext, px};

    let _guard = probes_on();
    let stores = ready_stores(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, Some("s".into()), WindowInput::new(cx), window, cx))
    });
    let space = view.read_with(cx, |v, _| v.space().clone());
    cx.update(|cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![probe_post("a1", "the quick brown fox")], cx)
        });
    });
    let vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(560.)));
    vcx.run_until_parked();

    let resting = fresh_entries(cx, window);
    let resting_height = resting
        .iter()
        .find(|(name, _)| name == "space/post/0")
        .expect("post probe at rest")
        .1
        .bounds
        .size
        .height;
    assert!(
        !resting.iter().any(|(name, _)| name == "space/post/0/edit"),
        "the reserved row remains visually empty at rest"
    );

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.focus_post("a1".into(), window, cx));
    })
    .unwrap();
    let focused = fresh_entries(cx, window);
    let focused_height = focused
        .iter()
        .find(|(name, _)| name == "space/post/0")
        .expect("focused post probe")
        .1
        .bounds
        .size
        .height;
    assert!(
        (resting_height - focused_height).abs() < px(0.5),
        "revealing compact actions must not resize the post ({resting_height:?} -> {focused_height:?})"
    );
    assert_probe(
        &focused,
        "space/post/0/edit",
        gpui::Role::Button,
        "Edit this post",
    );
}

#[gpui::test]
fn compact_settled_post_keeps_its_action_height_while_another_turn_streams(
    cx: &mut TestAppContext,
) {
    use gpui::{VisualTestContext, px};

    let _guard = probes_on();
    let stores = ready_stores(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, Some("s".into()), WindowInput::new(cx), window, cx))
    });
    let space = view.read_with(cx, |v, _| v.space().clone());
    cx.update(|cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![probe_post("a1", "the quick brown fox")], cx)
        });
    });
    let vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(560.)));
    vcx.run_until_parked();

    let settled_height = fresh_entries(cx, window)
        .iter()
        .find(|(name, _)| name == "space/post/0")
        .expect("settled post probe")
        .1
        .bounds
        .size
        .height;

    cx.update(|cx| {
        space.update(cx, |s, cx| {
            s.push_streaming_turn_for_test(None, Some("a1".into()), Default::default(), cx);
        });
    });
    vcx.run_until_parked();

    let streaming_entries = fresh_entries(cx, window);
    let streaming_height = streaming_entries
        .iter()
        .find(|(name, _)| name == "space/post/0")
        .expect("settled post probe while its reply streams")
        .1
        .bounds
        .size
        .height;
    assert!(
        (settled_height - streaming_height).abs() < px(0.5),
        "a settled actionable post must keep its compact action allocation while a reply streams \
         ({settled_height:?} -> {streaming_height:?})"
    );
    assert!(
        !streaming_entries
            .iter()
            .any(|(name, _)| name == "space/post/0/edit"),
        "the unavailable verb stays hidden while streaming"
    );
}

#[gpui::test]
fn compact_post_metadata_stays_inside_its_row_at_large_type(cx: &mut TestAppContext) {
    use gpui::{VisualTestContext, px};

    let _guard = probes_on();
    let stores = ready_stores(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, Some("s".into()), WindowInput::new(cx), window, cx))
    });
    let author = "A deliberately descriptive participant identity that remains recognizable";
    let backend = "private-inference-workstation-in-the-west-studio";
    let selection = format!("{author}@{backend}");
    let mut post = probe_post("a1", "the metadata must not paint into this prose");
    post.participant = PostParticipant {
        kind: "agent".into(),
        label: selection.clone(),
    };
    post.action_type = "inference".into();
    post.model = Some(selection);
    post.created_at = 1_700_000_000;
    let space = view.read_with(cx, |v, _| v.space().clone());
    cx.update(|cx| {
        space.update(cx, |s, cx| s.set_post_tree_for_test(vec![post], cx));
    });
    cx.update(|cx| gpui_component::Theme::global_mut(cx).font_size = px(24.));
    let vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(560.)));
    vcx.run_until_parked();

    let entries = fresh_entries(cx, window);
    let entry = |name: &str| {
        &entries
            .iter()
            .find(|(candidate, _)| candidate == name)
            .unwrap_or_else(|| panic!("missing {name}"))
            .1
    };
    let row = entry("space/post/a1/metadata");
    assert!(
        row.bounds.size.height > px(25.),
        "the probe exercises increased type scale"
    );
    for name in [
        "space/post/a1/metadata/author",
        "space/post/a1/metadata/backend",
        "space/post/a1/metadata/time",
    ] {
        let segment = entry(name);
        assert!(
            segment.bounds.left() >= row.bounds.left()
                && segment.bounds.right() <= row.bounds.right(),
            "{name} must stay inside the compact metadata row: {:?} vs {:?}",
            segment.bounds,
            row.bounds
        );
        assert!(
            segment.bounds.size.width > px(0.),
            "{name} retains a visible allocation"
        );
    }

    let article = entry("space/post/0");
    assert!(
        article.label.contains(author) && article.label.contains(backend),
        "visual truncation must not shorten the accessible byline: {:?}",
        article.label
    );

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn compact_composer_gutters_hold_post_parity_and_the_bottom_bar(cx: &mut TestAppContext) {
    use gpui::{VisualTestContext, px};

    let _guard = probes_on();
    let stores = ready_stores(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, Some("s".into()), WindowInput::new(cx), window, cx))
    });
    let space = view.read_with(cx, |v, _| v.space().clone());
    let body = "A paragraph long enough that the conversation overflows the window. ".repeat(30);
    cx.update(|cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![probe_post("a1", &body)], cx)
        });
    });
    let mut vcx = VisualTestContext::from_window(window, cx);
    const WIN: f32 = 560.0;
    vcx.simulate_resize(gpui::size(px(760.), px(WIN)));
    vcx.run_until_parked();
    // Activate the branch's own auto-minted tail draft (a seeded sibling
    // would sit off-path and float forever) and give it a line so the
    // actions reveal.
    let parents = view.read_with(&vcx, |v, _| v.draft_parents_for_test());
    let tail = parents
        .iter()
        .position(|parent| parent.as_deref() == Some("a1"))
        .expect("the leaf's tail draft exists");
    view.update(&mut vcx, |v, cx| v.activate_draft_for_test(tail, cx));
    vcx.run_until_parked();
    let editor = view
        .read_with(&vcx, |v, _| v.composer_state_for_test())
        .expect("the tail draft is active");
    editor.update(&mut vcx, |editor, cx| editor.set_value("a short draft", cx));
    vcx.run_until_parked();

    // Docked at the document floor: the quad fills the window exactly, the
    // byline sits a full post pad below the slot top (where a post's metadata
    // row sits), and the verbs ride the bottom action bar with real clearance
    // from the window edge.
    // Re-assert the floor each frame while estimated heights converge to
    // measured ones (the floor deepens as they settle).
    for _ in 0..4 {
        view.read_with(&vcx, |v, _| v.scroll_page_to_end_for_test());
        vcx.update(|window, _| window.refresh());
        vcx.run_until_parked();
    }
    let rem_size = vcx.update(|window, _| window.rem_size().as_f32());
    let bar_h = view.read_with(&vcx, |v, _| {
        v.compact_action_occupancy_for_test(760.0, rem_size)
    });
    let entries = fresh_entries(cx, window);
    let entry = |name: &str| {
        &entries
            .iter()
            .find(|(candidate, _)| candidate == name)
            .unwrap_or_else(|| panic!("missing {name}"))
            .1
    };
    let quad = entry("space/composer").bounds;
    assert!(
        quad.origin.y.abs() < px(0.5) && (quad.size.height - px(WIN)).abs() < px(0.5),
        "at the floor the docked quad fills the window exactly: {quad:?}"
    );
    let byline = entry("space/composer/byline").bounds;
    assert!(
        (byline.origin.y - quad.origin.y - px(40.)).abs() < px(1.0),
        "the docked byline sits POST_PAD_Y below the slot top, where a post's \
         metadata row sits (byline {byline:?}, quad {quad:?})"
    );
    let post = entry("space/composer/post").bounds;
    assert!(
        post.origin.y >= px(WIN - bar_h) && post.origin.y + post.size.height <= px(WIN - 8.),
        "Post rides the bottom action bar with clearance from the window edge \
         (post {post:?}, bar {}..{WIN})",
        WIN - bar_h
    );

    // Deactivated at the floor: the bottom bar survives the editor losing
    // its session — rendered by the inactive-tail path, fading in with the
    // slot — so Post meets the reader without a click into the editor first.
    vcx.update(|window, cx| {
        use gpui::Focusable as _;
        let focus = editor.read(cx).focus_handle(cx);
        window.focus(&focus, cx);
    });
    vcx.run_until_parked();
    vcx.simulate_keystrokes("escape");
    vcx.run_until_parked();
    let entries = fresh_entries(cx, window);
    assert!(
        !entries.iter().any(|(name, _)| name == "space/composer"),
        "escape retired the active composer overlay"
    );
    let post = entries
        .iter()
        .find(|(name, _)| name == "space/composer/post")
        .expect("the inactive tail draft keeps its bottom action bar")
        .1
        .bounds;
    assert!(
        post.origin.y >= px(WIN - bar_h) && post.origin.y + post.size.height <= px(WIN - 8.),
        "the inactive bar anchors to the same bottom allocation (post {post:?})"
    );

    // Floating: the byline is not part of the bar (its probe never paints),
    // while the verbs stay on the bottom bar and See in context joins them.
    view.update(&mut vcx, |v, cx| v.activate_draft_for_test(tail, cx));
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| v.set_page_scroll_for_test(0.0));
    vcx.run_until_parked();
    for _ in 0..2 {
        vcx.update(|window, _| window.refresh());
        vcx.run_until_parked();
    }
    let entries = fresh_entries(cx, window);
    assert!(
        view.read_with(&vcx, |v, _| v.composer_overlayed_for_test()),
        "scrolled to the top, the composer floats"
    );
    assert!(
        !entries
            .iter()
            .any(|(name, _)| name == "space/composer/byline"),
        "the floating bar carries no byline"
    );
    let post = entries
        .iter()
        .find(|(name, _)| name == "space/composer/post")
        .expect("Post stays on the bottom bar while floating")
        .1
        .bounds;
    assert!(
        post.origin.y >= px(WIN - bar_h) && post.origin.y + post.size.height <= px(WIN - 8.),
        "the floating bar keeps its verbs on the bottom action bar (post {post:?})"
    );
    assert_probe(
        &entries,
        "space/composer/home",
        gpui::Role::Button,
        "See in context",
    );

    // Large type: the settled byline anchor derives from the editor's offset,
    // not the separator height, so post parity (POST_PAD_Y under the slot
    // top) holds at every scale — and the taller action bar still keeps its
    // verbs inside the window with clearance.
    cx.update(|cx| gpui_component::Theme::global_mut(cx).font_size = px(24.));
    for _ in 0..4 {
        view.read_with(&vcx, |v, _| v.scroll_page_to_end_for_test());
        vcx.update(|window, _| window.refresh());
        vcx.run_until_parked();
    }
    let rem_size = vcx.update(|window, _| window.rem_size().as_f32());
    assert!(
        rem_size > 20.0,
        "precondition: the window rem follows the enlarged theme ({rem_size})"
    );
    let bar_h = view.read_with(&vcx, |v, _| {
        v.compact_action_occupancy_for_test(760.0, rem_size)
    });
    let entries = fresh_entries(cx, window);
    let entry = |name: &str| {
        &entries
            .iter()
            .find(|(candidate, _)| candidate == name)
            .unwrap_or_else(|| panic!("missing {name}"))
            .1
    };
    let quad = entry("space/composer").bounds;
    let byline = entry("space/composer/byline").bounds;
    assert!(
        (byline.origin.y - quad.origin.y - px(40.)).abs() < px(1.0),
        "at 24px rem the docked byline still tops out POST_PAD_Y under the \
         slot top (byline {byline:?}, quad {quad:?})"
    );
    let post = entry("space/composer/post").bounds;
    assert!(
        post.origin.y >= px(WIN - bar_h) && post.origin.y + post.size.height <= px(WIN - 8.),
        "the scaled bar still holds its verbs clear of the edge (post {post:?})"
    );
}

#[gpui::test]
fn entering_compact_affordances_reveals_an_oversized_posts_action(cx: &mut TestAppContext) {
    use gpui::{VisualTestContext, px};

    let _guard = probes_on();
    let stores = ready_stores(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, Some("s".into()), WindowInput::new(cx), window, cx))
    });
    let space = view.read_with(cx, |v, _| v.space().clone());
    let body =
        "A long paragraph that keeps wrapping through the compact reading column. ".repeat(90);
    cx.update(|cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![probe_post("a1", &body)], cx)
        });
    });
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(560.)));
    vcx.run_until_parked();

    vcx.simulate_keystrokes("down");
    vcx.run_until_parked();
    let before = fresh_entries(cx, window);
    let action = before
        .iter()
        .find(|(name, _)| name == "space/post/0/edit")
        .expect("focus reveals Edit on the oversized post")
        .1
        .bounds;
    assert!(
        action.origin.y + action.size.height > px(560.),
        "precondition: post-level focus aligns the oversized post's top and leaves its action below the window"
    );
    let before_offset = view.read_with(&vcx, |v, _| v.page_scroll_offset_y_for_test());

    vcx.simulate_keystrokes("enter");
    vcx.run_until_parked();
    let after = fresh_entries(cx, window);
    let action = after
        .iter()
        .find(|(name, _)| name == "space/post/0/edit")
        .expect("entered Edit affordance remains rendered")
        .1
        .bounds;
    assert!(
        action.origin.y >= px(0.) && action.origin.y + action.size.height <= px(560.),
        "the focused compact action is inside the window: {action:?}"
    );
    view.read_with(&vcx, |v, _| {
        assert!(
            v.page_scroll_offset_y_for_test() < before_offset,
            "entering the action row minimally advances the page"
        );
    });
}

#[gpui::test]
fn keyboard_focus_reveals_a_library_rows_verbs(cx: &mut TestAppContext) {
    // The Library's half of audit S7 — the same rule as a post's action gutter:
    // hover-gated verbs must also answer to keyboard focus, because gpui
    // suppresses hover outright while the input modality is keyboard.
    let _guard = probes_on();

    let stores = stub_stores(cx, |s| {
        s.spaces = vec![
            space_info("s1", Some("Tides and the moon")),
            space_info("s2", Some("Borrow checker")),
        ];
    });
    let (window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| LibraryView::new(stores, window, cx))
    });

    let entries = fresh_entries(cx, window);
    assert!(
        !entries.iter().any(|(n, _)| n == "library/row/0/rename"),
        "the verbs are hidden at rest"
    );

    // Tab into the listing: the first stop is the first row.
    cx.update_window(window, |_, window, cx| window.focus_next(cx))
        .unwrap();
    let entries = fresh_entries(cx, window);
    assert_probe(
        &entries,
        "library/row/0/rename",
        gpui::Role::Button,
        "Rename Tides and the moon",
    );
}

/// Press Enter as a real keyboard activation: gpui maps it on **key up** (the
/// key-down only records the pending activation), and `TestAppContext`'s
/// `dispatch_keystroke` sends the down alone.
fn press_enter(cx: &mut TestAppContext, window: AnyWindowHandle) {
    let ks = gpui::Keystroke::parse("enter").unwrap();
    cx.update_window(window, |_, window, cx| {
        window.dispatch_event(
            gpui::PlatformInput::KeyDown(gpui::KeyDownEvent {
                keystroke: ks.clone(),
                is_held: false,
                prefer_character_input: false,
            }),
            cx,
        );
        window.dispatch_event(
            gpui::PlatformInput::KeyUp(gpui::KeyUpEvent { keystroke: ks }),
            cx,
        );
    })
    .unwrap();
    cx.run_until_parked();
}

#[gpui::test]
fn tab_reaches_a_control_that_enter_actually_activates(cx: &mut TestAppContext) {
    // The P1 the focus model had to answer: gpui's Enter/Space activation runs
    // **only the focused element's own click listeners** (and registers the
    // whole keyboard-click block only when it has some), so a probed *wrapper*
    // around a `gpui_component::Button` is a tab stop that can never fire —
    // it rings, eats a Tab, and does nothing, with the working control one Tab
    // further on. `probe_delegating` takes the wrapper out of the tab order so
    // Tab lands on the button that owns the handler.
    //
    // Driven end to end: Tab into the listing, Tab to the focused row's verbs,
    // press Enter, and the space is archived.
    let _guard = probes_on();

    let stores = stub_stores(cx, |s| {
        s.spaces = vec![
            space_info("s1", Some("Tides and the moon")),
            space_info("s2", Some("Borrow checker")),
        ];
    });
    let spaces = stores.spaces.clone();
    let (window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| LibraryView::new(stores, window, cx))
    });

    // Stop 1: the listing itself (one tab stop, roving cursor on row 0), which
    // is what reveals that row's verbs.
    cx.update_window(window, |_, window, cx| window.focus_next(cx))
        .unwrap();
    let entries = fresh_entries(cx, window);
    assert_probe(
        &entries,
        "library/row/0/archive",
        gpui::Role::Button,
        "Archive Tides and the moon",
    );

    // Stops 2 and 3: the focused row's Rename then Archive. Exactly two — a
    // probed wrapper before each would make it four, and the second press
    // would land on the dead rename wrapper instead.
    cx.update_window(window, |_, window, cx| {
        window.focus_next(cx);
        window.focus_next(cx);
    })
    .unwrap();
    press_enter(cx, window);

    let titles: Vec<String> = spaces.read_with(cx, |s, _| {
        s.list().iter().filter_map(|s| s.title.clone()).collect()
    });
    assert_eq!(
        titles,
        vec!["Borrow checker".to_string()],
        "Enter on the focused Archive verb archived the space"
    );
}

#[gpui::test]
fn library_arrows_rove_the_listing_and_enter_opens_a_space(cx: &mut TestAppContext) {
    // `uniform_list` materializes only the visible window, so a tab stop per
    // row is a tab order that cannot contain the rows you haven't scrolled to
    // — Tab simply walked off the end of the visible slice and left the
    // library. The listing is therefore one tab stop with a roving cursor (the
    // shape the space tree already ships): ↑/↓/Home/End move the cursor, the
    // focused row reveals its verbs, and Enter opens it.
    let _guard = probes_on();

    let stores = stub_stores(cx, |s| {
        s.spaces = vec![
            space_info("s1", Some("Tides and the moon")),
            space_info("s2", Some("Borrow checker")),
            space_info("s3", Some("Sourdough")),
        ];
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| LibraryView::new(stores, window, cx))
    });

    cx.update_window(window, |_, window, cx| window.focus_next(cx))
        .unwrap();
    let entries = fresh_entries(cx, window);
    view.read_with(cx, |v, cx| assert_eq!(v.focused_row_for_test(cx), Some(0)));
    assert!(
        entries.iter().any(|(n, _)| n == "library/row/0/rename")
            && !entries.iter().any(|(n, _)| n == "library/row/1/rename"),
        "only the cursor's row reveals its verbs"
    );

    cx.simulate_keystrokes(window, "down");
    let entries = fresh_entries(cx, window);
    view.read_with(cx, |v, cx| assert_eq!(v.focused_row_for_test(cx), Some(1)));
    assert_probe(
        &entries,
        "library/row/1/rename",
        gpui::Role::Button,
        "Rename Borrow checker",
    );
    assert!(
        !entries.iter().any(|(n, _)| n == "library/row/0/rename"),
        "the cursor moved, so row 0's verbs went with it"
    );

    cx.simulate_keystrokes(window, "end");
    view.read_with(cx, |v, cx| assert_eq!(v.focused_row_for_test(cx), Some(2)));
    cx.simulate_keystrokes(window, "down");
    view.read_with(cx, |v, cx| {
        assert_eq!(
            v.focused_row_for_test(cx),
            Some(2),
            "the last row is the end"
        )
    });
    cx.simulate_keystrokes(window, "home");
    view.read_with(cx, |v, cx| assert_eq!(v.focused_row_for_test(cx), Some(0)));
    cx.simulate_keystrokes(window, "up");
    view.read_with(cx, |v, cx| {
        assert_eq!(
            v.focused_row_for_test(cx),
            Some(0),
            "the first row is the top"
        )
    });

    cx.simulate_keystrokes(window, "down enter");
    view.read_with(cx, |v, _| {
        assert_eq!(
            v.open_space_requests_for_test(),
            1,
            "Enter opened the row the cursor was on"
        );
    });
}

/// Walk the window's whole Tab cycle and return how many distinct stops it
/// contains. `focus_next` wraps, so a repeat ends the walk — and it is a no-op
/// when there is nothing to reach, so an *unchanged* focus ends it too (the
/// window's root handle is tracked but is not itself a stop, and would
/// otherwise be miscounted as one).
fn tab_stop_count(cx: &mut TestAppContext, window: AnyWindowHandle) -> usize {
    draw(cx, window);
    let focused = |cx: &mut TestAppContext| {
        cx.update_window(window, |_, window, cx| {
            window.focused(cx).map(|h| format!("{h:?}"))
        })
        .unwrap()
    };
    let mut seen: Vec<String> = Vec::new();
    let mut prev = focused(cx);
    for _ in 0..200 {
        cx.update_window(window, |_, window, cx| window.focus_next(cx))
            .unwrap();
        let id = focused(cx);
        if id == prev {
            break;
        }
        let Some(id) = id else { break };
        if seen.contains(&id) {
            break;
        }
        seen.push(id.clone());
        prev = Some(id);
    }
    seen.len()
}

#[gpui::test]
fn a_listing_contributes_one_tab_stop_regardless_of_row_count(cx: &mut TestAppContext) {
    // The classification cure for `Role::ListItem`. `listitem` is a
    // *structural* role, and treating it as interactive handed a tab stop to
    // every static row (a declined trace round, a footnote with nowhere to
    // navigate) while the clickable rows it was meant for live in
    // `uniform_list`s, where a per-row stop can only ever reach the
    // materialized window. Both listings rove instead — so the window's tab
    // order does not grow with the data.
    let _guard = probes_on();

    let stores = stub_stores(cx, |_| {});
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| RecordView::new(stores, window, cx))
    });
    view.update(cx, |v, cx| {
        v.select_section(RecordSection::Requests, cx);
        v.set_requests_for_test(vec![stub_request("req-1")], false);
    });
    let one = tab_stop_count(cx, window);

    view.update(cx, |v, _| {
        v.set_requests_for_test(
            (1..=5).map(|i| stub_request(&format!("req-{i}"))).collect(),
            false,
        );
    });
    let five = tab_stop_count(cx, window);

    assert_eq!(
        one, five,
        "five rows must contribute exactly what one row does: the listing's own stop"
    );
}

#[gpui::test]
fn record_arrows_rove_the_listing_and_enter_opens_a_detail(cx: &mut TestAppContext) {
    // The Record's listings are virtualized too, so they take the Library's
    // shape: one tab stop, a roving cursor, Enter opens the row it sits on.
    let _guard = probes_on();

    let stores = stub_stores(cx, |_| {});
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| RecordView::new(stores, window, cx))
    });
    view.update(cx, |v, cx| {
        v.select_section(RecordSection::Requests, cx);
        v.set_requests_for_test(
            (1..=3).map(|i| stub_request(&format!("req-{i}"))).collect(),
            false,
        );
    });
    draw(cx, window);
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.focus_listing_for_test(window, cx));
    })
    .unwrap();
    view.read_with(cx, |v, _| assert_eq!(v.focused_row_for_test(), Some(0)));

    cx.simulate_keystrokes(window, "down");
    view.read_with(cx, |v, _| assert_eq!(v.focused_row_for_test(), Some(1)));
    cx.simulate_keystrokes(window, "end");
    view.read_with(cx, |v, _| assert_eq!(v.focused_row_for_test(), Some(2)));
    cx.simulate_keystrokes(window, "home");
    view.read_with(cx, |v, _| assert_eq!(v.focused_row_for_test(), Some(0)));

    cx.simulate_keystrokes(window, "down enter");
    view.read_with(cx, |v, _| {
        assert_eq!(
            v.detail_pending(),
            Some("req-2"),
            "Enter opened the detail of the row the cursor was on"
        );
    });
}

#[gpui::test]
fn library_cursor_clamps_when_its_row_disappears(cx: &mut TestAppContext) {
    // Rows come and go under a roving cursor — archive the one it sits on and
    // a stored index points one past the end, where Enter is dead and no row
    // draws the ring. The effective cursor is clamped on read instead, so it
    // lands on the new last row and stays live.
    let _guard = probes_on();

    let stores = stub_stores(cx, |s| {
        s.spaces = vec![
            space_info("s1", Some("Tides and the moon")),
            space_info("s2", Some("Borrow checker")),
        ];
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| LibraryView::new(stores, window, cx))
    });

    cx.update_window(window, |_, window, cx| window.focus_next(cx))
        .unwrap();
    cx.simulate_keystrokes(window, "end");
    view.read_with(cx, |v, cx| assert_eq!(v.focused_row_for_test(cx), Some(1)));

    view.update(cx, |v, cx| v.archive("s2".into(), cx));
    view.read_with(cx, |v, cx| {
        assert_eq!(
            v.focused_row_for_test(cx),
            Some(0),
            "the cursor lands on the new last row rather than past the end"
        );
    });

    // …and it is still live: the row it now sits on opens.
    let entries = fresh_entries(cx, window);
    assert_probe(
        &entries,
        "library/row/0/rename",
        gpui::Role::Button,
        "Rename Tides and the moon",
    );
    cx.simulate_keystrokes(window, "enter");
    view.read_with(cx, |v, _| {
        assert_eq!(v.open_space_requests_for_test(), 1, "Enter still opens");
    });
}

#[gpui::test]
fn a_disabled_control_is_not_a_tab_stop(cx: &mut TestAppContext) {
    use eidola_gui::updates::UpdatesView;
    // The hazard the activation hoist introduces: with the handler on the
    // probed wrapper, the widget's own disabled state no longer guards the
    // keyboard. (A *pointer* press is still refused inside the widget, which
    // stops propagation on mouse-down while disabled, so the wrapper never
    // arms.) So the wrapper mirrors it: disabled means no tab stop.
    let _guard = probes_on();

    let idle = stub_stores(cx, |s| s.update_checking = false);
    let (idle_window, _v) = open_view(cx, |window, cx| {
        cx.new(|cx| UpdatesView::new(idle, window, cx))
    });
    let entries = fresh_entries(cx, idle_window);
    assert_probe(&entries, "updates/check", gpui::Role::Button, "Check Now");
    let idle_stops = tab_stop_count(cx, idle_window);

    let busy = stub_stores(cx, |s| s.update_checking = true);
    let (busy_window, _v) = open_view(cx, |window, cx| {
        cx.new(|cx| UpdatesView::new(busy, window, cx))
    });
    let entries = fresh_entries(cx, busy_window);
    assert_probe(&entries, "updates/check", gpui::Role::Button, "Checking…");
    let busy_stops = tab_stop_count(cx, busy_window);

    assert_eq!(
        busy_stops,
        idle_stops - 1,
        "the check affordance is still announced, but leaves the tab order while \
         it cannot be actioned"
    );
}

#[gpui::test]
fn library_cursor_yields_to_a_verb_it_tabs_into(cx: &mut TestAppContext) {
    // Two questions, two predicates. Tab from the listing into the cursor row's
    // Rename verb makes the *verb* the focused element — it paints its own ring
    // — so the row must stop claiming the cursor, or one focus is indicated
    // twice and AT is told a row is focused while a button inside it really is.
    // The verbs must nevertheless stay revealed: they are what the Tab reached.
    let _guard = probes_on();

    let stores = stub_stores(cx, |s| {
        s.spaces = vec![
            space_info("s1", Some("Tides and the moon")),
            space_info("s2", Some("Borrow checker")),
        ];
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| LibraryView::new(stores, window, cx))
    });

    cx.update_window(window, |_, window, cx| window.focus_next(cx))
        .unwrap();
    let entries = fresh_entries(cx, window);
    assert!(entries.iter().any(|(n, _)| n == "library/row/0/rename"));
    let (cursor, revealed) = cx
        .update_window(window, |_, window, cx| {
            view.read_with(cx, |v, cx| v.cursor_and_reveal_for_test(window, cx))
        })
        .unwrap();
    assert_eq!(
        (cursor, revealed),
        (Some(0), Some(0)),
        "the listing itself holds focus: it is both the cursor and the reveal"
    );

    // Tab on to the row's Rename verb.
    cx.update_window(window, |_, window, cx| window.focus_next(cx))
        .unwrap();
    let entries = fresh_entries(cx, window);
    let (cursor, revealed) = cx
        .update_window(window, |_, window, cx| {
            view.read_with(cx, |v, cx| v.cursor_and_reveal_for_test(window, cx))
        })
        .unwrap();
    assert_eq!(cursor, None, "the verb owns the focus indication now");
    assert_eq!(revealed, Some(0), "…and its row keeps showing the verbs");
    assert!(
        entries.iter().any(|(n, _)| n == "library/row/0/archive"),
        "including the one Tab has not reached yet"
    );
}

#[gpui::test]
fn a_faded_minimap_contributes_no_tab_stops(cx: &mut TestAppContext) {
    // The minimap's cells take their press handler only while the strip is up
    // (a faded 36px strip must contain nothing), so at rest each is a
    // `Role::Button` with no click listener of its own — and gpui's Enter/Space
    // runs only the focused element's own listeners. Left as stops they were N
    // dead ones at the end of every space window's tab order.
    let _guard = probes_on();

    let stores = ready_stores(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, Some("s".into()), WindowInput::new(cx), window, cx))
    });
    let space = view.read_with(cx, |v, _| v.space().clone());
    cx.update(|cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(
                vec![
                    probe_post("a1", "the root post"),
                    probe_post("a2", "the reply"),
                ],
                cx,
            )
        });
    });

    let entries = fresh_entries(cx, window);
    let cells = entries
        .iter()
        .filter(|(n, _)| n.starts_with("space/minimap/cell/"))
        .count();
    assert!(cells > 0, "the fixture really does render minimap cells");
    let faded = tab_stop_count(cx, window);

    view.update(cx, |v, cx| v.set_minimap_visible_for_test(true, cx));
    let live = tab_stop_count(cx, window);

    assert_eq!(
        faded + cells,
        live,
        "every cell is a stop while the map is up and none while it is faded"
    );
}

#[gpui::test]
fn record_spend_group_header_is_a_readable_node(cx: &mut TestAppContext) {
    // The roving cursor walks display rows, and a spending group header is one
    // of them — navigable, not activatable. `aria_active_descendant` resolves
    // through the a11y *node* tree, and gpui pushes a node only for an element
    // carrying a role, so without one the pointer at a header is silently
    // dropped. `Label` gives it a node and keeps it out of the tab order.
    let _guard = probes_on();

    let stores = stub_stores(cx, |_| {});
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| RecordView::new(stores, window, cx))
    });
    view.update(cx, |v, cx| {
        v.select_section(RecordSection::Spending, cx);
        v.set_spending_for_test(vec![stub_spend("req-1")], false);
    });

    let entries = fresh_entries(cx, window);
    let (_, header) = entries
        .iter()
        .find(|(n, _)| n == "record/spend/header/0")
        .expect("the group header carries an a11y node");
    assert_eq!(header.role, gpui::Role::Label);
    assert!(
        header.label.contains("credential"),
        "and reads as what it captions: {:?}",
        header.label
    );
}

/// Two regimes for `gpui-component` widgets that self-annotate for AccessKit
/// at our fork rev, each enforced **immediately after the constructor** (the
/// placement is what makes this source scan reliable):
///
/// - **Hoisted controls** (`Button`, `Checkbox`): the probed wrapper is the
///   accessible control (role, label, focus, activation), so the widget is
///   made presentational via `.role(None)` — without it AT sees two nodes for
///   one control.
/// - **Focus-bearing editors** (`Input`): the widget owns the tracked focus
///   handle, so *it* must be the node or AT reports focus on the window root
///   — it carries `.aria_label(..)` and the wrapper is a bounds-only probe
///   (`probe_bounds`, no node).
///
/// The emitted AccessKit `TreeUpdate` is crate-private, so neither invariant
/// can be asserted against the tree itself; this scan is the gate. `Switch`
/// is deliberately absent: it sets no role at our rev (comments at its two
/// call sites carry the tripwire).
#[test]
fn self_annotating_widgets_opt_out_of_their_own_a11y_nodes() {
    let src_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut offenders = Vec::new();
    let mut stack = vec![src_root];
    let mut seen = 0usize;
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap();
            for needle in ["Button::new(", "Input::new(", "Checkbox::new("] {
                let mut from = 0;
                while let Some(pos) = text[from..].find(needle) {
                    let start = from + pos;
                    // Skip matches that are part of a longer path segment
                    // (e.g. `WindowInput::new`): require a non-identifier
                    // character before the widget name.
                    let named_ok = text[..start]
                        .chars()
                        .next_back()
                        .is_none_or(|c| !c.is_alphanumeric() && c != '_');
                    // Walk to the constructor's matching close paren, then
                    // require the opt-out as the next builder call.
                    let open = start + needle.len() - 1;
                    let mut depth = 0usize;
                    let mut end = None;
                    for (i, c) in text[open..].char_indices() {
                        match c {
                            '(' => depth += 1,
                            ')' => {
                                depth -= 1;
                                if depth == 0 {
                                    end = Some(open + i + 1);
                                    break;
                                }
                            }
                            _ => {}
                        }
                    }
                    if named_ok {
                        seen += 1;
                        let rest = &text[end.expect("unbalanced parens")..];
                        let rest: String = rest.chars().filter(|c| !c.is_whitespace()).collect();
                        let required = if needle == "Input::new(" {
                            ".aria_label("
                        } else {
                            ".role(None)"
                        };
                        if !rest.starts_with(required) {
                            let line = text[..start].matches('\n').count() + 1;
                            offenders.push(format!(
                                "{}:{line} (wants `{required}..` first)",
                                path.display()
                            ));
                        }
                    }
                    from = start + needle.len();
                }
            }
        }
    }
    assert!(
        seen >= 30,
        "the scan found only {seen} widget constructors — the needle set or \
         src layout changed; fix the scan rather than losing the gate"
    );
    assert!(
        offenders.is_empty(),
        "self-annotating widgets missing their regime's annotation \
         immediately after the constructor (Button/Checkbox: `.role(None)`, \
         the wrapper is the control; Input: `.aria_label(..)`, the widget is \
         the node — see AGENTS.md → Accessibility & QA probes):\n{}",
        offenders.join("\n")
    );
}

// ---------------------------------------------------------------------------
// The space inspector (task 26.2) — the per-space settings panel. Its rows are
// ordinary probed affordances; the one that has to be exactly right is the
// router's cost note, which is what discloses per-post billing.
// ---------------------------------------------------------------------------

fn inspector_stores(cx: &mut TestAppContext, settings: eidola_app_core::SpaceSettings) -> Stores {
    stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        s.space_settings = Some(("s".into(), settings));
    })
}

fn open_inspector(
    cx: &mut TestAppContext,
    settings: eidola_app_core::SpaceSettings,
) -> (AnyWindowHandle, Entity<SpaceView>) {
    let stores = inspector_stores(cx, settings);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, Some("s".into()), WindowInput::new(cx), window, cx))
    });
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.set_inspector_open_for_test(true, window, cx))
    })
    .unwrap();
    cx.run_until_parked();
    (window, view)
}

#[gpui::test]
fn space_inspector_probes_its_rows(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let (window, _view) = open_inspector(cx, eidola_app_core::SpaceSettings::default());
    let entries = fresh_entries(cx, window);

    // The panel is a complementary landmark: settings *about* the conversation.
    assert_probe(
        &entries,
        "space/inspector",
        gpui::Role::Complementary,
        "Inspector",
    );
    // The title field's node is the `Input` itself (the two-regime rule), so
    // the wrapper is bounds-only — role and label describe it for the driver.
    assert_probe(
        &entries,
        "space/inspector/title",
        gpui::Role::TextInput,
        "Space title",
    );
    // The number is the thing a reader most wants from a stepper, so it rides
    // its own settled `Label` node rather than living in a node-less div.
    assert_probe_value(
        &entries,
        "space/inspector/cascade",
        gpui::Role::Label,
        "Cascade limit",
        "4",
    );
    assert_probe(
        &entries,
        "space/inspector/cascade/dec",
        gpui::Role::Button,
        "Decrease cascade limit",
    );
    assert_probe(
        &entries,
        "space/inspector/cascade/inc",
        gpui::Role::Button,
        "Increase cascade limit",
    );
    // The picker announces its selection — "Off" is a normal choice, and the
    // label alone would say nothing about whether this space bills per post.
    assert_probe_value(
        &entries,
        "space/inspector/router",
        gpui::Role::Button,
        "Router model",
        "Off",
    );

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn space_inspector_router_cost_note_is_exactly_remote_conditional(cx: &mut TestAppContext) {
    let _guard = probes_on();

    // A local (engine-served) router is genuinely free — no cost copy.
    let (window, _view) = open_inspector(
        cx,
        eidola_app_core::SpaceSettings {
            cascade_limit: 4,
            router_model: Some("tiny@local".into()),
            ..Default::default()
        },
    );
    let entries = fresh_entries(cx, window);
    assert!(
        !entries
            .iter()
            .any(|(n, _)| n == "space/inspector/router/cost"),
        "a local router must not claim a per-call cost"
    );

    // A remote one bills an inference on every post, and says so inline.
    let (window, _view) = open_inspector(
        cx,
        eidola_app_core::SpaceSettings {
            cascade_limit: 4,
            router_model: Some("gemma4-31b@eidola".into()),
            ..Default::default()
        },
    );
    let entries = fresh_entries(cx, window);
    assert_probe(
        &entries,
        "space/inspector/router/cost",
        gpui::Role::Label,
        eidola_gui::space_view::inspector::ROUTER_REMOTE_COST_NOTE,
    );

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn space_inspector_failed_settings_read_offers_a_retry_not_a_default(cx: &mut TestAppContext) {
    // "Failed is not empty": a failed read must not render as cascade 4 /
    // router Off with live controls that would write over settings we never
    // managed to read.
    let _guard = probes_on();

    let stores = inspector_stores(cx, eidola_app_core::SpaceSettings::default());
    let settings_store = stores.space_settings.clone();
    cx.update(|cx| {
        settings_store.update(cx, |s, _| s.set_failed_for_test("s", "database is locked"))
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, Some("s".into()), WindowInput::new(cx), window, cx))
    });
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.set_inspector_open_for_test(true, window, cx))
    })
    .unwrap();
    cx.run_until_parked();

    let entries = fresh_entries(cx, window);
    assert_probe(
        &entries,
        "space/inspector/retry",
        gpui::Role::Button,
        "Retry",
    );
    assert!(
        !entries.iter().any(|(n, _)| n == "space/inspector/cascade"),
        "a failed read shows no plausible default"
    );

    probe::set_probes_enabled(false);
}

/// **Two failed sections, and each Retry has to be its own control.** The Space
/// section and the Participants section read different stores and fail
/// independently, so both `load_error_panel`s can stand in one panel — and the
/// shared helper used to hard-code its button's element id (`load-retry`) while
/// neither section root carries an id of its own, so the two buttons resolved to
/// the *same* `GlobalElementId`. gpui keys per-element state on that id (and
/// derives each AccessKit node id from its hash), so the pair shared one
/// `pending_mouse_down`: mouse-up dispatches its capture phase in paint order,
/// the Space panel's handler ran first, found the press the *Participants*
/// button had armed, saw its own hitbox unhovered and swallowed it — leaving the
/// second Retry inert in the one state where Retry is the only way forward
/// (`ensure` declines once a `Failed` cell exists). Codex review, PR #278.
///
/// Driven against a real core, because the proof is that the click *lands*: the
/// retry has to have something to succeed at.
#[gpui::test]
fn each_failed_section_gets_a_retry_of_its_own(cx: &mut TestAppContext) {
    use gpui::{Modifiers, VisualTestContext, px};

    let _guard = probes_on();
    cx.executor().allow_parking();
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    let dir = tempfile::tempdir().unwrap();
    let core = std::sync::Arc::new(
        eidola_app_core::AppCore::new(dir.path().to_path_buf(), dir.path().join("data"))
            .expect("open core"),
    );
    core.runtime()
        .block_on(core.set_base_url("https://127.0.0.1:1/v1".into()))
        .unwrap();
    let space = core
        .runtime()
        .block_on(core.create_space(None))
        .expect("create space")
        .id;
    let stores = cx.update(|cx| Stores::for_test(core.clone(), cx));

    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| {
            SpaceView::new(
                stores.clone(),
                Some(space.clone()),
                WindowInput::new(cx),
                window,
                cx,
            )
        })
    });
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.set_inspector_open_for_test(true, window, cx))
    })
    .unwrap();
    // Wide enough that the panel splits the window rather than floating over it.
    let vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(900.), px(700.)));
    vcx.run_until_parked();
    poll_until(cx, "both sections load", |cx| {
        stores
            .participants
            .read_with(cx, |s, _| !s.list(&space).is_empty())
    });

    // Both reads fail — two panels, two Retrys.
    cx.update(|cx| {
        stores
            .participants
            .update(cx, |s, _| s.set_failed_for_test(&space, "boom"));
        stores
            .space_settings
            .update(cx, |s, _| s.set_failed_for_test(&space, "boom"));
    });
    let entries = fresh_entries(cx, window);
    let bounds = |name: &str| {
        entries
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("{name} painted"))
            .1
            .bounds
    };
    let space_retry = bounds("space/inspector/retry");
    let participants_retry = bounds("space/inspector/participants/retry");
    assert_ne!(
        space_retry.center(),
        participants_retry.center(),
        "precondition: the two panels stand side by side"
    );

    // The real gesture, on the *second* panel's button.
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_click(participants_retry.center(), Modifiers::default());
    vcx.run_until_parked();

    poll_until(cx, "the participants read is retried", |cx| {
        stores
            .participants
            .read_with(cx, |s, _| s.participants(&space).has_value())
    });
    assert!(
        stores
            .space_settings
            .read_with(cx, |s, _| s.settings(&space).error().is_some()),
        "and the press belonged to the button it landed on — the Space section's \
         read was never retried"
    );

    probe::set_probes_enabled(false);
    while core.runtime().metrics().num_alive_tasks() > 0 {
        std::thread::yield_now();
    }
}

/// Poll `run_until_parked` until `pred` holds — a real-core read round-trips
/// through the tokio runtime, which `run_until_parked` alone can return before.
fn poll_until(
    cx: &mut TestAppContext,
    what: &str,
    mut pred: impl FnMut(&mut TestAppContext) -> bool,
) {
    for _ in 0..400 {
        cx.run_until_parked();
        if pred(cx) {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    panic!("timed out waiting until {what}");
}

/// **Neither refusal may stand in for the other.** The panel writes to two
/// stores — the title goes through `SpacesStore` (the Library index), the rows
/// through `SpaceSettingsStore` — and while one band chose between them, an
/// older settings refusal shadowed a newer rename refusal: the title field
/// snapped back to the stored name with nothing on screen to say why, and
/// nothing ever cleared the settings refusal to reveal it. Both bands render,
/// the title's first, and each carries its own dismiss.
#[gpui::test]
fn space_inspector_shows_a_title_refusal_beside_a_standing_settings_one(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let stores = stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        s.space_settings = Some(("s".into(), eidola_app_core::SpaceSettings::default()));
        s.spaces = vec![space_info("s", Some("Tides"))];
    });
    // An older settings refusal, standing (nothing has cleared it)…
    stores.space_settings.update(cx, |s, _| {
        s.set_op_error_for_test("s", "Couldn't set the cascade limit: below the floor")
    });
    // …and then a rename this space refused, which is what the reader just
    // watched undo itself in the title field.
    stores.spaces.update(cx, |s, cx| {
        s.settle_for_test(
            Some("s".into()),
            Ok(vec![space_info("s", Some("Tides"))]),
            Some("Couldn't rename this space: space not found: s"),
            cx,
        )
    });

    let view_stores = stores.clone();
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| {
            SpaceView::new(
                view_stores,
                Some("s".into()),
                WindowInput::new(cx),
                window,
                cx,
            )
        })
    });
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.set_inspector_open_for_test(true, window, cx))
    })
    .unwrap();
    cx.run_until_parked();

    let entries = fresh_entries(cx, window);
    assert_probe(
        &entries,
        "space/inspector/title-error",
        gpui::Role::Alert,
        "Couldn't rename this space: space not found: s",
    );
    assert_probe(
        &entries,
        "space/inspector/error",
        gpui::Role::Alert,
        "Couldn't set the cascade limit: below the floor",
    );
    // Each is acknowledged on its own — the × says "I have read this", and the
    // other fact stays on screen.
    assert_probe(
        &entries,
        "space/inspector/title-error/dismiss",
        gpui::Role::Button,
        "Dismiss",
    );
    assert_probe(
        &entries,
        "space/inspector/error/dismiss",
        gpui::Role::Button,
        "Dismiss",
    );

    stores
        .spaces
        .update(cx, |s, cx| s.dismiss_op_error_for("s", cx));
    let entries = fresh_entries(cx, window);
    assert!(
        !entries
            .iter()
            .any(|(n, _)| n == "space/inspector/title-error"),
        "the dismissed refusal leaves"
    );
    assert_probe(
        &entries,
        "space/inspector/error",
        gpui::Role::Alert,
        "Couldn't set the cascade limit: below the floor",
    );

    probe::set_probes_enabled(false);
}

/// **The destination list is bounded *and* virtualized.** An unbounded column
/// inside a popover clipped its own overflow, putting the far conversations out
/// of reach; a capped height alone fixed the reach and left the cost, building
/// an element — hover style, click closure, probe — for every conversation the
/// reader has ever had, on every frame the picker stood open (Codex review, PR
/// #280). The Library is a history, not a menu, so this takes the shape the
/// doctrine already names for fixed-height lists: `uniform_list` renders the
/// visible window and nothing else.
#[gpui::test]
fn space_quote_destination_list_is_bounded(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let many: Vec<_> = (0..60)
        .map(|i| space_info(&format!("s{i}"), Some(&format!("Conversation {i}"))))
        .collect();
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        s.eidola_trust = Some(probe_eidola_trust());
        s.spaces = many;
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, Some("s0".into()), WindowInput::new(cx), window, cx))
    });
    let space = view.read_with(cx, |v, _| v.space().clone());
    let mut post = probe_post("a1", "the quick brown fox");
    post.blocks[0].id = "b1".into();
    cx.update(|cx| {
        space.update(cx, |s, cx| s.set_post_tree_for_test(vec![post], cx));
    });
    draw(cx, window);
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.select_in_post_for_test("a1", 4..15, cx));
        view.update(cx, |v, cx| {
            v.quote_elsewhere(&eidola_gui::actions::QuoteElsewhere, window, cx)
        });
    })
    .unwrap();

    let entries = fresh_entries(cx, window);
    let (_, picker) = entries
        .iter()
        .find(|(n, _)| n == "space/quote-destination")
        .expect("the picker painted");
    assert!(
        picker.bounds.size.height <= gpui::px(400.),
        "59 destinations must not grow the popover past a popover's height: {:?}",
        picker.bounds.size.height
    );
    // Only the visible window materializes. The count is what the frame paid
    // for: 59 candidates, a 220px cap, 22px rows — a dozen or so, never the
    // whole Library.
    let rows = entries
        .iter()
        .filter(|(n, _)| {
            n.starts_with("space/quote-destination/") && n[24..].parse::<u32>().is_ok()
        })
        .count();
    assert!(
        rows > 0 && rows <= 16,
        "the list renders its visible window, not all 59 destinations: {rows}"
    );
    assert!(
        entries
            .iter()
            .any(|(n, _)| n == "space/quote-destination/0"),
        "starting at the top"
    );
    // And the list itself is the `List` landmark, since `uniform_list` cannot
    // carry a role.
    assert_probe(
        &entries,
        "space/quote-destination/list",
        gpui::Role::List,
        "Conversations",
    );
    assert!(
        !entries
            .iter()
            .any(|(n, _)| n == "space/quote-destination/58"),
        "the far end of the index is not materialized at rest"
    );

    // **The arc the single tab stop exists for.** The list holds the keyboard,
    // End moves the cursor to the last destination, and `scroll_to_item`
    // materializes it — so a row no tab order could have contained is now
    // painted, readable, and the one Enter would arm.
    cx.simulate_keystrokes(window, "end");
    let entries = fresh_entries(cx, window);
    let (_, last) = entries
        .iter()
        .find(|(n, _)| n == "space/quote-destination/58")
        .expect("the cursor scrolled the last destination into being");
    assert_eq!(last.role, gpui::Role::ListItem, "a managed descendant");
    assert!(
        !entries
            .iter()
            .any(|(n, _)| n == "space/quote-destination/0"),
        "…and the window moved: the first row is gone from the slice"
    );

    probe::set_probes_enabled(false);
}

/// **The index has three states, and the picker reads all of them** (Codex
/// review, PR #280). `list()` answers `&[]` for a read that failed exactly as
/// it does for a genuinely empty Library, so collapsing them told the reader
/// "no other conversations" about a failure — with nothing to press, and
/// nothing else in a space window that ever re-reads the index.
#[gpui::test]
fn space_quote_destination_reads_the_index_states_apart(cx: &mut TestAppContext) {
    let _guard = probes_on();

    // An index that has not answered: `stub` leaves an empty seed `NotLoaded`.
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        s.eidola_trust = Some(probe_eidola_trust());
    });
    let spaces = stores.spaces.clone();
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, Some("s".into()), WindowInput::new(cx), window, cx))
    });
    let space = view.read_with(cx, |v, _| v.space().clone());
    let mut post = probe_post("a1", "the quick brown fox");
    post.blocks[0].id = "b1".into();
    cx.update(|cx| {
        space.update(cx, |s, cx| s.set_post_tree_for_test(vec![post], cx));
    });
    draw(cx, window);
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.select_in_post_for_test("a1", 4..15, cx));
        view.update(cx, |v, cx| {
            v.quote_elsewhere(&eidola_gui::actions::QuoteElsewhere, window, cx)
        });
    })
    .unwrap();

    let unanswered = fresh_entries(cx, window);
    assert_probe_value(
        &unanswered,
        "space/quote-destination/empty",
        gpui::Role::Label,
        "Conversations",
        "Loading…",
    );
    assert!(
        !unanswered
            .iter()
            .any(|(n, _)| n == "space/quote-destination/retry"),
        "a read in flight is not a failure, so there is nothing to retry yet"
    );

    // A failed *initial* read: say so, and offer the door back — the only one
    // there is, since nothing else in this window re-reads the index.
    cx.update(|cx| {
        spaces.update(cx, |s, cx| {
            s.settle_for_test(None, Err("database is locked"), None, cx)
        });
    });
    let failed = fresh_entries(cx, window);
    assert_probe_value(
        &failed,
        "space/quote-destination/empty",
        gpui::Role::Label,
        "Conversations",
        "Couldn't load your conversations.",
    );
    assert_probe(
        &failed,
        "space/quote-destination/retry",
        gpui::Role::Button,
        "Retry",
    );

    // And a genuinely empty Library keeps the honest sentence, with nothing to
    // retry — a reader with one conversation has nowhere else to quote into.
    cx.update(|cx| {
        spaces.update(cx, |s, cx| {
            s.settle_for_test(None, Ok(Vec::new()), None, cx)
        });
    });
    let empty = fresh_entries(cx, window);
    assert_probe_value(
        &empty,
        "space/quote-destination/empty",
        gpui::Role::Label,
        "Conversations",
        "No other conversations yet.",
    );
    assert!(
        !empty
            .iter()
            .any(|(n, _)| n == "space/quote-destination/retry"),
        "nothing failed, so nothing offers a retry"
    );

    probe::set_probes_enabled(false);
}

/// The cross-space creation UI and the denied-follow notice (task 37): the
/// destination picker's rows, the statement the reader must be shown, its two
/// verbs, and the notice's own dismiss — every one a driver target and an
/// AccessKit node.
#[gpui::test]
fn space_probes_record_the_quote_destination_and_denied_follow(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let stores = stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        s.eidola_trust = Some(probe_eidola_trust());
        s.spaces = vec![
            space_info("s", Some("Here")),
            space_info("other", Some("Tides")),
        ];
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, Some("s".into()), WindowInput::new(cx), window, cx))
    });
    let space = view.read_with(cx, |v, _| v.space().clone());
    let mut post = probe_post("a1", "the quick brown fox");
    post.blocks[0].id = "b1".into();
    cx.update(|cx| {
        space.update(cx, |s, cx| s.set_post_tree_for_test(vec![post], cx));
    });
    draw(cx, window);

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.select_in_post_for_test("a1", 4..15, cx));
        view.update(cx, |v, cx| {
            v.quote_elsewhere(&eidola_gui::actions::QuoteElsewhere, window, cx)
        });
    })
    .unwrap();
    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"space/quote-destination".to_string())
            && names.contains(&"space/quote-destination/0".to_string()),
        "the destination picker and its rows are probed: {names:?}"
    );

    // Arming one grows the statement — carried as the node's **value**, the
    // channel a screen reader reads, because it is the content of the surface
    // rather than its name.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.arm_quote_destination_for_test("other", "Tides", window, cx)
        });
    })
    .unwrap();
    draw(cx, window);
    let entries = probe::window_entries(window.window_id().as_u64());
    let note = entries
        .iter()
        .find(|(n, _)| n == "space/quote-destination/note")
        .expect("the visibility statement is probed");
    assert!(
        note.1
            .value
            .as_ref()
            .is_some_and(|v| v.contains("visible to everyone in Tides")),
        "the statement names the destination: {:?}",
        note.1.value
    );
    let names: Vec<String> = entries.iter().map(|(n, _)| n.to_string()).collect();
    assert!(
        names.contains(&"space/quote-destination/confirm".to_string())
            && names.contains(&"space/quote-destination/cancel".to_string()),
        "both verbs are probed: {names:?}"
    );

    // The denied follow's quiet notice: an Alert carrying its sentence, plus a
    // dismiss.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.close_quote_destination(window, cx);
            v.report_navigation_failure_for_test(
                eidola_app_core::error::AppError::NotAParticipant {
                    participant_id: "p1".into(),
                    action_id: "a-private".into(),
                },
                cx,
            );
        });
    })
    .unwrap();
    draw(cx, window);
    let entries = probe::window_entries(window.window_id().as_u64());
    let notice = entries
        .iter()
        .find(|(n, _)| n == "space/reference-notice")
        .expect("the denial notice is probed");
    assert_eq!(notice.1.role, gpui::Role::Alert);
    let said = notice.1.value.clone().unwrap_or_default();
    assert!(
        said.contains("don't take part in"),
        "the sentence rides as the value: {said}"
    );
    assert!(
        !said.contains("a-private") && !said.contains("p1"),
        "and it names nothing about the refused conversation: {said}"
    );
    let names: Vec<String> = entries.iter().map(|(n, _)| n.to_string()).collect();
    assert!(
        names.contains(&"space/reference-notice/dismiss".to_string()),
        "the notice is dismissible: {names:?}"
    );

    probe::set_probes_enabled(false);
}

/// **The candidate list is bounded and virtualized.** The picker offers every
/// live agent the reader could add, and a seeded space owns an agent — so this
/// list grows with their *conversations*, exactly as the Library does (Codex
/// review, PR #280). The panel around it scrolls, which the virtualized-list
/// doctrine warns about; the warning's mechanism is `Auto` sizing collapsing
/// with no parent height to fill, and this list's height is explicit, so the
/// question is settled here rather than argued: the rows paint, and only the
/// visible ones do.
#[gpui::test]
fn space_inspector_invite_list_is_bounded(cx: &mut TestAppContext) {
    let _guard = probes_on();
    let (window, view) = open_participants_inspector(cx);
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_begin_invite(window, cx));
        view.update(cx, |v, cx| {
            v.seed_invite_candidates_for_test(
                (0..60)
                    .map(|i| eidola_app_core::GrantableAgent {
                        id: format!("agent-{i}"),
                        label: format!("Agent {i}"),
                        shared: true,
                        home_space_title: None,
                    })
                    .collect(),
                cx,
            )
        });
    })
    .unwrap();

    let entries = fresh_entries(cx, window);
    let rows = entries
        .iter()
        .filter(|(n, _)| {
            n.starts_with("space/inspector/participants/invite/")
                && n.rsplit('/')
                    .next()
                    .is_some_and(|t| t.parse::<u32>().is_ok())
        })
        .count();
    assert!(
        rows > 0,
        "the list renders inside the panel's own scroller — it does not collapse: {:?}",
        entries.iter().map(|(n, _)| n).collect::<Vec<_>>()
    );
    assert!(
        rows <= 14,
        "…and only its visible window, never all 60 candidates: {rows}"
    );
    assert_probe(
        &entries,
        "space/inspector/participants/invite/list",
        gpui::Role::List,
        "Agents you can invite",
    );
    assert!(
        !entries
            .iter()
            .any(|(n, _)| n == "space/inspector/participants/invite/59"),
        "the far end is not materialized at rest"
    );

    // The same arc: the list is the tab stop, End moves the cursor to the last
    // candidate, and `scroll_to_item` materializes it.
    let list = view
        .read_with(cx, |v, _| v.invite_list_focus_handle())
        .expect("the form is open");
    cx.update_window(window, |_, window, cx| window.focus(&list, cx))
        .unwrap();
    cx.simulate_keystrokes(window, "end");
    let entries = fresh_entries(cx, window);
    let (_, last) = entries
        .iter()
        .find(|(n, _)| n == "space/inspector/participants/invite/59")
        .expect("the cursor scrolled the last candidate into being");
    assert_eq!(last.role, gpui::Role::ListItem, "a managed descendant");

    // **Reopening starts at the top.** The scroll handle is a view field, so it
    // outlives the form: a reopened list left showing the far end while its
    // fresh cursor sat on candidate 0 would arm someone nobody could see
    // (Codex review, PR #280).
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_cancel_invite(window, cx));
        view.update(cx, |v, cx| v.inspector_begin_invite(window, cx));
        view.update(cx, |v, cx| {
            v.seed_invite_candidates_for_test(
                (0..60)
                    .map(|i| eidola_app_core::GrantableAgent {
                        id: format!("agent-{i}"),
                        label: format!("Agent {i}"),
                        shared: true,
                        home_space_title: None,
                    })
                    .collect(),
                cx,
            )
        });
    })
    .unwrap();
    let entries = fresh_entries(cx, window);
    assert!(
        entries
            .iter()
            .any(|(n, _)| n == "space/inspector/participants/invite/0"),
        "the reopened list shows the candidate its cursor is on"
    );
    assert!(
        !entries
            .iter()
            .any(|(n, _)| n == "space/inspector/participants/invite/59"),
        "…and not where the last one was left"
    );

    probe::set_probes_enabled(false);
}

/// The grant door (task 37): "Invite an agent…" beside Add, and the form it
/// opens — whose exit is probed in every state, including the one where the
/// stub has no backend to list candidates from.
#[gpui::test]
fn space_inspector_invite_probes_its_door_and_its_form(cx: &mut TestAppContext) {
    let _guard = probes_on();
    let (window, view) = open_participants_inspector(cx);

    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"space/inspector/participants/invite".to_string()),
        "the grant door sits with the roster's other verbs: {names:?}"
    );

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_begin_invite(window, cx));
    })
    .unwrap();
    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"space/inspector/participants/invite/form".to_string()),
        "the form is a named group: {names:?}"
    );
    assert!(
        names.contains(&"space/inspector/participants/invite/cancel".to_string()),
        "and its way out is offered even with nothing to list: {names:?}"
    );
    assert!(
        !names.contains(&"space/inspector/participants/invite".to_string()),
        "the door is replaced by the form it opened: {names:?}"
    );

    probe::set_probes_enabled(false);
}

/// **An affordance may not offer what the core refuses.**
///
/// A `brief` — the post an agent writes to open a delegated conversation —
/// renders in the assistant column, because its author *is* an agent and the
/// byline and the column are right. What it is not is a *response*: nothing was
/// inferred, so there is no attempt to repeat, and `regenerate` refuses it
/// (`AppError::WrongPostKind`). The role alone cannot tell the two apart, so
/// the row carries the fact and the gutter reads it.
///
/// Edit is checked from the other side: it is offered on `role == "user"`,
/// which is exactly the `user_input` the core allows, so a brief never had that
/// ghost — asserted here so it stays that way.
#[gpui::test]
fn a_brief_offers_no_regenerate_and_an_answer_still_does(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let stores = ready_stores(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, Some("s".into()), WindowInput::new(cx), window, cx))
    });
    let space = view.read_with(cx, |v, _| v.space().clone());

    let mut brief = probe_post("a1", "Check the tide tables for Friday.");
    brief.action_type = "brief".into();
    brief.participant = PostParticipant {
        kind: "agent".into(),
        label: "Navigator".into(),
    };
    let mut answer = probe_post("a2", "Low water at 06:12 and 18:41.");
    answer.action_type = "inference".into();
    answer.parent_action_id = Some("a1".into());
    answer.participant = PostParticipant {
        kind: "agent".into(),
        label: "Surveyor".into(),
    };
    cx.update(|cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![brief, answer], cx)
        });
    });
    draw(cx, window);

    // One gutter at a time (hover is a single row), so an absent verb is absent
    // rather than merely unrevealed.
    cx.update(|cx| {
        view.update(cx, |v, cx| v.reveal_post_affordances_for_test("a1", cx));
    });
    let names = fresh_names(cx, window);
    assert!(
        !names.contains(&"space/post/0/regenerate".to_string()),
        "a brief was never inferred, so there is nothing to regenerate: {names:?}"
    );
    assert!(
        !names.contains(&"space/post/0/edit".to_string()),
        "and it is not the reader's post to edit either: {names:?}"
    );

    // The agent's actual answer keeps the verb — this is the half that fails if
    // the rule were "no assistant post may be regenerated".
    cx.update(|cx| {
        view.update(cx, |v, cx| v.reveal_post_affordances_for_test("a2", cx));
    });
    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"space/post/1/regenerate".to_string()),
        "an inferred answer is still regenerable: {names:?}"
    );

    probe::set_probes_enabled(false);
}

/// **A window onto a room you only watch offers no verb that would be
/// refused.**
///
/// The human can open any conversation their agents opened between themselves,
/// and what opens is an ordinary window. App-core refuses every acting verb
/// there (`AppError::NotJoined`) and that refusal is the guarantee; this is the
/// window declining to offer what it knows would be refused. Both facts have
/// to be known before it does: the roster has answered and does not carry the
/// human, and the settings have answered that this is not an agent's
/// **notebook** — which the human is also not a member of, and may nonetheless
/// write in.
#[gpui::test]
fn a_room_the_reader_only_watches_offers_no_regenerate(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let agent_only = vec![eidola_app_core::ParticipantInfo {
        id: "agent-a".into(),
        scope: "global".into(),
        source: "referenced".into(),
        kind: "agent".into(),
        label: "Surveyor".into(),
        model_ref: Some("gemma4-31b".into()),
        system_prompt: None,
        notify_policy: "all".into(),
        role: "member".into(),
        reference: None,
    }];
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        s.eidola_trust = Some(probe_eidola_trust());
        // The roster has answered, and the reader is not in it.
        s.participants = Some(("s".to_string(), agent_only));
        // …and this is not a notebook, so the refusal really applies.
        s.space_settings = Some((
            "s".to_string(),
            eidola_app_core::SpaceSettings {
                notebook_participant_id: None,
                ..Default::default()
            },
        ));
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, Some("s".into()), WindowInput::new(cx), window, cx))
    });
    let space = view.read_with(cx, |v, _| v.space().clone());

    let mut answer = probe_post("a1", "Low water at 06:12.");
    answer.action_type = "inference".into();
    answer.participant = PostParticipant {
        kind: "agent".into(),
        label: "Surveyor".into(),
    };
    cx.update(|cx| {
        space.update(cx, |s, cx| s.set_post_tree_for_test(vec![answer], cx));
    });
    draw(cx, window);
    cx.update(|cx| {
        view.update(cx, |v, cx| v.reveal_post_affordances_for_test("a1", cx));
    });

    let names = fresh_names(cx, window);
    assert!(
        !names.contains(&"space/post/0/regenerate".to_string()),
        "an inferred answer is regenerable in general — but not by someone who has not joined \
         the conversation, because the core refuses it: {names:?}"
    );

    // **No door into acting, anywhere on the surface.** The band's `+` opens
    // Reply and Ask together, so it is gone; Ask is the sharp one, because it
    // is the single acting verb app-core cannot refuse on the reader's behalf
    // (it drives `respond_stream_as`, which names the agent it acts as and is
    // the door a turn driver uses) — an Ask chip that renders is a billed
    // inference that will run.
    for probe_name in ["space/band/add", "space/band/menu"] {
        assert!(
            !names.contains(&probe_name.to_string()),
            "{probe_name} opens Reply and Ask, which a watching reader may not do: {names:?}"
        );
    }
    assert!(
        !names
            .iter()
            .any(|n| n.starts_with("space/band/menu/ask/") || n.starts_with("space/cascade/ask/")),
        "and no Ask reaches the surface by any route: {names:?}"
    );

    probe::set_probes_enabled(false);
}

/// The same conversation, read by a **member** — every affordance the test
/// above finds absent is present here, so that one is proving suppression
/// rather than an empty fixture.
#[gpui::test]
fn a_room_the_reader_belongs_to_keeps_its_verbs(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let roster = vec![
        eidola_app_core::ParticipantInfo {
            id: eidola_app_core::HUMAN_PARTICIPANT_ID.into(),
            scope: "global".into(),
            source: "referenced".into(),
            kind: "human".into(),
            label: "User".into(),
            model_ref: None,
            system_prompt: None,
            notify_policy: "explicit".into(),
            role: "owner".into(),
            reference: None,
        },
        eidola_app_core::ParticipantInfo {
            id: "agent-a".into(),
            scope: "global".into(),
            source: "referenced".into(),
            kind: "agent".into(),
            label: "Surveyor".into(),
            model_ref: Some("gemma4-31b".into()),
            system_prompt: None,
            notify_policy: "all".into(),
            role: "member".into(),
            reference: None,
        },
    ];
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        s.eidola_trust = Some(probe_eidola_trust());
        s.participants = Some(("s".to_string(), roster));
        s.space_settings = Some((
            "s".to_string(),
            eidola_app_core::SpaceSettings {
                notebook_participant_id: None,
                ..Default::default()
            },
        ));
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, Some("s".into()), WindowInput::new(cx), window, cx))
    });
    let space = view.read_with(cx, |v, _| v.space().clone());

    let mut answer = probe_post("a1", "Low water at 06:12.");
    answer.action_type = "inference".into();
    answer.participant = PostParticipant {
        kind: "agent".into(),
        label: "Surveyor".into(),
    };
    cx.update(|cx| {
        space.update(cx, |s, cx| s.set_post_tree_for_test(vec![answer], cx));
    });
    draw(cx, window);
    cx.update(|cx| {
        view.update(cx, |v, cx| v.reveal_post_affordances_for_test("a1", cx));
    });

    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"space/post/0/regenerate".to_string()),
        "a member may regenerate an inferred answer: {names:?}"
    );
    assert!(
        names.contains(&"space/band/add".to_string()),
        "and reach Reply and Ask: {names:?}"
    );

    probe::set_probes_enabled(false);
}

/// **The gate has to converge, and its interim must not be the open one.**
///
/// The verdict reads two cells that land asynchronously, and the roster's
/// arrival is what asks for the second. Two failures live there, and this
/// walks the ordering that finds both: the observers must *re-drive* the
/// request rather than merely repaint (otherwise the notebook cell is never
/// asked for, and the gate sits on "unknown" for the life of the window), and
/// "unknown" must not mean "may act" once the roster has answered without the
/// reader in it — that window is exactly where an Ask nobody downstream will
/// refuse would be a billed inference.
///
/// The cost is a notebook's verbs for as long as one cell takes to answer.
/// That they come back is asserted here rather than assumed.
#[gpui::test]
fn the_acting_gate_converges_and_its_interim_withholds(cx: &mut TestAppContext) {
    let _guard = probes_on();

    // Nothing seeded: neither cell has answered.
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        s.eidola_trust = Some(probe_eidola_trust());
    });
    let participants = stores.participants.clone();
    let settings = stores.space_settings.clone();
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, Some("s".into()), WindowInput::new(cx), window, cx))
    });
    let space = view.read_with(cx, |v, _| v.space().clone());

    let mut answer = probe_post("a1", "Low water at 06:12.");
    answer.action_type = "inference".into();
    answer.model = Some("gemma4-31b".into());
    answer.participant = PostParticipant {
        kind: "agent".into(),
        label: "Surveyor".into(),
    };
    cx.update(|cx| {
        space.update(cx, |s, cx| s.set_post_tree_for_test(vec![answer], cx));
    });
    draw(cx, window);
    cx.update(|cx| {
        view.update(cx, |v, cx| v.reveal_post_affordances_for_test("a1", cx));
    });

    // **Roster unanswered: the verbs are there.** An ordinary conversation —
    // where the reader is always a member — must not flicker on the way in,
    // and until the roster speaks there is nothing to suspect.
    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"space/post/0/regenerate".to_string()),
        "an unanswered roster must not cost an ordinary room its verbs: {names:?}"
    );

    // **The roster answers without the reader.** This is the whole window the
    // second cell exists to close, and until it answers the verbs go.
    cx.update(|cx| {
        participants.update(cx, |p, cx| {
            p.seed_for_test(
                "s",
                vec![eidola_app_core::ParticipantInfo {
                    id: "agent-a".into(),
                    scope: "global".into(),
                    source: "referenced".into(),
                    kind: "agent".into(),
                    label: "Surveyor".into(),
                    model_ref: Some("gemma4-31b".into()),
                    system_prompt: None,
                    notify_policy: "all".into(),
                    role: "member".into(),
                    reference: None,
                }],
            );
            cx.notify();
        });
    });
    let names = fresh_names(cx, window);
    assert!(
        !names.contains(&"space/post/0/regenerate".to_string()),
        "provably not a member, and nothing yet says this is a notebook: {names:?}"
    );
    assert!(
        !names.contains(&"space/band/add".to_string()),
        "and no door into Reply or the billed Ask: {names:?}"
    );

    // **The notebook cell answers, and the verbs come back.** A notebook
    // legitimately lacks the human and legitimately allows writing, so the
    // suppression above has to be an interim rather than a verdict.
    cx.update(|cx| {
        settings.update(cx, |s, cx| {
            s.seed_for_test(
                "s",
                eidola_app_core::SpaceSettings {
                    notebook_participant_id: Some("agent-a".into()),
                    ..Default::default()
                },
            );
            cx.notify();
        });
    });
    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"space/post/0/regenerate".to_string()),
        "a notebook's reader gets its verbs back once the cell answers: {names:?}"
    );
    assert!(
        names.contains(&"space/band/add".to_string()),
        "including the door into Reply and Ask: {names:?}"
    );

    probe::set_probes_enabled(false);
}

/// **Chrome belongs to a post that made a request.**
///
/// A `brief` renders in the assistant column because its author is an agent,
/// but nothing was requested for it: no model, no backend, no spend. Reading
/// its byline as a model reference named a serving backend that never served
/// it — and `parse_model_ref` takes any label, so the author's own name became
/// one. The accessible label folds the same pair, so both follow one fix.
#[gpui::test]
fn a_brief_wears_no_backend_and_an_answer_still_does(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let stores = ready_stores(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, Some("s".into()), WindowInput::new(cx), window, cx))
    });
    let space = view.read_with(cx, |v, _| v.space().clone());

    let mut brief = probe_post("a1", "Check the tide tables for Friday.");
    brief.action_type = "brief".into();
    brief.model = None;
    brief.participant = PostParticipant {
        kind: "agent".into(),
        label: "Navigator".into(),
    };
    let mut answer = probe_post("a2", "Low water at 06:12.");
    answer.action_type = "inference".into();
    answer.parent_action_id = Some("a1".into());
    answer.model = Some("gemma4-31b".into());
    answer.participant = PostParticipant {
        kind: "agent".into(),
        label: "Surveyor".into(),
    };
    cx.update(|cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![brief, answer], cx)
        });
    });
    draw(cx, window);

    let entries = fresh_entries(cx, window);
    let names: Vec<String> = entries.iter().map(|(n, _)| n.clone()).collect();
    assert!(
        !names.contains(&"space/post/a1/metadata/backend".to_string()),
        "a brief made no request, so it wears no serving backend: {names:?}"
    );
    assert!(
        names.contains(&"space/post/a2/metadata/backend".to_string()),
        "an inference keeps its chrome: {names:?}"
    );

    // The accessible label folds the same byline pair, so it is where the
    // claim was actually spoken: "Navigator · Eidola" for a post that reached
    // no backend. It now names the author and stops.
    let label_of = |name: &str| {
        entries
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("{name} is probed: {entries:?}"))
            .1
            .label
            .clone()
    };
    assert_eq!(
        label_of("space/post/0"),
        "Navigator",
        "the brief's spoken label is its author, not a model name parsed out of it"
    );
    assert!(
        label_of("space/post/1").contains("Eidola"),
        "an inference still says where it was served: {:?}",
        label_of("space/post/1")
    );

    probe::set_probes_enabled(false);
}

/// **The gate's loading is re-driven by the cells it waits on.**
///
/// `viewer_may_act` reads two cells; `ensure_viewer_gate` asks for the second
/// only once the first has answered without the reader in it. Nothing else in
/// this window rebuilds on either store, so if their observers merely repaint,
/// that request is never made and the gate waits forever — which, before the
/// interim was made to fail closed, meant the billed Ask stayed exposed for the
/// life of the window.
///
/// Asserted lexically because it cannot be asserted otherwise here: a stubbed
/// store has no `AppCore`, so its `refresh` completes nothing and the request
/// leaves no trace to observe. What is checkable is that the trigger is wired,
/// and that is the half that was missing.
#[test]
fn the_acting_gates_cells_re_drive_its_loading() {
    let source = include_str!("../src/space_view/mod.rs");
    for store in ["stores.participants", "stores.space_settings"] {
        let at = source
            .find(&format!("cx.observe(&{store},"))
            .unwrap_or_else(|| panic!("{store} is observed by SpaceView"));
        // The observer's own body — up to the next observer registration.
        let body_end = source[at..]
            .find("cx.observe")
            .map(|i| at + i)
            .and_then(|start| {
                source[start + 10..]
                    .find("cx.observe")
                    .map(|i| start + 10 + i)
            })
            .unwrap_or(source.len());
        assert!(
            source[at..body_end].contains("ensure_viewer_gate"),
            "{store}'s observer must re-drive ensure_viewer_gate, not merely repaint: its \
             arrival is what asks for the acting gate's other cell, and nothing else here \
             rebuilds"
        );
    }
}
