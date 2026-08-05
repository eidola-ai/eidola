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

use eidola_app_core::{
    AttestationInfo, BalancesResult, ConfigState, ModelInfo, PostBlock, PostNode, PostParticipant,
    PriceInfo, RequestInfo, SpendTrailEntry,
};
use eidola_app_core::{
    ParticipantInfo, ParticipantReference, SpaceTemplateInfo, TemplateParticipantInfo,
};
use eidola_gui::general::GeneralView;
use eidola_gui::library::LibraryView;
use eidola_gui::onboarding::{OnboardingView, Slide};
use eidola_gui::participants_view::{EditMode, ParticipantsView};
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

    let stores = stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        s.participants = Some(probe_participants());
    });
    let (window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| ParticipantsView::new(stores, "demo".into(), None, window, cx))
    });

    // Repeated "Edit"/"Remove" with nothing to distinguish them is exactly the
    // audit's context-free-label finding; the row's subject supplies it.
    let entries = fresh_entries(cx, window);
    assert_probe(
        &entries,
        "participants/agent-1/edit",
        gpui::Role::Button,
        "Edit Assistant",
    );
    assert_probe(
        &entries,
        "participants/agent-1/remove",
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
        domain_separator: "ACT-v1:eidola:inference:production:2026-03-05".into(),
        attestation_url: None,
        appearance: eidola_app_core::config::AppearanceSetting::System,
        time_of_day_tint: eidola_app_core::config::TimeOfDayTint::On,
        light_character: eidola_app_core::config::LightCharacter::Neutral,
        font_scale: 1.0,
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
                eidola_app_core::error::AppError::ChatFailed {
                    space_id: "s".into(),
                    source: Box::new(eidola_app_core::error::AppError::Network {
                        message: "dns error".into(),
                    }),
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
        label: "You".into(),
        model_ref: None,
        system_prompt: None,
        notify_policy: "explicit".into(),
        role: "member".into(),
        reference: Some(ParticipantReference {
            base_label: "You".into(),
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

#[gpui::test]
fn participants_view_probes_cover_rows_editor_and_add(cx: &mut TestAppContext) {
    let _guard = probes_on();
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        s.participants = Some(probe_participants());
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| {
            ParticipantsView::new(
                stores.clone(),
                "demo".into(),
                Some("Demo".into()),
                window,
                cx,
            )
        })
    });

    // The resting rows: You is not removable; the agent is; plus the two links.
    let names = fresh_names(cx, window);
    let you = eidola_app_core::HUMAN_PARTICIPANT_ID;
    for expected in [
        format!("participants/{you}/edit"),
        "participants/agent-1/edit".to_string(),
        "participants/agent-1/remove".to_string(),
        "participants/add".to_string(),
        "participants/save-template".to_string(),
    ] {
        assert!(
            names.contains(&expected),
            "row probe {expected:?} missing: {names:?}"
        );
    }
    assert!(
        !names.contains(&format!("participants/{you}/remove")),
        "the shared human must not be removable: {names:?}"
    );

    // Editing You surfaces the edit-everywhere-vs-override-here fork.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.begin_edit(you, window, cx));
    })
    .unwrap();
    let names = fresh_names(cx, window);
    for expected in [
        "participants/editor/mode/everywhere",
        "participants/editor/mode/override",
        "participants/editor/label",
        "participants/editor/cancel",
        "participants/editor/save",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "editor probe {expected:?} missing: {names:?}"
        );
    }
    // The human editor shows only the mode toggle + name (no model/prompt).
    assert!(!names.contains(&"participants/editor/model".to_string()));

    // Editing an agent adds the model field + notify chips.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.set_edit_mode(EditMode::Everywhere, window, cx)
        });
        view.update(cx, |v, cx| v.begin_edit("agent-1", window, cx));
    })
    .unwrap();
    let names = fresh_names(cx, window);
    for expected in [
        "participants/editor/model",
        "participants/editor/system-prompt",
        "participants/editor/notify/human",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "agent editor probe {expected:?} missing: {names:?}"
        );
    }

    // The add form.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.begin_add(window, cx));
    })
    .unwrap();
    let names = fresh_names(cx, window);
    for expected in [
        "participants/add/name",
        "participants/add/model",
        "participants/add/system-prompt",
        "participants/add/notify/human",
        "participants/add/submit",
        "participants/add/cancel",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "add-form probe {expected:?} missing: {names:?}"
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
                label: "You".into(),
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
    let (p_window, p_view) = open_view(cx, |window, cx| {
        cx.new(|cx| {
            ParticipantsView::new(
                stores.clone(),
                "demo".into(),
                Some("Demo".into()),
                window,
                cx,
            )
        })
    });
    cx.update_window(p_window, |_, window, cx| {
        p_view.update(cx, |v, cx| v.begin_add(window, cx));
    })
    .unwrap();
    p_view.update(cx, |v, cx| v.open_add_picker_for_test(cx));
    let entries = fresh_entries(cx, p_window);
    assert_probe(
        &entries,
        "participants/add/model/option/0/0",
        gpui::Role::Button,
        "Gemma 4 E2B · Local",
    );
    assert_probe(
        &entries,
        "participants/add/model/option/1/0",
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
    let (p_window, p_view) = open_view(cx, |window, cx| {
        cx.new(|cx| {
            ParticipantsView::new(
                stores.clone(),
                "demo".into(),
                Some("Demo".into()),
                window,
                cx,
            )
        })
    });
    cx.update_window(p_window, |_, window, cx| {
        p_view.update(cx, |v, cx| v.begin_edit("agent-1", window, cx));
    })
    .unwrap();
    let entries = fresh_entries(cx, p_window);
    assert_probe_value(
        &entries,
        "participants/editor/model",
        gpui::Role::Button,
        "Model",
        "gemma4-31b · Eidola",
    );

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn participants_view_failed_load_shows_retry_not_controls(cx: &mut TestAppContext) {
    let _guard = probes_on();
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| {
            ParticipantsView::new(
                stores.clone(),
                "demo".into(),
                Some("Demo".into()),
                window,
                cx,
            )
        })
    });
    // A failed *initial* load: no prior data, no live roster.
    stores
        .participants
        .update(cx, |s, _| s.set_failed_for_test("demo", "boom"));
    let _ = view;

    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"participants/retry".to_string()),
        "failed load must offer Retry: {names:?}"
    );
    for absent in ["participants/add", "participants/save-template"] {
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
        },
        eidola_app_core::PostReference {
            antecedent_action_id: "x2".into(),
            ordinal: 2,
            content_block_id: Some("by".into()),
            range_start: Some(0),
            range_end: Some(4),
            annotation: None,
            snippet: None, // the honest "quoted an earlier version" row
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
