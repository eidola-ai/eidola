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
use eidola_gui::actions::ToggleModelPicker;
use eidola_gui::general::GeneralView;
use eidola_gui::library::LibraryView;
use eidola_gui::onboarding::{OnboardingView, Slide};
use eidola_gui::probe;
use eidola_gui::record::{RecordSection, RecordView};
use eidola_gui::space_view::SpaceView;
use eidola_gui::stores::{Stores, StoresStub};
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
        default_model: "gemma4-31b".into(),
        has_account: true,
        has_account_secret: true,
        domain_separator: "ACT-v1:eidola:inference:production:2026-03-05".into(),
        attestation_url: None,
        appearance: eidola_app_core::config::AppearanceSetting::System,
        time_of_day_tint: eidola_app_core::config::TimeOfDayTint::On,
        light_character: eidola_app_core::config::LightCharacter::Neutral,
    }
}

fn probe_eidola_trust() -> eidola_app_core::EidolaTrust {
    eidola_app_core::EidolaTrust {
        base_url: "https://eidola.example/v1".into(),
        base_url_pin: "https://eidola.example/v1".into(),
        base_url_is_override: false,
        trusted_measurements: Vec::new(),
        trusted_measurements_are_override: false,
        has_hardware_root_ca: false,
        has_hardware_intermediate_ca: false,
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
        (window.into(), inner.expect("build closure produced a view"))
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
    // Give the active draft content and open the request panel, so the action
    // gutter (Ask / model chip) and the panel record their probes too.
    let editor = view
        .read_with(cx, |v, _| v.composer_state_for_test())
        .expect("blank space opens with the composer");
    cx.update(|cx| {
        editor.update(cx, |e, cx| e.set_value("a draft".to_string(), cx));
        view.update(cx, |v, cx| v.toggle_request_panel(cx));
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
    // The action gutter: Ask (the discoverable submit) and the model chip.
    assert!(
        names.contains(&"space/composer/ask"),
        "Ask affordance probe missing; recorded: {names:?}"
    );
    assert!(
        names.contains(&"space/composer/model"),
        "model chip probe missing; recorded: {names:?}"
    );
    // The request panel (opened above) with its model rows.
    assert!(
        names.contains(&"space/request-panel"),
        "request panel probe missing; recorded: {names:?}"
    );
    assert!(
        names
            .iter()
            .any(|n| n.starts_with("space/request-panel/remote/eidola/")),
        "request panel model rows missing; recorded: {names:?}"
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
fn general_pane_probes_cover_appearance_and_advanced(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let stores = ready_stores(cx);
    let window_input = cx.update(WindowInput::new);
    let wi = window_input.clone();
    let (window, _view) = open_view(cx, move |window, cx| {
        cx.new(|cx| GeneralView::new(stores.config.clone(), wi, window, cx))
    });

    // At rest: the appearance chips are probed. (The base-URL editor moved to
    // Backends → Eidola; General no longer carries a "Change…" affordance.)
    let names = fresh_names(cx, window);
    for expected in [
        "settings/general/appearance/system",
        "settings/general/time-of-day/on",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "general probe {expected:?} missing; recorded: {names:?}"
        );
    }
    assert!(
        !names.contains(&"settings/general/change".to_string()),
        "the base-URL editor must no longer live in General: {names:?}"
    );

    // Holding ⌥ reveals the advanced rows, including the Record cross-link.
    cx.update(|cx| {
        window_input.update(cx, |wi, cx| wi.set_alt_for_test(true, cx));
    });
    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"settings/general/open-record".to_string()),
        "advanced open-record link probe missing under ⌥; recorded: {names:?}"
    );

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn request_panel_probes_appear_on_open_and_clear_on_dismiss(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let stores = ready_stores(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, None, WindowInput::new(cx), window, cx))
    });

    // Closed panel: no listbox probes.
    draw(cx, window);
    let names = fresh_names(cx, window);
    assert!(
        !names.iter().any(|n| n.starts_with("space/request-panel")),
        "panel probes before opening: {names:?}"
    );

    // Open via the real action dispatch path (the blank space's root draft
    // anchors the panel).
    let focus = view.read_with(cx, |v, _| v.focus_handle());
    cx.update_window(window, |_, window, cx| {
        focus.dispatch_action(&ToggleModelPicker, window, cx);
    })
    .unwrap();
    cx.run_until_parked();

    let names = fresh_names(cx, window);
    assert!(
        names.contains(&"space/request-panel".to_string()),
        "panel probe missing after open: {names:?}"
    );
    assert!(
        names.contains(&"space/request-panel/remote/eidola/0".to_string())
            && names.contains(&"space/request-panel/remote/eidola/2".to_string()),
        "per-model row probes missing: {names:?}"
    );

    // Dismiss: the clear-then-redraw dance must drop the unmounted panel —
    // stale entries would be ghost click targets for the driver.
    cx.update_window(window, |_, window, cx| {
        focus.dispatch_action(&ToggleModelPicker, window, cx);
    })
    .unwrap();
    cx.run_until_parked();

    let names = fresh_names(cx, window);
    assert!(
        !names.iter().any(|n| n.starts_with("space/request-panel")),
        "panel probes must clear after dismiss: {names:?}"
    );
    assert!(
        names.contains(&"space/composer".to_string()),
        "still-mounted probes must survive the refresh: {names:?}"
    );
}

/// Linux window chrome: `ChromeRoot` wraps every production window and hosts
/// the primary menu (the macOS app/File-menu replacement). The wordmark
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
        cx.new(|cx| BackendsSettingsView::new(stores, WindowInput::new(cx), window, cx))
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
    // Root CA overridden so the Clear verb renders; intermediate left at the
    // pin (only Set shows there).
    trust.has_hardware_root_ca = true;

    let stores = stub_stores(cx, |s| {
        s.config_state = Some(probe_config_state());
        s.eidola_trust = Some(trust);
        s.backends = backends_fixture();
    });
    let window_input = cx.update(WindowInput::new);
    let wi = window_input.clone();
    let (window, view) = open_view(cx, move |window, cx| {
        cx.new(|cx| BackendsSettingsView::new(stores, wi, window, cx))
    });

    // At rest (⌥ not held): the override warning band, the base-URL Change
    // affordance, and both revert-to-pin verbs. The measurement/CA *editors*
    // are gated behind ⌥, so they must be absent here.
    let names = fresh_names(cx, window);
    for expected in [
        "settings/backends/eidola/trust-warning",
        "settings/backends/eidola/url/change",
        "settings/backends/eidola/url/revert-to-pin",
        "settings/backends/eidola/measurements/revert",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "eidola trust-surface probe {expected:?} missing: {names:?}"
        );
    }
    for gated in [
        "settings/backends/eidola/measurements/0/untrust",
        "settings/backends/eidola/measurements/add",
        "settings/backends/eidola/measurements/add/submit",
        "settings/backends/eidola/ca/root/set",
        "settings/backends/eidola/ca/intermediate/set",
    ] {
        assert!(
            !names.contains(&gated.to_string()),
            "trust editor probe {gated:?} must be hidden without ⌥: {names:?}"
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

    // Holding ⌥ reveals the measurement + CA editors.
    cx.update(|cx| {
        window_input.update(cx, |wi, cx| wi.set_alt_for_test(true, cx));
    });
    let names = fresh_names(cx, window);
    for expected in [
        "settings/backends/eidola/measurements/0/untrust",
        "settings/backends/eidola/measurements/add",
        "settings/backends/eidola/measurements/add/submit",
        "settings/backends/eidola/ca/root/input",
        "settings/backends/eidola/ca/root/set",
        "settings/backends/eidola/ca/root/clear",
        "settings/backends/eidola/ca/intermediate/input",
        "settings/backends/eidola/ca/intermediate/set",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "eidola ⌥-editor probe {expected:?} missing: {names:?}"
        );
    }

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

#[gpui::test]
fn backends_pane_add_form_probes_appear_per_kind(cx: &mut TestAppContext) {
    use eidola_gui::backends_settings::{AddKind, BackendsSettingsView, BackendsTab};

    let _guard = probes_on();

    let stores = stub_stores(cx, |s| {
        s.backends = backends_fixture();
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| BackendsSettingsView::new(stores, WindowInput::new(cx), window, cx))
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

#[gpui::test]
fn request_panel_lists_all_on_disk_local_models_first(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let stores = stub_stores(cx, |s| {
        s.config_state = None;
        s.models = vec![ModelInfo {
            id: "gemma4-31b".into(),
            context_length: 131_072,
            prompt_credits_per_token: 0.53,
            completion_credits_per_token: 1.5,
            request_credits: None,
        }];
        s.local_models = Some(local_models_fixture());
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, None, WindowInput::new(cx), window, cx))
    });
    draw(cx, window);

    let focus = view.read_with(cx, |v, _| v.focus_handle());
    cx.update_window(window, |_, window, cx| {
        focus.dispatch_action(&ToggleModelPicker, window, cx);
    })
    .unwrap();
    cx.run_until_parked();

    let names = fresh_names(cx, window);
    // *Every* on-disk local model appears ahead of the remote rows — the
    // merely-downloaded Tiny A included (a request loads it on demand).
    assert!(
        names.contains(&"space/request-panel/engine/local/0".to_string()),
        "downloaded (unloaded) local model row missing: {names:?}"
    );
    assert!(
        names.contains(&"space/request-panel/engine/local/1".to_string()),
        "loaded local model row missing: {names:?}"
    );
    assert!(
        names.contains(&"space/request-panel/remote/eidola/0".to_string()),
        "remote model rows must still render below the local group: {names:?}"
    );

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn request_panel_offline_shows_local_models_and_retry(cx: &mut TestAppContext) {
    let _guard = probes_on();

    // Offline: no remote models, and the remote store forced into a failed
    // state (the app-launch fetch couldn't reach the upstream).
    let stores = stub_stores(cx, |s| {
        s.config_state = None;
        s.local_models = Some(local_models_fixture());
    });
    cx.update(|cx| {
        stores
            .models
            .update(cx, |s, cx| s.set_failed_for_test("offline", cx));
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, None, WindowInput::new(cx), window, cx))
    });
    draw(cx, window);

    let focus = view.read_with(cx, |v, _| v.focus_handle());
    cx.update_window(window, |_, window, cx| {
        focus.dispatch_action(&ToggleModelPicker, window, cx);
    })
    .unwrap();
    cx.run_until_parked();

    let names = fresh_names(cx, window);
    // The loaded local model stays selectable even though the remote fetch
    // failed — the panel is never a dead end.
    assert!(
        names.contains(&"space/request-panel/engine/local/0".to_string()),
        "loaded local model must show while offline: {names:?}"
    );
    // ...and the remote list offers a retry, not silent nothing.
    assert!(
        names.contains(&"space/request-panel/eidola/retry".to_string()),
        "per-backend retry affordance missing on a failed eidola fetch: {names:?}"
    );
    assert!(
        !names.contains(&"space/request-panel/eidola/refresh".to_string()),
        "a failed fetch shows retry, not refresh: {names:?}"
    );

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn request_panel_shows_refresh_when_remote_loaded(cx: &mut TestAppContext) {
    let _guard = probes_on();

    let stores = stub_stores(cx, |s| {
        s.config_state = None;
        s.models = vec![ModelInfo {
            id: "gemma4-31b".into(),
            context_length: 131_072,
            prompt_credits_per_token: 0.53,
            completion_credits_per_token: 1.5,
            request_credits: None,
        }];
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, None, WindowInput::new(cx), window, cx))
    });
    draw(cx, window);

    let focus = view.read_with(cx, |v, _| v.focus_handle());
    cx.update_window(window, |_, window, cx| {
        focus.dispatch_action(&ToggleModelPicker, window, cx);
    })
    .unwrap();
    cx.run_until_parked();

    let names = fresh_names(cx, window);
    // A refresh is offered even over a *successful* remote list.
    assert!(
        names.contains(&"space/request-panel/eidola/refresh".to_string()),
        "per-backend refresh affordance must be available over a good list: {names:?}"
    );
    assert!(
        !names.contains(&"space/request-panel/eidola/retry".to_string()),
        "a good fetch shows refresh, not retry: {names:?}"
    );

    probe::set_probes_enabled(false);
}

#[gpui::test]
fn request_panel_groups_per_backend_with_independent_health(cx: &mut TestAppContext) {
    use eidola_gui::loadable::Loadable;
    use eidola_gui::stores::BackendCatalog;

    let _guard = probes_on();

    // A full multi-backend scene: the local singleton with a loaded engine,
    // a llamacpp backend with a loaded engine, a healthy eidola catalog,
    // and an openai backend whose fetch failed.
    let mut external_fixture = local_models_fixture();
    if let Some(m) = external_fixture
        .external
        .get_mut(0)
        .and_then(|b| b.models.get_mut(0))
    {
        m.status = eidola_app_core::LocalModelStatus::Loaded {
            port: 4243,
            context_tokens: 8192,
            pinned: false,
        };
    }
    let stores = stub_stores(cx, |s| {
        s.config_state = None;
        s.backends = backends_fixture();
        s.local_models = Some(external_fixture);
        s.backend_catalogs = Some(vec![
            BackendCatalog {
                backend: eidola_app_core::BackendInfo {
                    id: "eidola".into(),
                    kind: eidola_app_core::BackendKind::Eidola,
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
                models: Loadable::loaded(vec![ModelInfo {
                    id: "gemma4-31b".into(),
                    context_length: 131_072,
                    prompt_credits_per_token: 0.53,
                    completion_credits_per_token: 1.5,
                    request_credits: None,
                }]),
            },
            BackendCatalog {
                backend: eidola_app_core::BackendInfo {
                    id: "my-vllm".into(),
                    kind: eidola_app_core::BackendKind::OpenAi,
                    display_name: "My vLLM".into(),
                    enabled: true,
                    base_url: Some("http://10.0.0.2:8000".into()),
                    has_api_key: true,
                    models_dir: None,
                    model_overrides: None,
                    engine_path: None,
                    auto_start: true,
                    created_at: 1,
                },
                models: Loadable::Failed {
                    error: eidola_app_core::error::AppError::Internal {
                        message: "connection refused".into(),
                    },
                    prior: None,
                },
            },
        ]);
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, None, WindowInput::new(cx), window, cx))
    });
    draw(cx, window);

    let focus = view.read_with(cx, |v, _| v.focus_handle());
    cx.update_window(window, |_, window, cx| {
        focus.dispatch_action(&ToggleModelPicker, window, cx);
    })
    .unwrap();
    cx.run_until_parked();

    let names = fresh_names(cx, window);
    for expected in [
        // Engine groups: the managed store and the llamacpp backend.
        "space/request-panel/engine/local/0",
        "space/request-panel/engine/my-box/0",
        // The healthy eidola catalog: rows + refresh, no retry.
        "space/request-panel/remote/eidola/0",
        "space/request-panel/eidola/refresh",
        // The dead openai backend: its own retry — nobody else's health.
        "space/request-panel/my-vllm/retry",
    ] {
        assert!(
            names.contains(&expected.to_string()),
            "multi-backend panel probe {expected:?} missing: {names:?}"
        );
    }
    assert!(
        !names.contains(&"space/request-panel/eidola/retry".to_string()),
        "one backend's failure must not mark another's group: {names:?}"
    );

    probe::set_probes_enabled(false);
}
