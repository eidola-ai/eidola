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

use eidola_app_core::AppCore;
use eidola_app_core::changes::{Change, ChangeOrigin};
use eidola_app_core::error::AppError;
use eidola_app_core::updates::{
    Claim, ClaimDelta, ClaimsComparison, UpdateCheckResult, UpdateCheckSnapshot, VerifiedRelease,
};
use eidola_app_core::{
    AttestationDetail, AttestationInfo, BalancesResult, ConfigState, CredentialLifecycleInfo,
    PostBlock, PostNode, PostParticipant, PostTrace, RequestDetail, RequestInfo, SpaceInfo,
    SpaceMessage, TraceEntry,
};
use eidola_gui::about::AboutView;
use eidola_gui::account::AccountView;
use eidola_gui::actions::{PostOnly, Send};
use eidola_gui::agents_settings::AgentsSettingsView;
use eidola_gui::library::LibraryView;
use eidola_gui::onboarding::{OnboardingView, Slide};
use eidola_gui::participants::EditMode;
use eidola_gui::record::{RecordDetail, RecordSection, RecordView};
use eidola_gui::settings::{SettingsPane, SettingsView};
use eidola_gui::space_view::SpaceView;
use eidola_gui::space_view::model::streaming_node_id;
use eidola_gui::stores::{self, Stores, StoresStub};
use eidola_gui::templates_settings::TemplatesSettingsView;
use eidola_gui::updates::{UpdatesDisplay, UpdatesView, relative_time};
use eidola_gui::wallet::WalletView;
use eidola_gui::window_input::WindowInput;
use gpui::{
    AnyWindowHandle, AppContext, Entity, Focusable, Modifiers, Point, TestAppContext,
    VisualTestContext, WindowOptions, px,
};
use gpui_component::{Root, Theme};

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
// Post fixtures (shared by the space-view tests)
// ---------------------------------------------------------------------------

/// A minimal fixture post — a user post with a real `action_id`, so the
/// per-post affordances apply.
fn fixture_user_post(action_id: &str, text: &str) -> PostNode {
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

// The persistent participant-indicator system was retired in the wave-5.3
// post redesign: post identity now lives in each post's own gutter byline
// (rendered by `post_frame`), not a title-bar band cue. Its two derivation
// tests were removed with it. Byline content is covered by the post-tree DTO
// tests (app-core) and the visual snapshots.

#[gpui::test]
fn a_new_space_is_registered_under_its_id_from_the_first_frame(cx: &mut TestAppContext) {
    use eidola_gui::space::Space;

    // ⌘N creates the space: the id is minted client-side and the entity joins
    // the registry in the same breath, so the window is addressed by a real id
    // before the row's insert has committed. So a later `open` of that id
    // shares the SAME entity, with no moment in between where two entities
    // could describe one conversation.
    let stores = stub_stores_with_config(cx);

    let fresh: Entity<Space> = stores.spaces.update(cx, |store, cx| store.create(cx));
    cx.run_until_parked();
    let id = fresh.read_with(cx, |s, _| s.id().to_string());
    assert!(!id.is_empty(), "a new space has its id from birth");

    // The page is answered-and-empty at once: no transcript read stands
    // between ⌘N and a composer (STATE.md's blank-page-stays-instant rule).
    fresh.read_with(cx, |s, _| {
        assert!(s.transcript_visible(), "the blank page answers immediately");
        assert!(s.messages().is_empty());
    });

    let reopened = stores.spaces.update(cx, |store, cx| store.open(id, cx));
    assert_eq!(
        reopened.entity_id(),
        fresh.entity_id(),
        "open(id) must join the registered entity — no fork"
    );
}

/// A refused insert is a window-level error state, not a silent blank page.
///
/// ⌘N opens the window before the row commits, so the one thing the reader
/// must never see is an ordinary empty notebook standing on a space that does
/// not exist: a composer there would write nowhere. The failure lands in the
/// transcript cell — the same door a failed *read* lands in — so the tail
/// composer is never minted, and the window's own error band says what
/// happened. And no phantom registry entry survives it: a later `open` of that
/// id cannot join an entity standing on an error.
#[gpui::test]
fn a_refused_space_creation_says_so_instead_of_showing_a_blank_page(cx: &mut TestAppContext) {
    let stores = stub_stores_with_config(cx);
    let (_window, view) = open_space(cx, &stores, None);
    cx.run_until_parked();

    let space = view.read_with(cx, |v, _| v.space().clone());
    let space_id = space.read_with(cx, |s, _| s.id().to_string());
    space.read_with(cx, |s, _| {
        assert!(
            s.transcript_visible(),
            "the page answers immediately while the insert runs"
        );
    });

    stores.spaces.update(cx, |s, cx| {
        s.fail_creation_for_test(
            &space_id,
            AppError::Database {
                message: "disk is full".into(),
            },
            cx,
        )
    });
    cx.run_until_parked();

    space.read_with(cx, |s, _| {
        assert!(
            !s.transcript_visible(),
            "a space that was never created has no tree to compose into"
        );
    });
    let shown = view
        .read_with(cx, |v, cx| v.error_for_test(cx))
        .expect("the window says the space could not be created rather than showing a blank page");
    assert!(shown.contains("disk is full"), "got {shown}");

    let reopened = stores
        .spaces
        .update(cx, |s, cx| s.open(space_id.clone(), cx));
    assert_ne!(
        reopened.entity_id(),
        space.entity_id(),
        "the registry must not keep pointing at a conversation that does not exist"
    );
    assert!(
        stores
            .spaces
            .read_with(cx, |s, _| s.op_error_for(&space_id).map(str::to_string))
            .is_some(),
        "and the refusal stands for the Library banner too"
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
        s.eidola_trust = Some(eidola_trust());
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
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.cancel_rename(window, cx));
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert_eq!(v.renaming_space_id(), None);
    });
}

#[gpui::test]
fn library_hover_survives_out_of_order_leave(cx: &mut TestAppContext) {
    // Hover-event ordering: moving the cursor *up* the list, gpui can fire the
    // left row's `on_hover(false)` AFTER the entered row's `on_hover(true)`. The
    // clear must be conditional on still being the hovered row, or the new row's
    // hover (and its reveal buttons) gets wiped. Replay the down-the-list order:
    // row B becomes hovered, then row A's stale leave fires — hover must stay B.
    let stores = stub_stores(cx, |s| {
        s.spaces = vec![
            stub_space("s0", Some("Row A"), None, 1_000),
            stub_space("s1", Some("Row B"), None, 2_000),
        ];
    });
    let (_window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| LibraryView::new(stores.clone(), window, cx))
    });

    // Cursor enters row A (index 0).
    view.update(cx, |v, cx| v.set_row_hover(0, true, cx));
    view.read_with(cx, |v, _| assert_eq!(v.hovered_row(), Some(0)));

    // Cursor moves to row B: B's enter (true) fires first…
    view.update(cx, |v, cx| v.set_row_hover(1, true, cx));
    view.read_with(cx, |v, _| assert_eq!(v.hovered_row(), Some(1)));

    // …then A's leave (false) arrives late. Because A is no longer the hovered
    // row, the clear is a no-op — hover stays on B.
    view.update(cx, |v, cx| v.set_row_hover(0, false, cx));
    view.read_with(cx, |v, _| {
        assert_eq!(
            v.hovered_row(),
            Some(1),
            "a stale leave from the left row must not clobber the entered row's hover"
        );
    });

    // A real leave of the currently-hovered row still clears.
    view.update(cx, |v, cx| v.set_row_hover(1, false, cx));
    view.read_with(cx, |v, _| assert_eq!(v.hovered_row(), None));
}

#[gpui::test]
fn library_pencil_begins_rename(cx: &mut TestAppContext) {
    // The hover-revealed pencil starts the inline rename (replacing the
    // unreachable double-click trigger). It calls `begin_rename` directly, so
    // exercising that method is the same path the pencil's on_click takes.
    let stores = stub_stores(cx, |s| {
        s.spaces = vec![stub_space("s1", Some("Tides"), None, 1_000)];
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| LibraryView::new(stores.clone(), window, cx))
    });

    view.read_with(cx, |v, _| assert_eq!(v.renaming_space_id(), None));
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.begin_rename("s1".into(), Some("Tides".into()), window, cx);
        });
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert_eq!(
            v.renaming_space_id(),
            Some("s1"),
            "the pencil's begin_rename puts the row into inline-rename mode"
        );
    });
}

#[gpui::test]
fn library_pencil_click_does_not_also_open_row(cx: &mut TestAppContext) {
    // REGRESSION (wave-4 QA round 2, finding 4): clicking the rename pencil
    // sometimes BOTH started the inline rename AND opened the space window — a
    // propagation/phase race. The row records its own pending mouse-down and
    // captures it on the mouse-up *capture* phase (before the button's bubble
    // click + stop_propagation runs), and `begin_rename` reshaping the row
    // between down and up (title → input, reveal slot hidden) moves the
    // hitboxes so the up can complete the ROW's click too. The fix blocks
    // propagation on the affordance slot for BOTH mouse-down and mouse-up, and
    // defers `begin_rename` so the click sequence resolves against the old
    // layout.
    //
    // Mechanism (proven from gpui `div.rs` paint + `window.rs::dispatch_mouse_event`):
    //   - On mouse-DOWN (bubble), every element with click listeners whose
    //     hitbox is hovered records its own `pending_mouse_down`. The pencil is
    //     nested in the row, so the row's hitbox is hovered too → the row arms.
    //   - On mouse-UP, capture phase runs outer→inner and each element with a
    //     pending down captures it (if still hovered); bubble phase runs
    //     inner→outer and fires the captured click, breaking on stop_propagation.
    //   The pencil's `stop_propagation` covers the *synchronous* up, but the
    //   intermittent production failure is the reshape race: the pencil's click
    //   runs `begin_rename`, which hides the reveal slot and swaps the title for
    //   an input *between* the captured down and the firing up, so the up lands
    //   on the row and completes the ROW's click too — opening the space on top
    //   of the rename.
    //
    // The fix blocks propagation on the affordance slot for BOTH mouse-down (so
    // the row never arms) AND mouse-up, and defers `begin_rename` so the reshape
    // happens after the gesture resolves. This drives the real DOWN+UP gesture
    // over the pencil's hitbox (located via `debug_bounds`) and asserts the
    // invariant the fix guarantees: the pencil renames and the row never opens.
    //
    // Harness note: the *intermittent* production failure is a multi-frame
    // timing race (hover flicker / re-paint between the physical down and up)
    // that the deterministic test dispatcher cannot reproduce against the
    // committed code — gpui's own #24600 "clear pending if the hitbox moved out
    // from under the up" guard, the row dropping its `on_click` the moment it
    // reshapes into rename mode, and the pencil's `stop_propagation` together
    // cover every *synchronous* path. This test therefore guards the invariant
    // (and regresses if the protections are removed: stripping the pencil's
    // propagation blocking makes `open_space` fire here, `left: 1 == right: 0`).
    // The mechanism evidence above is the primary artifact; the manual repro is
    // clicking the pencil rapidly on a hovered row in the running app.
    let stores = stub_stores(cx, |s| {
        s.spaces = vec![stub_space("s1", Some("Tides"), None, 1_000)];
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| LibraryView::new(stores.clone(), window, cx))
    });
    cx.run_until_parked();

    // Reveal the row's affordances (the pencil only paints while hovered).
    view.update(cx, |v, _| v.set_hovered_for_test(Some(0)));
    cx.run_until_parked();

    let mut vcx = VisualTestContext::from_window(window, cx);
    // Give the window a definite size so the virtualized list lays out its
    // rows (and the pencil's painted bounds become queryable).
    vcx.simulate_resize(gpui::size(px(520.), px(620.)));
    vcx.run_until_parked();
    let bounds = vcx
        .debug_bounds("rename-pencil-0")
        .expect("the hover-revealed rename pencil must be painted with its debug selector");
    let center: Point<gpui::Pixels> = bounds.center();

    // The real user gesture: a single down+up click on the pencil.
    vcx.simulate_click(center, Modifiers::default());
    vcx.run_until_parked();

    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.renaming_space_id(),
            Some("s1"),
            "clicking the pencil must start the inline rename"
        );
        assert_eq!(
            v.open_space_requests_for_test(),
            0,
            "clicking the pencil must NOT also open the row (the propagation/reshape race)"
        );
    });
}

// ---------------------------------------------------------------------------
// Onboarding state machine
// ---------------------------------------------------------------------------

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
        cx.new(|cx| SettingsView::new(stores.clone(), window, cx))
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

    view.update(cx, |v, cx| v.select(SettingsPane::Backends, cx));
    view.read_with(cx, |v, _| assert_eq!(v.selected(), SettingsPane::Backends));
}

#[gpui::test]
fn settings_backends_tabs_switch(cx: &mut TestAppContext) {
    use eidola_gui::backends_settings::BackendsTab;

    // The Backends pane's internal tab strip is view-local state. The three
    // tabs (Eidola · Local · External) split the registry; the Eidola tab is
    // the connection + trust surface (base URL / measurements / hardware CAs).
    let stores = stub_stores_with_config(cx);
    let (_window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SettingsView::new(stores.clone(), window, cx))
    });
    let pane = view.read_with(cx, |v, _| v.backends_pane());

    // Eidola is the resting tab.
    pane.read_with(cx, |p, _| assert_eq!(p.tab(), BackendsTab::Eidola));

    pane.update(cx, |p, cx| p.select_tab(BackendsTab::Local, cx));
    pane.read_with(cx, |p, _| assert_eq!(p.tab(), BackendsTab::Local));

    pane.update(cx, |p, cx| p.select_tab(BackendsTab::External, cx));
    pane.read_with(cx, |p, _| assert_eq!(p.tab(), BackendsTab::External));

    pane.update(cx, |p, cx| p.select_tab(BackendsTab::Eidola, cx));
    pane.read_with(cx, |p, _| assert_eq!(p.tab(), BackendsTab::Eidola));
}

#[gpui::test]
fn settings_nav_gates_account_wallet_on_eidola(cx: &mut TestAppContext) {
    // Account and Wallet nav items render only while the eidola backend is
    // enabled — nav visibility doubles as state.
    let enabled = stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.eidola_trust = Some(eidola_trust());
        s.backends = backends_fixture(true);
    });
    let (_w, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SettingsView::new(enabled.clone(), window, cx))
    });
    view.read_with(cx, |v, cx| {
        assert_eq!(
            v.visible_panes(cx),
            vec![
                SettingsPane::General,
                SettingsPane::Backends,
                SettingsPane::Templates,
                SettingsPane::Agents,
                SettingsPane::Account,
                SettingsPane::Wallet,
            ],
            "eidola enabled shows all panes"
        );
    });

    let disabled = stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.eidola_trust = Some(eidola_trust());
        s.backends = backends_fixture(false);
    });
    let (_w2, view2) = open_view(cx, |window, cx| {
        cx.new(|cx| SettingsView::new(disabled.clone(), window, cx))
    });
    view2.read_with(cx, |v, cx| {
        assert_eq!(
            v.visible_panes(cx),
            vec![
                SettingsPane::General,
                SettingsPane::Backends,
                SettingsPane::Templates,
                SettingsPane::Agents,
            ],
            "eidola disabled hides Account and Wallet"
        );
    });
}

#[gpui::test]
fn settings_selection_falls_back_when_eidola_disabled(cx: &mut TestAppContext) {
    // Selecting Account, then disabling eidola (the toggle lives in Backends →
    // Eidola, but a Change::Backends refresh can arrive any time), must fall
    // the selection back to Backends — never a blank body or phantom nav.
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.eidola_trust = Some(eidola_trust());
        s.backends = backends_fixture(true);
    });
    let (_w, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SettingsView::new(stores.clone(), window, cx))
    });

    view.update(cx, |v, cx| v.select(SettingsPane::Account, cx));
    view.read_with(cx, |v, _| assert_eq!(v.selected(), SettingsPane::Account));

    // Disable the eidola singleton through the store (the optimistic flip
    // fires the bus-less observer path).
    stores
        .backends
        .update(cx, |b, cx| b.set_enabled("eidola".into(), false, cx));
    cx.run_until_parked();

    view.read_with(cx, |v, _| {
        assert_eq!(
            v.selected(),
            SettingsPane::Backends,
            "a hidden selection reconciles to Backends"
        );
    });
}

#[gpui::test]
fn settings_eidola_url_editor_save_cancel_revert(cx: &mut TestAppContext) {
    // The base-URL override editor moved out of General into Backends → Eidola.
    // Its edit/cancel state machine is view-local; save/revert write through
    // the config store (a stub write stops at the backend guard — the same
    // honest no-op the other panes assert).
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.eidola_trust = Some(eidola_trust());
        s.backends = backends_fixture(true);
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SettingsView::new(stores.clone(), window, cx))
    });
    let pane = view.read_with(cx, |v, _| v.backends_pane());

    // At rest: not editing.
    pane.read_with(cx, |p, _| assert!(!p.editing_base_url()));

    // Change… enters the edit state; Cancel leaves it.
    cx.update_window(window, |_, window, cx| {
        pane.update(cx, |p, cx| p.begin_edit_base_url(window, cx));
    })
    .unwrap();
    pane.read_with(cx, |p, _| assert!(p.editing_base_url()));
    pane.update(cx, |p, cx| p.cancel_edit_base_url(cx));
    pane.read_with(cx, |p, _| assert!(!p.editing_base_url()));

    // Save exits the edit state (stub write is a no-op past the guard).
    cx.update_window(window, |_, window, cx| {
        pane.update(cx, |p, cx| p.begin_edit_base_url(window, cx));
    })
    .unwrap();
    pane.update(cx, |p, cx| p.save_base_url(cx));
    pane.read_with(cx, |p, _| assert!(!p.editing_base_url()));

    // Revert-to-pin is a no-op-safe path too.
    pane.update(cx, |p, cx| p.revert_base_url(cx));
    pane.read_with(cx, |p, _| assert!(!p.editing_base_url()));

    // Reverting measurements routes through the store without panicking.
    pane.update(cx, |p, cx| p.revert_measurements(cx));
}

/// The measurement add input and the hardware-CA textareas reveal in place
/// on demand (the base-URL row's edit-in-place shape, generalized — no
/// disclosure, no hidden state): "Trust a measurement…" / "Set custom
/// certificate…" open them, Cancel closes them.
#[gpui::test]
fn settings_eidola_trust_editors_reveal_in_place(cx: &mut TestAppContext) {
    use eidola_gui::backends_settings::CaKind;

    let stores = stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.eidola_trust = Some(eidola_trust());
        s.backends = backends_fixture(true);
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SettingsView::new(stores.clone(), window, cx))
    });
    let pane = view.read_with(cx, |v, _| v.backends_pane());

    // At rest: value displays only, no inputs.
    pane.read_with(cx, |p, _| {
        assert!(!p.adding_measurement(), "add input hidden at rest");
        assert_eq!(p.editing_ca(), None, "CA textareas hidden at rest");
    });

    // Trust a measurement… reveals the add input; Cancel hides it.
    cx.update_window(window, |_, window, cx| {
        pane.update(cx, |p, cx| p.begin_add_measurement(window, cx));
    })
    .unwrap();
    pane.read_with(cx, |p, _| assert!(p.adding_measurement()));
    pane.update(cx, |p, cx| p.cancel_add_measurement(cx));
    pane.read_with(cx, |p, _| assert!(!p.adding_measurement()));

    // Set custom certificate… reveals that CA's textarea; Cancel hides it.
    cx.update_window(window, |_, window, cx| {
        pane.update(cx, |p, cx| p.begin_edit_ca(CaKind::Root, window, cx));
    })
    .unwrap();
    pane.read_with(cx, |p, _| assert_eq!(p.editing_ca(), Some(CaKind::Root)));
    pane.update(cx, |p, cx| p.cancel_edit_ca(cx));
    pane.read_with(cx, |p, _| assert_eq!(p.editing_ca(), None));
}

/// The Eidola trust editors route through the config store. On a stub (no
/// backend) each write stops at the store's backend guard — an honest no-op —
/// so a successful submit clears its input and nothing panics. Exercises the
/// add-measurement, untrust, and CA set/clear paths.
#[gpui::test]
fn settings_eidola_trust_editors_call_through(cx: &mut TestAppContext) {
    use eidola_gui::backends_settings::CaKind;

    let mut trust = eidola_trust();
    trust.trusted_measurements = vec![eidola_app_core::MeasurementInfo {
        snp: "9d2bb3ef58af1e7c0c12f3b4a5d6e7f8901a2b3c4d5e6f708192a3b4c5d6e7f8".into(),
        tdx_rtmr1: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into(),
        tdx_rtmr2: "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210".into(),
    }];
    trust.trusted_measurements_are_override = true;
    trust.has_hardware_root_ca = true;
    trust.hardware_root_ca_pem =
        Some("-----BEGIN CERTIFICATE-----\nMIIBcustomroot\n-----END CERTIFICATE-----".into());

    let stores = stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.eidola_trust = Some(trust);
        s.backends = backends_fixture(true);
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SettingsView::new(stores.clone(), window, cx))
    });
    let pane = view.read_with(cx, |v, _| v.backends_pane());

    // Add a measurement: reveal the input, seed the triple, submit — expect a
    // cleared field and the input closed again (the stub write reports no
    // error, so the success branch runs).
    cx.update_window(window, |_, window, cx| {
        pane.update(cx, |p, cx| p.begin_add_measurement(window, cx));
    })
    .unwrap();
    let triple = "aa11bb22cc33dd44ee55ff66aa11bb22cc33dd44ee55ff66aa11bb22cc33dd44\
                  ee55ff66aa11bb22cc33dd44:\
                  0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef:\
                  fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";
    let add_input = pane.read_with(cx, |p, _| p.add_measurement_input());
    cx.update_window(window, |_, window, cx| {
        add_input.update(cx, |s, cx| s.set_value(triple, window, cx));
    })
    .unwrap();
    cx.update_window(window, |_, window, cx| {
        pane.update(cx, |p, cx| p.submit_add_measurement(window, cx));
    })
    .unwrap();
    add_input.read_with(cx, |s, _| {
        assert!(s.value().is_empty(), "a successful add clears the input");
    });
    pane.read_with(cx, |p, _| {
        assert!(
            !p.adding_measurement(),
            "a successful add closes the input again"
        );
    });

    // Untrust routes through without panicking.
    pane.update(cx, |p, cx| {
        p.untrust_measurement(
            "9d2bb3ef58af1e7c0c12f3b4a5d6e7f8901a2b3c4d5e6f708192a3b4c5d6e7f8".into(),
            cx,
        )
    });

    // CA set: reveal the textarea, seed a PEM, submit — expect a cleared
    // field and the editor closed; then Clear routes through.
    cx.update_window(window, |_, window, cx| {
        pane.update(cx, |p, cx| p.begin_edit_ca(CaKind::Root, window, cx));
    })
    .unwrap();
    let ca_input = pane.read_with(cx, |p, _| p.ca_input(CaKind::Root));
    cx.update_window(window, |_, window, cx| {
        ca_input.update(cx, |s, cx| {
            s.set_value(
                "-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----",
                window,
                cx,
            )
        });
    })
    .unwrap();
    cx.update_window(window, |_, window, cx| {
        pane.update(cx, |p, cx| p.submit_ca(CaKind::Root, window, cx));
    })
    .unwrap();
    ca_input.read_with(cx, |s, _| {
        assert!(s.value().is_empty(), "a successful CA set clears the input");
    });
    pane.read_with(cx, |p, _| {
        assert_eq!(
            p.editing_ca(),
            None,
            "a successful CA set closes the editor"
        );
    });
    pane.update(cx, |p, cx| p.clear_ca(CaKind::Root, cx));
}

#[gpui::test]
fn settings_account_pane_reachable_at_top_level(cx: &mut TestAppContext) {
    // Account is a top-level pane again; its reset-confirm flow is reachable
    // through `SettingsView::account_pane()`.
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.eidola_trust = Some(eidola_trust());
        s.backends = backends_fixture(true);
        s.balances = Some(BalancesResult {
            available: 5_000_000,
            pools: Vec::new(),
        });
    });
    let (_w, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SettingsView::new(stores.clone(), window, cx))
    });

    view.update(cx, |v, cx| v.select(SettingsPane::Account, cx));
    view.read_with(cx, |v, _| assert_eq!(v.selected(), SettingsPane::Account));

    let account = view.read_with(cx, |v, _| v.account_pane());
    account.read_with(cx, |a, _| assert!(!a.reset_armed()));
    account.update(cx, |a, cx| a.request_reset(cx));
    account.read_with(cx, |a, _| assert!(a.reset_armed()));
}

#[gpui::test]
fn settings_asks_the_account_pane_for_its_live_reads_on_every_visit(cx: &mut TestAppContext) {
    // The Account pane's balance and subscription are live server reads that
    // **no `Change` can invalidate**: a webhook credits the account, or the
    // reader cancels a subscription in a browser portal, and nothing local
    // commits for the bus to announce. `SettingsView` builds all six panes
    // when the *window* opens, so construction is not the moment to ask —
    // selecting the pane is, and selecting it again is too.
    //
    // Only a real backend can answer "did the pane ask?": the stub stores'
    // `refresh_*` are no-ops by design. Nothing here waits on the network —
    // a refresh marks its cell before it spawns, which is the whole
    // assertion.
    use eidola_gui::loadable::Loadable;

    cx.executor().allow_parking();
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    let dir = tempfile::tempdir().unwrap();
    let core = std::sync::Arc::new(
        AppCore::new(dir.path().to_path_buf(), dir.path().join("data")).expect("open core"),
    );
    core.runtime()
        .block_on(core.set_base_url("https://127.0.0.1:1/v1".into()))
        .unwrap();
    // A configured account is what makes the balance and subscription reads
    // meaningful; the pane gates them on exactly this.
    core.set_account_credentials("acct".into(), "secret".into())
        .unwrap();
    let stores = cx.update(|cx| Stores::for_test(core.clone(), cx));
    stores.config.update(cx, |s, cx| s.refresh(cx));

    let (_w, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SettingsView::new(stores.clone(), window, cx))
    });

    // Settings opens on General. Every pane exists; none of Account's live
    // reads has been asked for.
    stores.account.read_with(cx, |s, _| {
        assert!(
            matches!(s.subscription(), Loadable::NotLoaded),
            "building the pane must not read the subscription"
        );
        assert!(
            matches!(s.balances(), Loadable::NotLoaded),
            "building the pane must not read the balance"
        );
    });

    view.update(cx, |v, cx| v.select(SettingsPane::Account, cx));
    stores.account.read_with(cx, |s, _| {
        assert!(
            !matches!(s.subscription(), Loadable::NotLoaded),
            "selecting the pane must ask for the subscription"
        );
        assert!(
            !matches!(s.balances(), Loadable::NotLoaded),
            "selecting the pane must ask for the balance"
        );
    });

    // Now the reader leaves, changes their subscription in a browser, and
    // comes back. Standing in for "the answer we already have", so the
    // second visit has something it could wrongly leave alone.
    stores.account.update(cx, |s, cx| {
        s.set_subscription_for_test(
            Loadable::loaded(eidola_app_core::SubscriptionInfo {
                state: eidola_app_core::SubscriptionState::Active,
                status: Some("active".into()),
                current_period_end: None,
            }),
            cx,
        );
    });
    view.update(cx, |v, cx| v.select(SettingsPane::General, cx));
    view.update(cx, |v, cx| v.select(SettingsPane::Account, cx));
    stores.account.read_with(cx, |s, _| {
        assert!(
            !matches!(s.subscription(), Loadable::Loaded { stale: false, .. }),
            "coming back to the pane must re-ask rather than leave the \
             previous answer standing as current"
        );
    });

    // Deterministic teardown — join the in-flight bridge tasks before `cx`
    // drops the last `Arc<AppCore>` (see `record_refresh_supersedes_in_flight_fetch`
    // for why the runtime must be idle first).
    while core.runtime().metrics().num_alive_tasks() > 0 {
        std::thread::yield_now();
    }
}

#[gpui::test]
fn settings_backends_pane_stub_ops_stop_at_backend_guard(cx: &mut TestAppContext) {
    // With stub stores, every local-model operation clears the standing
    // error and stops at the backend guard — an honest no-op, no phantom
    // Loading states, no panics.
    let stores = stub_stores(cx, |s| {
        s.local_models = Some(eidola_app_core::LocalModelsState {
            engine_path: None,
            external: Vec::new(),
            models: vec![eidola_app_core::LocalModelInfo {
                id: "tiny@local".into(),
                slug: "tiny".into(),
                display_name: "Tiny".into(),
                file_name: "tiny.gguf".into(),
                size_bytes: Some(1_000_000_000),
                source_url: None,
                status: eidola_app_core::LocalModelStatus::Loaded {
                    port: 4242,
                    context_tokens: 8192,
                    pinned: false,
                },
                last_error: None,
                on_disk: true,
            }],
        });
    });
    let (_window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SettingsView::new(stores.clone(), window, cx))
    });

    let backends_view = view.read_with(cx, |v, _| v.backends_pane());
    backends_view.update(cx, |m, cx| {
        m.download_catalog("https://example.com/some.gguf", cx)
    });
    cx.run_until_parked();

    stores.local_models.read_with(cx, |s, _| {
        assert!(s.op_error().is_none(), "stub ops must not surface errors");
        // The fixture snapshot survives untouched (no refresh happened).
        assert_eq!(s.models().len(), 1);
        assert_eq!(s.loaded_models().len(), 1);
        assert_eq!(s.loaded_models()[0].id, "tiny@local");
    });
}

#[gpui::test]
fn space_model_display_splits_name_and_backend(cx: &mut TestAppContext) {
    // The gutter/chip display pair: human model name over backend name.
    let stores = stub_stores(cx, |s| {
        s.backends = vec![
            eidola_app_core::BackendInfo {
                id: "local".into(),
                kind: eidola_app_core::BackendKind::Local,
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
            eidola_app_core::BackendInfo {
                id: "my-vllm".into(),
                kind: eidola_app_core::BackendKind::OpenAi,
                display_name: "My vLLM box".into(),
                enabled: true,
                base_url: Some("http://x".into()),
                has_api_key: false,
                models_dir: None,
                model_overrides: None,
                engine_path: None,
                auto_start: true,
                created_at: 1,
            },
        ];
        s.local_models = Some(eidola_app_core::LocalModelsState {
            engine_path: None,
            external: Vec::new(),
            models: vec![eidola_app_core::LocalModelInfo {
                id: "gemma-4-E2B_q4_0-it@local".into(),
                slug: "gemma-4-E2B_q4_0-it".into(),
                display_name: "Gemma 4 E2B".into(),
                file_name: "gemma-4-E2B_q4_0-it.gguf".into(),
                size_bytes: Some(3_349_514_112),
                source_url: None,
                status: eidola_app_core::LocalModelStatus::Available,
                last_error: None,
                on_disk: true,
            }],
        });
    });
    let (_window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SpaceView::new(stores, None, WindowInput::new(cx), window, cx))
    });

    view.read_with(cx, |v, cx| {
        // An engine model resolves to its sidecar display name + "Local".
        assert_eq!(
            v.model_display("gemma-4-E2B_q4_0-it@local", cx),
            ("Gemma 4 E2B".into(), "Local".into())
        );
        // A catalog model keeps its wire id as the name; backend resolves.
        assert_eq!(
            v.model_display("gemma4-31b", cx),
            ("gemma4-31b".into(), "Eidola".into())
        );
        assert_eq!(
            v.model_display("qwen3-8b@my-vllm", cx),
            ("qwen3-8b".into(), "My vLLM box".into())
        );
        // An unknown/deleted engine model falls back to its raw parts.
        assert_eq!(
            v.model_display("gone@local", cx),
            ("gone".into(), "Local".into())
        );
    });
}

/// The picker offers files, and a snapshot row is not always one: a failed
/// download leaves a row carrying only its error, and a mid-download row's
/// bytes are still in a `.part`. Both belong in Settings (that is where the
/// error and the progress are read) and neither can be picked.
#[gpui::test]
fn model_picker_offers_only_rows_with_a_file_behind_them(cx: &mut TestAppContext) {
    let row = |id: &str, status, on_disk| eidola_app_core::LocalModelInfo {
        id: id.into(),
        slug: id.split('@').next().unwrap().into(),
        display_name: id.into(),
        file_name: format!("{}.gguf", id.split('@').next().unwrap()),
        size_bytes: None,
        source_url: None,
        status,
        last_error: None,
        on_disk,
    };
    let stores = stub_stores(cx, |s| {
        s.local_models = Some(eidola_app_core::LocalModelsState {
            engine_path: None,
            models: vec![
                row(
                    "here@local",
                    eidola_app_core::LocalModelStatus::Available,
                    true,
                ),
                // The failed re-download of a slug whose file went away.
                row(
                    "gone@local",
                    eidola_app_core::LocalModelStatus::Available,
                    false,
                ),
                row(
                    "coming@local",
                    eidola_app_core::LocalModelStatus::Downloading {
                        received: 1,
                        total: None,
                    },
                    false,
                ),
            ],
            external: vec![eidola_app_core::ExternalEngineBackend {
                backend_id: "mine".into(),
                display_name: "Mine".into(),
                enabled: true,
                models_dir: "/models".into(),
                engine_path: None,
                auto_start: false,
                models: vec![
                    row(
                        "there@mine",
                        eidola_app_core::LocalModelStatus::Available,
                        true,
                    ),
                    row(
                        "ghost@mine",
                        eidola_app_core::LocalModelStatus::Available,
                        false,
                    ),
                ],
            }],
        });
    });
    stores.local_models.read_with(cx, |s, _| {
        assert_eq!(
            s.selectable_models()
                .iter()
                .map(|m| m.id.as_str())
                .collect::<Vec<_>>(),
            vec!["here@local"]
        );
        assert_eq!(
            s.external_selectable_models("mine")
                .iter()
                .map(|m| m.id.as_str())
                .collect::<Vec<_>>(),
            vec!["there@mine"]
        );
        // Settings still shows every row — the error and the progress live there.
        assert_eq!(s.models().len(), 3);
        assert_eq!(s.external_models("mine").len(), 2);
    });
}

#[gpui::test]
fn local_models_store_pin_op_is_stub_safe(cx: &mut TestAppContext) {
    // The pin/unpin op follows the standard thin-initiating-call shape:
    // with stub stores it clears the standing error and stops at the
    // backend guard — no phantom state, no panic.
    let stores = stub_stores(cx, |s| {
        s.local_models = Some(eidola_app_core::LocalModelsState {
            engine_path: None,
            external: Vec::new(),
            models: vec![eidola_app_core::LocalModelInfo {
                id: "tiny@local".into(),
                slug: "tiny".into(),
                display_name: "Tiny".into(),
                file_name: "tiny.gguf".into(),
                size_bytes: Some(1_000_000_000),
                source_url: None,
                status: eidola_app_core::LocalModelStatus::Loaded {
                    port: 4242,
                    context_tokens: 8192,
                    pinned: false,
                },
                last_error: None,
                on_disk: true,
            }],
        });
    });
    stores.local_models.update(cx, |s, cx| {
        s.set_pinned("tiny@local".into(), true, cx);
    });
    cx.run_until_parked();
    stores.local_models.read_with(cx, |s, _| {
        assert!(s.op_error().is_none(), "stub ops must not surface errors");
        // The fixture snapshot is untouched (no backend, no refresh) —
        // the real pin lands via Change::LocalModels in production.
        assert_eq!(s.selectable_models().len(), 1);
    });
}

#[gpui::test]
fn settings_backends_pane_add_form_and_toggle(cx: &mut TestAppContext) {
    use eidola_gui::backends_settings::AddKind;

    let stores = stub_stores(cx, |s| {
        s.backends = vec![eidola_app_core::BackendInfo {
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
        }];
    });
    let (_window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SettingsView::new(stores.clone(), window, cx))
    });
    let pane = view.read_with(cx, |v, _| v.backends_pane());

    // The add form opens per kind, is idempotent, and cancel closes it.
    pane.read_with(cx, |p, _| assert_eq!(p.adding(), None));
    cx.update(|cx| {
        _window
            .update(cx, |_, window, cx| {
                pane.update(cx, |p, cx| p.begin_add(AddKind::OpenAi, window, cx));
            })
            .unwrap();
    });
    pane.read_with(cx, |p, _| assert_eq!(p.adding(), Some(AddKind::OpenAi)));
    cx.update(|cx| {
        _window
            .update(cx, |_, window, cx| {
                pane.update(cx, |p, cx| p.begin_add(AddKind::LlamaCpp, window, cx));
            })
            .unwrap();
    });
    pane.read_with(cx, |p, _| assert_eq!(p.adding(), Some(AddKind::LlamaCpp)));
    pane.update(cx, |p, cx| p.cancel_add(cx));
    pane.read_with(cx, |p, _| assert_eq!(p.adding(), None));

    // Submitting with no form open is a quiet no-op.
    pane.update(cx, |p, cx| p.submit_add(cx));
    pane.read_with(cx, |p, _| assert_eq!(p.adding(), None));

    // Disabling the eidola backend flips the cached row immediately (the
    // optimistic write; the stub has no backend, so the op stops there).
    pane.update(cx, |p, cx| p.toggle_backend("eidola".into(), false, cx));
    stores.backends.read_with(cx, |b, _| {
        assert!(!b.is_enabled("eidola"), "optimistic flip must be visible");
        assert!(b.op_error().is_none(), "stub ops must not surface errors");
    });
}

#[gpui::test]
fn settings_backends_pane_auto_start_toggle(cx: &mut TestAppContext) {
    // A llamacpp backend's auto-start toggle flips the cached row optimistically
    // and stops at the stub backend guard (no phantom op-error).
    let stores = stub_stores(cx, |s| {
        s.backends = vec![eidola_app_core::BackendInfo {
            id: "my-box".into(),
            kind: eidola_app_core::BackendKind::LlamaCpp,
            display_name: "My box".into(),
            enabled: true,
            base_url: None,
            has_api_key: false,
            models_dir: Some("/Users/me/models".into()),
            model_overrides: None,
            engine_path: None,
            auto_start: true,
            created_at: 0,
        }];
    });
    let (_window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SettingsView::new(stores.clone(), window, cx))
    });
    let pane = view.read_with(cx, |v, _| v.backends_pane());

    pane.update(cx, |p, cx| p.set_auto_start("my-box".into(), false, cx));
    stores.backends.read_with(cx, |b, _| {
        assert!(
            !b.get("my-box").unwrap().auto_start,
            "optimistic auto-start flip must be visible"
        );
        assert!(b.op_error().is_none(), "stub ops must not surface errors");
    });
}

/// The circadian appearance choices write through the `ConfigStore`; on a
/// stub (no backend) the write stops at the store's backend guard, so the
/// snapshot keeps the fixture values — the same honest no-op the other
/// panes assert. The real write-through is covered at the store level in
/// `tests/stores.rs` (`config_store_circadian_settings_write_through`).
#[gpui::test]
fn settings_appearance_choices_route_through_config_store(cx: &mut TestAppContext) {
    use eidola_app_core::config::{AppearanceSetting, TimeOfDayTint};

    let stores = stub_stores_with_config(cx);
    let (_window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SettingsView::new(stores.clone(), window, cx))
    });

    let general = view.read_with(cx, |v, _| v.general());
    general.update(cx, |g, cx| {
        g.set_appearance(AppearanceSetting::Night, cx);
        g.set_time_of_day_tint(TimeOfDayTint::Off, cx);
        g.set_light_character(eidola_app_core::config::LightCharacter::Warm, cx);
    });

    stores.config.read_with(cx, |c, _| {
        let s = c.state().expect("fixture config state");
        assert_eq!(
            s.appearance,
            AppearanceSetting::System,
            "stub write must stop at the backend guard"
        );
        assert_eq!(s.time_of_day_tint, TimeOfDayTint::On);
        assert_eq!(
            s.light_character,
            eidola_app_core::config::LightCharacter::Neutral
        );
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

#[gpui::test]
fn a_new_identity_inherits_no_state_from_the_one_it_replaced(cx: &mut TestAppContext) {
    // Everything this pane holds about an account — a revealed secret, an
    // armed reset, a checkout or portal in flight — describes the identity
    // configured when it was set. None of it means anything about the next
    // one, and a pending flag left standing is the worst of them: it renders
    // as "Opening…" and the early return refuses every click until a request
    // belonging to a forgotten account happens to land.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| AccountView::new(stores.clone(), window, cx))
    });
    draw_window(cx, window);

    // Put the pane into every account-scoped state at once.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.toggle_account_secret_revealed(window, cx));
    })
    .unwrap();
    view.update(cx, |v, cx| {
        v.request_reset(cx);
        v.begin_manage(cx);
        v.begin_checkout("price_month".into(), cx);
    });
    draw_window(cx, window);
    view.read_with(cx, |v, _| {
        assert!(v.account_secret_revealed(), "precondition: revealed");
        assert!(v.reset_armed(), "precondition: reset armed");
        assert!(v.manage_pending(), "precondition: portal in flight");
        assert_eq!(
            v.checkout_pending(),
            Some("price_month"),
            "precondition: checkout in flight"
        );
    });

    // A different account is configured under it.
    stores.config.update(cx, |c, _| {
        let mut state = config_state(true);
        state.account_id = Some("00000000-0000-7000-8000-000000000444".into());
        state.account_secret = Some("the-next-accounts-secret".into());
        c.set_state_for_test(Some(state));
    });
    draw_window(cx, window);

    view.read_with(cx, |v, _| {
        assert!(
            !v.manage_pending(),
            "the new account's billing door must not read as already opening"
        );
        assert_eq!(
            v.checkout_pending(),
            None,
            "the new account's plans must be purchasable"
        );
        assert!(!v.account_secret_revealed(), "the new secret starts masked");
        assert!(
            !v.reset_armed(),
            "an arming named the account it was armed over"
        );
    });
}

#[gpui::test]
fn onboarding_linking_another_account_clears_the_checkout_it_started(cx: &mut TestAppContext) {
    // Onboarding's half of the same rule. Its back-chevron reaches the
    // existing-account slide, so a reader can press a plan and then link a
    // different account; returning to Purchase must not find the row still
    // in flight for the account they walked away from.
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
    });
    let (_w, view) = open_onboarding(cx, &stores);

    view.update(cx, |v, cx| v.begin_checkout("price_month".into(), cx));
    view.read_with(cx, |v, _| {
        assert_eq!(v.checkout_pending(), Some("price_month"));
    });

    // Linking a different account commits a new identity — driven through
    // the same completion the request's own task runs.
    view.update(cx, |v, cx| {
        v.finish_verify(
            Ok(eidola_app_core::BalancesResult {
                available: 1_000_000,
                pools: Vec::new(),
            }),
            cx,
        )
    });
    view.read_with(cx, |v, _| {
        assert_eq!(
            v.checkout_pending(),
            None,
            "a checkout started for the previous account must not survive the link"
        );
    });
}

#[gpui::test]
fn a_reset_disowns_a_pending_link_before_the_bus_catches_up(cx: &mut TestAppContext) {
    // The guard has to read past `ConfigStore`. A reset commits to app-core's
    // config synchronously and then emits `Change::Config`; the store's cached
    // snapshot only catches up when the bus bridge delivers that on a later
    // tick. A mint landing inside that window would be compared against a
    // cached id that still names the account which has just been forgotten —
    // the guard asking the stale copy about the very staleness it exists to
    // detect, and passing. So no `run_until_parked` here between the reset and
    // the mint: the gap is the test.
    cx.executor().allow_parking();
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    let dir = tempfile::tempdir().unwrap();
    let core = std::sync::Arc::new(
        AppCore::new(dir.path().to_path_buf(), dir.path().join("data")).expect("open core"),
    );
    core.runtime()
        .block_on(core.set_base_url("https://127.0.0.1:1/v1".into()))
        .unwrap();
    core.set_account_credentials("acct-a".into(), "secret-a".into())
        .unwrap();
    let stores = cx.update(|cx| Stores::for_test(core.clone(), cx));
    stores.config.update(cx, |s, cx| s.refresh(cx));

    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| AccountView::new(stores.clone(), window, cx))
    });
    draw_window(cx, window);

    let minted_for = Some(gpui::SharedString::from("acct-a"));

    // The account is forgotten. Nothing is pumped, so the cache still says
    // "acct-a" — which is precisely the state the old guard trusted.
    core.reset_account().expect("reset");
    stores.config.read_with(cx, |c, _| {
        assert_eq!(
            c.state().and_then(|s| s.account_id.as_deref()),
            Some("acct-a"),
            "the cache must still be stale here, or this test proves nothing"
        );
    });

    view.update(cx, |v, cx| {
        v.finish_manage(
            minted_for,
            Ok("https://billing.example/portal/disowned".into()),
            cx,
        )
    });
    assert_eq!(
        cx.opened_url(),
        None,
        "a portal for the account just reset must not open, cache lag or not"
    );
    view.read_with(cx, |v, _| {
        assert!(
            v.manage_error()
                .is_some_and(|e| e.contains("account changed"))
        );
    });
}

#[gpui::test]
fn a_payment_link_minted_for_a_replaced_account_is_never_opened(cx: &mut TestAppContext) {
    // Both billing doors mint against the credentials held at click time and
    // then take a round trip. Reset or replace the account inside that window
    // and the link that comes back belongs to the previous identity — opening
    // it would show the old account's portal, or fund an account the reader no
    // longer holds the secret for, under the new account's name.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| AccountView::new(stores.clone(), window, cx))
    });
    draw_window(cx, window);

    let minted_for = Some(gpui::SharedString::from(
        config_state(true)
            .account_id
            .expect("fixture has an account"),
    ));

    // The account is unchanged: the link opens.
    view.update(cx, |v, cx| {
        v.finish_manage(
            minted_for.clone(),
            Ok("https://billing.example/portal/current".into()),
            cx,
        )
    });
    assert_eq!(
        cx.opened_url().as_deref(),
        Some("https://billing.example/portal/current"),
        "the current account's portal must open"
    );
    view.read_with(cx, |v, _| assert!(v.manage_error().is_none()));

    // A different account is configured while the next mint is in flight.
    stores.config.update(cx, |c, _| {
        let mut state = config_state(true);
        state.account_id = Some("00000000-0000-7000-8000-000000000222".into());
        state.account_secret = Some("a-different-accounts-secret".into());
        c.set_state_for_test(Some(state));
    });
    draw_window(cx, window);

    view.update(cx, |v, cx| {
        v.finish_manage(
            minted_for.clone(),
            Ok("https://billing.example/portal/stale".into()),
            cx,
        )
    });
    assert_eq!(
        cx.opened_url().as_deref(),
        Some("https://billing.example/portal/current"),
        "the replaced account's portal must not open"
    );
    view.read_with(cx, |v, _| {
        assert!(
            v.manage_error()
                .is_some_and(|e| e.contains("account changed")),
            "the refusal must be said out loud, not swallowed"
        );
    });

    // Checkout is the same door with the same hazard — money, not a portal.
    view.update(cx, |v, cx| {
        v.finish_checkout(
            minted_for,
            Ok("https://checkout.example/session/stale".into()),
            cx,
        )
    });
    assert_eq!(
        cx.opened_url().as_deref(),
        Some("https://billing.example/portal/current"),
        "a checkout funding the replaced account must not open"
    );
    view.read_with(cx, |v, _| {
        assert!(
            v.checkout_error()
                .is_some_and(|e| e.contains("account changed"))
        );
    });
}

#[gpui::test]
fn a_revealed_secret_re_masks_when_the_account_changes_under_it(cx: &mut TestAppContext) {
    // Revealing is consent to show one identity's secret. If the account is
    // reset, created or linked while the pane stays open, the field now holds
    // a credential nobody asked to see — so it must go back behind the mask
    // rather than inherit the previous identity's reveal.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| AccountView::new(stores.clone(), window, cx))
    });
    draw_window(cx, window);

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.toggle_account_secret_revealed(window, cx));
    })
    .unwrap();
    draw_window(cx, window);
    view.read_with(cx, |v, _| assert!(v.account_secret_revealed()));

    // A different account is now configured.
    stores.config.update(cx, |c, _| {
        let mut state = config_state(true);
        state.account_id = Some("00000000-0000-7000-8000-000000000222".into());
        state.account_secret = Some("a-different-accounts-secret".into());
        c.set_state_for_test(Some(state));
    });
    draw_window(cx, window);
    view.read_with(cx, |v, _| {
        assert!(
            !v.account_secret_revealed(),
            "the new identity's secret must start masked"
        );
    });

    // Revealing again, then losing the account entirely (a reset), re-masks
    // for the same reason.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.toggle_account_secret_revealed(window, cx));
    })
    .unwrap();
    draw_window(cx, window);
    view.read_with(cx, |v, _| assert!(v.account_secret_revealed()));
    stores.config.update(cx, |c, _| {
        let mut state = config_state(false);
        state.account_id = None;
        state.account_secret = None;
        c.set_state_for_test(Some(state));
    });
    draw_window(cx, window);
    view.read_with(cx, |v, _| {
        assert!(
            !v.account_secret_revealed(),
            "an account that went away leaves no reveal behind"
        );
    });
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
    let (window, view) = open_view(cx, |window, cx| {
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
    cx.update_window(window, |_, win, cx| {
        view.update(cx, |v, cx| v.close_detail(win, cx));
    })
    .unwrap();
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
        backend_id: Some("eidola".into()),
        backend_display_name: Some("Eidola".into()),
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
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| RecordView::new(stores.clone(), window, cx))
    });

    let one_page: Vec<_> = (0..51).map(|i| stub_request(&format!("r{i}"), i)).collect();
    let ten_pages: Vec<_> = (0..510)
        .map(|i| stub_request(&format!("r{i}"), i))
        .collect();

    // A fixed visible window (what a ~640px-tall viewport shows at ROW_H).
    let visible = 0..12usize;

    // One page loaded.
    let (one_window, one_total, one_dur) = cx
        .update_window(window, |_, win, cx| {
            view.update(cx, |v, cx| {
                v.set_requests_for_test(one_page.clone(), true);
                v.select_section(RecordSection::Requests, cx);
                let start = std::time::Instant::now();
                let mut n = 0;
                for _ in 0..200 {
                    n = v.render_visible_window_for_test(visible.clone(), win, cx);
                }
                (n, v.display_len_for_test(), start.elapsed())
            })
        })
        .unwrap();

    // Ten pages loaded.
    let (ten_window, ten_total, ten_dur) = cx
        .update_window(window, |_, win, cx| {
            view.update(cx, |v, cx| {
                v.set_requests_for_test(ten_pages.clone(), true);
                let start = std::time::Instant::now();
                let mut n = 0;
                for _ in 0..200 {
                    n = v.render_visible_window_for_test(visible.clone(), win, cx);
                }
                (n, v.display_len_for_test(), start.elapsed())
            })
        })
        .unwrap();

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

/// An open Record window must learn that the local trail grew (codex finding
/// on PR #179: `Change::Record` was dropped by the bus dispatch, so an open
/// window never updated). The bus routes `Change::Record` to the
/// `RecordStore` relay; the window observes it and marks itself stale —
/// surfacing the "new entries — refresh" affordance rather than mutating
/// rows under the user's scroll position. Refresh clears the marker.
#[gpui::test]
fn record_marks_stale_on_record_change_and_refresh_clears(cx: &mut TestAppContext) {
    let stores = stub_stores_with_config(cx);
    let (_window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| RecordView::new(stores.clone(), window, cx))
    });
    cx.run_until_parked();

    view.read_with(cx, |v, _| {
        assert!(!v.stale(), "a fresh window starts un-stale")
    });

    // A durable Record write reaches the window through the bus bridge's
    // dispatch (driven via the deterministic test seam).
    cx.update(|cx| stores::dispatch_change_for_test(&stores, Some(Change::Record), cx));
    cx.run_until_parked();
    view.read_with(cx, |v, _| {
        assert!(
            v.stale(),
            "a Change::Record must mark the open Record window stale"
        );
    });

    // A `Lagged` (refresh-everything) may have dropped a Record change, so it
    // must keep / set the marker too.
    view.update(cx, |v, cx| v.refresh(cx));
    view.read_with(cx, |v, _| assert!(!v.stale(), "refresh clears the marker"));
    cx.update(|cx| stores::dispatch_change_for_test(&stores, None, cx));
    cx.run_until_parked();
    view.read_with(cx, |v, _| {
        assert!(v.stale(), "a Lagged bus signal must also mark stale");
    });
}

/// Replay of the Record stale-fetch race (codex finding on PR #179): a
/// refresh while a page fetch is in flight must *cancel* that fetch, not let
/// it land later and append duplicate/stale rows over the reset listing. The
/// structural fix is task-as-field — the `Listing` owns its fetch task, so
/// `refresh()`'s `Listing::default()` replacement drops (cancels) the
/// in-flight task at the moment of reset. A cancelled task can never run its
/// completion, so the race is impossible by construction; this test asserts
/// the synchronous half (real backend, so fetches genuinely start; the
/// listing resets to empty with exactly the new fetch in flight).
#[gpui::test]
fn record_refresh_supersedes_in_flight_fetch(cx: &mut TestAppContext) {
    // A real `AppCore` over tempdirs (the Record queries are local-DB reads),
    // mirroring `tests/stores.rs::test_core`. `_dir` keeps the tempdir alive.
    // The RecordView's page fetches run as spawned tasks on the AppCore's own
    // tokio runtime (each holding an `Arc<AppCore>`), so declare that
    // cross-thread mixing to the test scheduler — the established idiom (see
    // `tests/stores.rs`), which also lets us join those tasks below without the
    // non-determinism detector flagging their cross-thread completion wakes.
    cx.executor().allow_parking();
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    let _dir = tempfile::tempdir().unwrap();
    let core = std::sync::Arc::new(
        AppCore::new(_dir.path().to_path_buf(), _dir.path().join("data")).expect("open core"),
    );
    core.runtime()
        .block_on(core.set_base_url("https://127.0.0.1:1/v1".into()))
        .unwrap();
    let stores = cx.update(|cx| Stores::for_test(core.clone(), cx));
    let (_window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| RecordView::new(stores.clone(), window, cx))
    });

    // Construction kicked the attestations fetch: in flight, no rows yet.
    view.read_with(cx, |v, _| {
        assert_eq!(
            v.listing_state_for_test(),
            (0, true),
            "construction starts the first page fetch"
        );
    });

    // Refresh mid-flight. Replacing the listing drops the in-flight task —
    // the old fetch is cancelled and can never append — and starts exactly
    // one new fetch over the reset (empty) rows.
    view.update(cx, |v, cx| v.refresh(cx));
    view.read_with(cx, |v, _| {
        assert_eq!(
            v.listing_state_for_test(),
            (0, true),
            "refresh resets the rows and owns a single fresh fetch; the \
             superseded task was cancelled at replacement"
        );
    });

    // Deterministic teardown — join every in-flight bridge task before the test
    // returns. Each page fetch (including the one `refresh` superseded, which is
    // *orphaned* on the tokio side once its gpui receiver is dropped, so no
    // `run_until_parked` ever drains it) holds an `Arc<AppCore>` until it
    // finishes on a tokio worker. The store entities keep the AppCore alive
    // until `cx` teardown on this thread, but if a bridge task is still running
    // then, it becomes the *last* owner and drops the tokio runtime from within
    // its own worker → "Cannot drop a runtime ... from within an asynchronous
    // context" — the intermittent CI panic (Linux-scheduler-sensitive; a local
    // DB read almost always finishes first on macOS, so this never reproduced
    // here). Waiting until the runtime is idle guarantees the last `Arc` always
    // drops on the test thread. This is a precise join on the runtime's live
    // task count, not a timed sleep, and does not touch the assertions above.
    while core.runtime().metrics().num_alive_tasks() > 0 {
        std::thread::yield_now();
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn config_state(has_account: bool) -> ConfigState {
    ConfigState {
        default_template: "00000000-0000-7000-8000-000000000010".into(),
        has_account,
        has_account_secret: has_account,
        account_id: has_account.then(|| "00000000-0000-7000-8000-000000000111".into()),
        account_secret: has_account.then(|| "behavior-account-secret".into()),
        domain_separator: "ACT-v1:eidola:inference:production:2026-03-05".into(),
        appearance: eidola_app_core::config::AppearanceSetting::System,
        time_of_day_tint: eidola_app_core::config::TimeOfDayTint::On,
        light_character: eidola_app_core::config::LightCharacter::Neutral,
        font_scale: 1.0,
        language: None,
    }
}

/// The eidola connection + trust bundle fixture (moved off `ConfigState`);
/// the General pane's base-URL + trust rows read it.
fn eidola_trust() -> eidola_app_core::EidolaTrust {
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

/// A backend registry fixture with the two singletons; `eidola_enabled`
/// flips the eidola row so nav-gating tests can exercise both states.
fn backends_fixture(eidola_enabled: bool) -> Vec<eidola_app_core::BackendInfo> {
    use eidola_app_core::{BackendInfo, BackendKind};
    vec![
        BackendInfo {
            id: "eidola".into(),
            kind: BackendKind::Eidola,
            display_name: "Eidola".into(),
            enabled: eidola_enabled,
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
    ]
}

// ---------------------------------------------------------------------------
// Scroll indicators — render smoke: the overlay `Scrollbar` binds to each
// view's tracked scroll handle. A mis-bound handle (wrong field, wrong type)
// or a panic in `crate::scrollbar::vertical` would surface as a draw panic
// here. The overlay is `ScrollbarShow::Scrolling`, so nothing is asserted
// visible — this proves construction + binding, not appearance.
// ---------------------------------------------------------------------------

/// Force one frame on a test window (mark dirty, then run the scheduled draw),
/// so the view's `render` — and the scroll-indicator overlay it builds — runs.
fn draw_frame(cx: &mut TestAppContext, window: AnyWindowHandle) {
    cx.update_window(window, |_, window, _| window.refresh())
        .unwrap();
    cx.run_until_parked();
}

#[gpui::test]
fn library_renders_with_scroll_indicator(cx: &mut TestAppContext) {
    let stores = stub_stores(cx, |s| {
        s.spaces = vec![
            stub_space("a", Some("Alpha"), None, 2_000),
            stub_space("b", Some("Beta"), None, 1_000),
        ];
    });
    let (window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| LibraryView::new(stores.clone(), window, cx))
    });
    draw_frame(cx, window);
}

#[gpui::test]
fn record_renders_with_scroll_indicator(cx: &mut TestAppContext) {
    // Both body shapes: the virtualized listing (default) and, after opening a
    // detail, the `scroll_wrap` body — each drives a differently-typed handle
    // the overlay must bind.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| RecordView::new(stores.clone(), window, cx))
    });
    draw_frame(cx, window);

    view.update(cx, |v, _cx| {
        v.set_detail_for_test(Some(RecordDetail::Attestation(AttestationDetail {
            hash: "deadbeef".into(),
            pcr_digest: None,
            created_at: 0,
            doc: b"{}".to_vec(),
        })))
    });
    draw_frame(cx, window);
}

#[gpui::test]
fn settings_renders_with_scroll_indicator(cx: &mut TestAppContext) {
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SettingsView::new(stores.clone(), window, cx))
    });
    // Every pane rides the one shared body scroll container + its overlay.
    for pane in [
        SettingsPane::General,
        SettingsPane::Backends,
        SettingsPane::Templates,
        SettingsPane::Account,
        SettingsPane::Wallet,
    ] {
        view.update(cx, |v, cx| v.select(pane, cx));
        draw_frame(cx, window);
    }
}

#[gpui::test]
fn updates_renders_with_scroll_indicator(cx: &mut TestAppContext) {
    let stores = stub_stores(cx, |_| {});
    let (window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| UpdatesView::new(stores.clone(), window, cx))
    });
    draw_frame(cx, window);
}

#[gpui::test]
fn inspector_participants_render_with_scroll_indicator(cx: &mut TestAppContext) {
    // Seed enough eidola-catalog models that the model-picker dropdown
    // overflows its 220px max-height — the nested scroller whose own overlay
    // indicator this exercises (a Codex P1 on PR #232). The draw below opens
    // the add form + picker; a mis-bound picker handle would panic here.
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.eidola_trust = Some(eidola_trust());
        s.space_settings = Some(("demo".into(), eidola_app_core::SpaceSettings::default()));
        s.models = (0..12)
            .map(|i| eidola_app_core::ModelInfo {
                id: format!("model-{i:02}"),
                context_length: 131_072,
                prompt_credits_per_token: 1.0,
                completion_credits_per_token: 2.0,
                request_credits: None,
            })
            .collect();
    });
    let (window, view) = open_space(cx, &stores, Some("demo".into()));
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.set_inspector_open_for_test(true, window, cx));
    })
    .unwrap();
    // The panel body's indicator.
    draw_frame(cx, window);
    // Open the add form + its overflowing model picker; the picker's own
    // overlay indicator binds to `inspector_participant_picker_scroll`.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.inspector_begin_add_participant(window, cx);
            v.inspector_open_add_picker_for_test(cx);
        });
    })
    .unwrap();
    draw_frame(cx, window);
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
        s.eidola_trust = Some(eidola_trust());
        s.balances = Some(BalancesResult {
            available: 5_000_000,
            pools: Vec::new(),
        });
    })
}

/// Open a window whose root is `gpui_component::Root` wrapping the inner
/// view, the same way production does (`lib.rs::open_main_window`). The
/// `Root` wrapper is required by `gpui_component::Input`: a focused input's
/// `on_blur` calls `Root::update`, which panics if the window root isn't a
/// `Root` (SettingsView and the onboarding slides use `Input`; keeping the
/// wrap everywhere mirrors production, where every window root is `Root`).
/// Returns both the `AnyWindowHandle` (for action dispatch /
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

/// Force a paint so element bounds (child scroll bounds, probe rects) populate.
fn draw_window(cx: &mut TestAppContext, window: AnyWindowHandle) {
    cx.update_window(window, |_, window, _| window.refresh())
        .unwrap();
    cx.run_until_parked();
}

// ---------------------------------------------------------------------------
// SpaceView — the tree-navigation conversation surface (wave-6).
// ---------------------------------------------------------------------------

fn open_space(
    cx: &mut TestAppContext,
    stores: &Stores,
    space_id: Option<String>,
) -> (AnyWindowHandle, Entity<SpaceView>) {
    let stores = stores.clone();
    open_view(cx, |window, cx| {
        cx.new(|cx| {
            SpaceView::new(
                stores.clone(),
                space_id.clone(),
                WindowInput::new(cx),
                window,
                cx,
            )
        })
    })
}

/// Set the active draft's composer markdown directly (bypassing IME).
fn set_space_composer_text(
    view: &Entity<SpaceView>,
    window: AnyWindowHandle,
    cx: &mut TestAppContext,
    text: &str,
) {
    let editor = view
        .read_with(cx, |v, _| v.composer_state_for_test())
        .expect("an active draft (composer) is open");
    let text = text.to_string();
    cx.update_window(window, |_, window, cx| {
        let _ = window;
        editor.update(cx, |e, cx| e.set_value(text, cx));
    })
    .unwrap();
}

/// Open a draft (the blank-page composer, or a band reply when `parent` is set).
fn open_space_draft(
    view: &Entity<SpaceView>,
    window: AnyWindowHandle,
    cx: &mut TestAppContext,
    parent: Option<&str>,
) {
    let parent = parent.map(str::to_string);
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.create_draft_for_test(parent.clone(), window, cx)
        });
    })
    .unwrap();
    cx.run_until_parked();
}

/// A scene whose active tail draft **floats**: a tall eight-post transcript,
/// the (empty) draft replying to the last post, a 760×620 window, and the page
/// parked at the top so the draft's slot sits far below the fold. The shared
/// setup for the composer resize-handle tests (the caret / glide tests build
/// the same scene inline with their own draft content).
fn open_floating_composer_scene(
    cx: &mut TestAppContext,
    space_id: &str,
) -> (AnyWindowHandle, Entity<SpaceView>, VisualTestContext) {
    use eidola_app_core::{PostBlock, PostNode, PostParticipant};
    let post = |aid: &str, parent: Option<&str>, user: bool| PostNode {
        action_id: aid.into(),
        item_id: format!("item-{aid}"),
        parent_action_id: parent.map(Into::into),
        participant: PostParticipant {
            kind: if user { "human".into() } else { "agent".into() },
            label: if user { "You".into() } else { "kimi".into() },
        },
        action_type: if user {
            "user_input".into()
        } else {
            "inference".into()
        },
        generation: 0,
        generation_count: 1,
        is_current: true,
        model: None,
        credits_consumed: None,
        relation: parent.map(|_| "reply".to_string()),
        depth: 0,
        is_branch: false,
        blocks: vec![PostBlock {
            id: String::new(),
            block_type: "text".into(),
            text: Some(
                "A few sentences of body text so each post has a realistic \
                 measured height, tall enough that the transcript overflows."
                    .into(),
            ),
            tool_name: None,
            tool_call_id: None,
            data: None,
        }],
        references: Vec::new(),
        created_at: 0,
    };
    let nodes: Vec<PostNode> = (0..8)
        .map(|i| {
            let parent = (i > 0).then(|| format!("a{}", i - 1));
            post(&format!("a{i}"), parent.as_deref(), i % 2 == 0)
        })
        .collect();

    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some(space_id.into()));
    view.update(cx, |v, cx| {
        v.space()
            .update(cx, |s, cx| s.set_post_tree_for_test(nodes, cx));
    });
    cx.run_until_parked();
    open_space_draft(&view, window, cx, Some("a7"));

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(620.)));
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| v.scroll_page_to_top_for_test());
    view.update(&mut vcx, |_, cx| cx.notify());
    vcx.run_until_parked();
    (window, view, vcx)
}

fn dispatch_space_action<A: gpui::Action>(
    view: &Entity<SpaceView>,
    window: AnyWindowHandle,
    cx: &mut TestAppContext,
    action: A,
) {
    let focus = view.read_with(cx, |v, _| v.focus_handle());
    cx.update_window(window, |_, window, cx| {
        focus.dispatch_action(&action, window, cx);
    })
    .unwrap();
    cx.run_until_parked();
}

/// A long single paragraph that soft-wraps into many rows in the composer's
/// **Newsreader** column exercises the display-line vertical navigation
/// end-to-end. Regression for two coupled bugs that only surfaced a few rows
/// into a wrapped paragraph (and only in the real prose font): a float-rounding
/// row-boundary bug in `visual_move_caret` made Down **stall** once the drift
/// between our `row_height` and gpui's internal line spacing crossed a row
/// boundary — and the same drift made Home/End (and the caret render) act on the
/// row *above*. The fix samples each target row's vertical center (`(row+0.5)·h`)
/// so gpui's `(y/line_height) as usize` floor is exact. This must run through the
/// real composer (`VisualTestContext`, Newsreader) — the default test font
/// doesn't reproduce the drift.
#[gpui::test]
fn space_composer_vertical_nav_traverses_every_wrapped_row(cx: &mut TestAppContext) {
    let text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.";
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, None);
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(700.)));
    vcx.run_until_parked();
    let editor = view
        .read_with(&vcx, |v, _| v.composer_state_for_test())
        .expect("composer");
    vcx.update(|_, cx| editor.update(cx, |e, cx| e.set_value(text, cx)));
    vcx.run_until_parked();

    let focus = editor.read_with(&vcx, |e, cx| e.focus_handle(cx));
    let act = |vcx: &mut VisualTestContext, a: &dyn gpui::Action| {
        vcx.update_window(window, |_, window, cx| {
            window.focus(&focus, cx);
            focus.dispatch_action(a, window, cx);
        })
        .unwrap();
        vcx.run_until_parked();
    };
    // The caret's visual row via the (biased, so render-accurate) caret geometry.
    let row_of = |vcx: &mut VisualTestContext| -> i32 {
        editor.read_with(vcx, |e, _| {
            let (top, bot) = e.caret_content_y().expect("laid-out caret");
            let rh = (bot - top).as_f32();
            if rh > 0. {
                (top.as_f32() / rh).round() as i32
            } else {
                -1
            }
        })
    };

    // (1) Down from the top must traverse EVERY wrapped row with no mid-paragraph
    // stall — before the fix it stopped a few rows in.
    act(&mut vcx, &gpui_markdown_editor::DocumentStart);
    let mut visited = std::collections::BTreeSet::new();
    let mut prev = -1;
    for _ in 0..40 {
        let row = row_of(&mut vcx);
        if row == prev {
            break; // no advance — reached the last row
        }
        visited.insert(row);
        prev = row;
        act(&mut vcx, &gpui_markdown_editor::Down);
    }
    let max_row = *visited.iter().max().unwrap();
    assert!(
        max_row >= 6,
        "the paragraph wraps into many rows; Down only reached row {max_row}"
    );
    assert_eq!(
        visited.len() as i32,
        max_row + 1,
        "Down visited every row 0..={max_row} with no stall (visited {})",
        visited.len()
    );

    // (2) On every row, Home and End keep the caret on that same display row —
    // they must not snap to the row above.
    for target in 0..=max_row {
        act(&mut vcx, &gpui_markdown_editor::DocumentStart);
        for _ in 0..target {
            act(&mut vcx, &gpui_markdown_editor::Down);
        }
        assert_eq!(
            row_of(&mut vcx),
            target,
            "Down×{target} lands on row {target}"
        );
        act(&mut vcx, &gpui_markdown_editor::Home);
        assert_eq!(
            row_of(&mut vcx),
            target,
            "Home stays on display row {target}"
        );

        act(&mut vcx, &gpui_markdown_editor::DocumentStart);
        for _ in 0..target {
            act(&mut vcx, &gpui_markdown_editor::Down);
        }
        act(&mut vcx, &gpui_markdown_editor::End);
        assert_eq!(
            row_of(&mut vcx),
            target,
            "End stays on display row {target}"
        );
    }
}

/// The minimap records its container's true top (window-space) so a mousedown's
/// window-y converts to a correct minimap-local y. Regression: the bounds-
/// recording canvas had no explicit inset, so under `absolute` it took its
/// static position — after the full-height `col` — and recorded the container's
/// BOTTOM (≈ window height) as the origin. Every press then computed a negative
/// minimap-local y, which read as a track press below the handle and clamped the
/// page scroll to the very top ("mousedown anywhere jumps to the top").
#[gpui::test]
fn space_minimap_records_container_top_not_bottom(cx: &mut TestAppContext) {
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, None);
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(700.)));
    vcx.run_until_parked();
    // A non-empty draft keeps the minimap live (a selected path to map).
    let editor = view
        .read_with(&vcx, |v, _| v.composer_state_for_test())
        .expect("composer");
    vcx.update(|_, cx| editor.update(cx, |e, cx| e.set_value("A draft.", cx)));
    vcx.run_until_parked();
    let top = view.read_with(&vcx, |v, _| v.minimap_bounds_top_for_test());
    // The space view fills the window, so the minimap container sits at the very
    // top (y ≈ 0) — NOT at the window's bottom edge (700), the pre-fix value.
    assert!(
        top.abs() < 1.0,
        "minimap container top should be ~0 (its real top), was {top} (the \
         bottom-of-rows bug makes every drag jump to the top)"
    );
}

#[gpui::test]
fn space_composer_dock_shadow_is_stable_cold(cx: &mut TestAppContext) {
    // REGRESSION: opening a conversation parks the composer near the bottom with
    // the posts above still *estimated* (unmeasured — the user hasn't scrolled
    // through them). Scrolling up toward the float threshold, each post that
    // crossed into view re-measured estimate→real, shifting the whole document
    // below it; the composer's slot lurched across the dock threshold and its
    // drop shadow flipped on/off — the "two thresholds 20px apart" jump. The
    // warm pass now measures the on-path posts up front, so the document height
    // is stable from a cold open and the float threshold is crossed exactly once
    // as the page scrolls monotonically.
    use eidola_app_core::{PostBlock, PostNode, PostParticipant};
    let post = |aid: &str, parent: Option<&str>, role_user: bool, text: &str| PostNode {
        action_id: aid.into(),
        item_id: format!("item-{aid}"),
        parent_action_id: parent.map(Into::into),
        participant: PostParticipant {
            kind: if role_user {
                "human".into()
            } else {
                "agent".into()
            },
            label: if role_user {
                "You".into()
            } else {
                "kimi".into()
            },
        },
        action_type: if role_user {
            "user_input".into()
        } else {
            "inference".into()
        },
        generation: 0,
        generation_count: 1,
        is_current: true,
        model: None,
        credits_consumed: None,
        relation: parent.map(|_| "reply".to_string()),
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
    };
    let mut nodes = Vec::new();
    for i in 0..12 {
        let aid = format!("a{i}");
        let parent = if i == 0 {
            None
        } else {
            Some(format!("a{}", i - 1))
        };
        nodes.push(post(
            &aid,
            parent.as_deref(),
            i % 2 == 0,
            "A few sentences of body text so each post has a realistic measured height under \
             the prose typography, tall enough that the transcript overflows the window.",
        ));
    }

    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("warm".into()));
    view.update(cx, |v, cx| {
        v.space()
            .update(cx, |s, cx| s.set_post_tree_for_test(nodes, cx));
    });
    cx.run_until_parked();
    // Activate the tail composer (reply to the last post a11).
    open_space_draft(&view, window, cx, Some("a11"));

    let vcx = VisualTestContext::from_window(window, cx);
    // Cold open at a window far smaller than the document: only a few posts are
    // ever on screen. Do NOT scroll — just let the warm pass run.
    vcx.simulate_resize(gpui::size(px(760.), px(620.)));
    vcx.run_until_parked();

    // The warm pass renders every on-path post real for a few frames, so all 12
    // measure into the cache up front — even the off-screen ones. Without it,
    // only the handful on screen would be measured, and the rest would lurch the
    // document (and the dock shadow) as they measured during a later scroll.
    let (measured, total) = view.read_with(&vcx, |v, _| {
        (v.measured_post_count_for_test(), v.post_count_for_test())
    });
    assert_eq!(total, 12);
    assert_eq!(
        measured, total,
        "cold open should warm every on-path post into the height cache \
         (measured {measured} of {total}); unmeasured posts shift the layout when scrolled into view"
    );
}

#[gpui::test]
fn space_resize_above_column_cap_does_not_churn_height_cache(cx: &mut TestAppContext) {
    // REGRESSION (resize jitter): a post's measured height depends only on the
    // reading column (`body_width`), which is capped at `BODY_MAX_WIDTH` above a
    // ~836px window. So resizing a *wide* window does not reflow any post — yet
    // the height cache used to be keyed on the raw window width, so every resize
    // cleared it, dropped every post back to a rough estimate, and jittered the
    // page (and minimap) as the near-viewport posts re-measured estimate→real.
    // The cache is now keyed on `body_width`, so a resize that leaves the column
    // unchanged does not invalidate it at all.
    use eidola_app_core::{PostBlock, PostNode, PostParticipant};
    let post = |aid: &str, parent: Option<&str>, text: &str| PostNode {
        action_id: aid.into(),
        item_id: format!("item-{aid}"),
        parent_action_id: parent.map(Into::into),
        participant: PostParticipant {
            kind: "human".into(),
            label: "You".into(),
        },
        action_type: "user_input".into(),
        generation: 0,
        generation_count: 1,
        is_current: true,
        model: None,
        credits_consumed: None,
        relation: parent.map(|_| "reply".to_string()),
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
    };
    let mut nodes = Vec::new();
    for i in 0..12 {
        let aid = format!("a{i}");
        let parent = if i == 0 {
            None
        } else {
            Some(format!("a{}", i - 1))
        };
        nodes.push(post(
            &aid,
            parent.as_deref(),
            "A few sentences of body text so each post has a realistic height, tall enough \
             that the transcript overflows the window several times over.",
        ));
    }
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("warm".into()));
    view.update(cx, |v, cx| {
        v.space()
            .update(cx, |s, cx| s.set_post_tree_for_test(nodes, cx));
    });
    cx.run_until_parked();

    let vcx = VisualTestContext::from_window(window, cx);
    // Settle at 1200px (well above the ~836px column cap → body_width == 600).
    vcx.simulate_resize(gpui::size(px(1200.), px(560.)));
    vcx.run_until_parked();
    let clears_start = view.read_with(&vcx, |v, _| v.layout_clears_for_test());

    // Resize among several widths, all above the cap: the column never changes,
    // so the cache must never be invalidated — no churn, no jitter.
    for w in [1100., 1000., 950., 1400.] {
        vcx.simulate_resize(gpui::size(px(w), px(560.)));
        vcx.run_until_parked();
    }
    let clears_wide = view.read_with(&vcx, |v, _| v.layout_clears_for_test());
    assert_eq!(
        clears_wide, clears_start,
        "resizing above the reading-column cap must not invalidate the height cache \
         (started {clears_start}, ended {clears_wide}); a clear is what jittered the page"
    );

    // Sanity: a resize *below* the cap genuinely changes the column, so the cache
    // must invalidate — proving the assertion above can actually observe a clear.
    vcx.simulate_resize(gpui::size(px(600.), px(560.)));
    vcx.run_until_parked();
    let clears_narrow = view.read_with(&vcx, |v, _| v.layout_clears_for_test());
    assert!(
        clears_narrow > clears_wide,
        "resizing below the column cap should invalidate the cache (the column really \
         did change): clears {clears_wide} -> {clears_narrow}"
    );
}

#[gpui::test]
fn space_blank_composer_does_not_scroll(cx: &mut TestAppContext) {
    // REGRESSION: a brand-new space (just the composer) reserved a phantom
    // scroll range equal to the titlebar reserve. The document is laid out
    // beneath a top reserve that holds whatever leads it clear of the overlaid
    // titlebar, and the composer's slot used to claim a *whole* window under
    // it, so the total came to window + reserve.
    //
    // The reserve itself is now unconditional (task 40 — a composer-only
    // notebook needs the same headroom its first post will get, or posting
    // moves the words; see `space_composer_text_sits_where_its_post_will_land`).
    // What keeps this invariant is `standalone_slot_h`: a slot that stands
    // alone claims one window *below* the reserve, so a sole composer plus the
    // reserve is exactly one window — no scroll until the content overflows.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, None);
    view.read_with(cx, |v, _| assert!(v.has_active_draft_for_test()));

    let vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(560.)));
    vcx.run_until_parked();

    let min_y = view.read_with(&vcx, |v, _| v.scroll_min_y_for_test());
    assert!(
        min_y > -1.0,
        "a blank space's sole composer should not reserve scroll (min_y = {min_y}, \
         expected ~0; the pre-fix bug parked it at -titlebar_reserve)"
    );
}

#[gpui::test]
fn a_populated_composer_has_full_window_docking_runway(cx: &mut TestAppContext) {
    const WINDOW_H: f32 = 560.0;
    let stores = stub_stores_with_config(cx);
    let (blank_window, blank) = open_space(cx, &stores, None);
    let reserve = blank.read_with(cx, |v, _| v.doc_reserve_for_test());
    blank.read_with(cx, |v, _| {
        assert_eq!(
            v.runway_height_for_test(WINDOW_H),
            WINDOW_H - reserve,
            "a blank notebook keeps its titlebar-adjusted no-scroll slot"
        );
    });
    cx.update_window(blank_window, |_, _, _| {}).unwrap();

    let (window, populated) = open_space(cx, &stores, Some("runway".into()));
    populated.update(cx, |v, cx| {
        v.space().update(cx, |space, cx| {
            space.set_post_tree_for_test(
                vec![fixture_post_with_block("a1", "b1", "A settled post.")],
                cx,
            )
        });
    });
    cx.run_until_parked();
    populated.read_with(cx, |v, _| {
        let runway = v.runway_height_for_test(WINDOW_H);
        let chrome = v.composer_chrome_for_test();
        assert_eq!(
            runway, WINDOW_H,
            "a populated branch's trailing slot claims exactly one window — no more, no less"
        );
        assert_eq!(
            WINDOW_H - runway,
            0.0,
            "at the document floor the slot top — which is also the docked surface's \
             top edge — lands exactly at the window top, the previous separator band \
             just cleared above it"
        );
        assert_eq!(
            chrome, 40.0,
            "the bar's top chrome is the full post pad, so the docked editor's text \
             sits POST_PAD_Y below the slot top exactly as a post's body does"
        );
    });
    cx.update_window(window, |_, _, _| {}).unwrap();
}

#[gpui::test]
fn an_inactive_tail_draft_does_not_borrow_an_off_branch_composers_height(cx: &mut TestAppContext) {
    const WINDOW_H: f32 = 560.0;
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("draft-heights".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());
    let mut left = fixture_assistant_post("a2", "left branch");
    left.parent_action_id = Some("a1".into());
    let mut right = fixture_assistant_post("a3", "right branch");
    right.parent_action_id = Some("a1".into());
    space.update(cx, |s, cx| {
        s.set_post_tree_for_test(vec![fixture_user_post("a1", "root"), left, right], cx)
    });

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(WINDOW_H)));
    vcx.run_until_parked();
    let parents = view.read_with(&vcx, |v, _| v.draft_parents_for_test());
    let left_index = parents
        .iter()
        .position(|parent| parent.as_deref() == Some("a2"))
        .expect("left branch tail draft");
    let right_index = parents
        .iter()
        .position(|parent| parent.as_deref() == Some("a3"))
        .expect("right branch tail draft");

    view.update(&mut vcx, |v, cx| v.activate_draft_for_test(right_index, cx));
    vcx.run_until_parked();
    let left_before = view
        .read_with(&vcx, |v, _| {
            v.inactive_draft_height_for_test(left_index, WINDOW_H)
        })
        .expect("left draft remains inactive");

    let editor = view
        .read_with(&vcx, |v, _| v.composer_state_for_test())
        .expect("right draft is active");
    let long = (0..80)
        .map(|i| format!("Paragraph {i} fills the off-branch composer with its own content."))
        .collect::<Vec<_>>()
        .join("\n\n");
    editor.update(&mut vcx, |editor, cx| editor.set_value(long, cx));
    vcx.run_until_parked();

    let (left_after, active_runway) = view.read_with(&vcx, |v, _| {
        (
            v.inactive_draft_height_for_test(left_index, WINDOW_H)
                .expect("left draft remains inactive"),
            v.runway_height_for_test(WINDOW_H),
        )
    });
    assert!(
        active_runway > left_after + 1.0,
        "the long active composer must exceed the inactive draft's own slot \
         ({active_runway} vs {left_after})"
    );
    assert!(
        (left_before - left_after).abs() < 0.5,
        "off-branch composer growth must not inflate an inactive tail draft \
         ({left_before} -> {left_after})"
    );
}

#[gpui::test]
fn space_composer_text_sits_where_its_post_will_land(cx: &mut TestAppContext) {
    // REGRESSION (task 40): the composer *becomes* the post, so the words in it
    // must already sit where the post will render them — submitting must not
    // move the text the user just typed.
    //
    // It moved by exactly `TITLE_BAR_RESERVE`: the document's top reserve
    // existed only once there were posts, so a composer-only notebook laid its
    // draft out a reserve higher than the post it was about to become, and the
    // whole document dropped at the submit moment.
    //
    // Both positions are read off **painted line geometry** (the editor's own
    // `debug_line_geometry`, in window coordinates), not recomputed from the
    // padding constants the fix touches — so the test measures the same thing a
    // scanline diff of the two renders would.
    const ASK: &str = "Why is the sky blue?";
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, None);
    view.read_with(cx, |v, _| {
        assert!(
            v.has_active_draft_for_test(),
            "a blank ⌘N space opens with its composer"
        )
    });

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(900.), px(760.)));
    vcx.run_until_parked();

    // The fresh, composer-only window: type the ask.
    let draft_editor = view
        .read_with(&vcx, |v, _| v.composer_state_for_test())
        .expect("the blank space's composer");
    vcx.update(|_, cx| {
        draft_editor.update(cx, |e, cx| e.set_value(ASK.to_string(), cx));
    });
    vcx.run_until_parked();
    vcx.update(|window, _| window.refresh());
    vcx.run_until_parked();

    let draft_top = view
        .read_with(&vcx, |_, cx| first_painted_line_top(&draft_editor, cx))
        .expect("the draft's first line is painted");
    assert!(
        !view.read_with(&vcx, |v, _| v.composer_overlayed_for_test()),
        "a composer-only notebook docks into its slot — a floating bar would be \
         measuring the transient state, not the one the post replaces"
    );

    // The submit moment: the words become a post and the draft is consumed
    // (the space's own tail composer comes back empty beneath it).
    vcx.update(|_, cx| {
        draft_editor.update(cx, |e, cx| e.set_value(String::new(), cx));
        view.update(cx, |v, cx| {
            v.space().update(cx, |s, cx| {
                s.set_post_tree_for_test(vec![fixture_user_post("a1", ASK)], cx)
            });
        });
    });
    vcx.run_until_parked();
    vcx.update(|_, cx| view.update(cx, |v, _| v.scroll_page_to_top_for_test()));
    vcx.update(|window, _| window.refresh());
    vcx.run_until_parked();

    let post_editor = view
        .read_with(&vcx, |v, _| v.post_body_editor_for_test("a1"))
        .expect("the posted ask's body editor");
    let post_top = view
        .read_with(&vcx, |_, cx| first_painted_line_top(&post_editor, cx))
        .expect("the post's first line is painted");

    assert!(
        (draft_top - post_top).abs() < 0.5,
        "the draft's first line must land exactly where the post's does — \
         draft {draft_top}, post {post_top} (a {}px jump at submit)",
        (post_top - draft_top).abs()
    );
}

#[gpui::test]
fn space_reply_draft_text_sits_where_its_post_will_land(cx: &mut TestAppContext) {
    // The same continuity, one post in: a draft replying to an existing post
    // lays its text out where that reply will render. This one held before task
    // 40 (a space with posts always had its top reserve) — it is here because
    // the *rule* is what matters: a docked composer's editor starts
    // `POST_PAD_Y` below its slot top, exactly where a post's body starts below
    // its own, so wherever the slot sits the words don't move when they're
    // posted. Without it, only the blank-notebook case would be pinned and the
    // shared rule could drift for every other draft.
    const ASK: &str = "Why is the sky blue?";
    const REPLY: &str = "And at sunset?";
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    view.update(cx, |v, cx| {
        v.space().update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("a1", ASK)], cx)
        });
    });
    cx.run_until_parked();

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(900.), px(760.)));
    vcx.run_until_parked();

    // The tail draft under that post, with the follow-up typed into it.
    vcx.update(|window, cx| {
        view.update(cx, |v, cx| {
            v.create_draft_for_test(Some("a1".into()), window, cx)
        });
    });
    vcx.run_until_parked();
    let draft_editor = view
        .read_with(&vcx, |v, _| v.composer_state_for_test())
        .expect("the tail draft is the active composer");
    vcx.update(|_, cx| {
        draft_editor.update(cx, |e, cx| e.set_value(REPLY.to_string(), cx));
    });
    vcx.run_until_parked();
    vcx.update(|_, cx| view.update(cx, |v, _| v.scroll_page_to_top_for_test()));
    vcx.update(|window, _| window.refresh());
    vcx.run_until_parked();

    assert!(
        !view.read_with(&vcx, |v, _| v.composer_overlayed_for_test()),
        "the draft's slot is well above the float line here, so it docks — a \
         floating bar is the transient state, not the one the post replaces"
    );
    let draft_top = view
        .read_with(&vcx, |_, cx| first_painted_line_top(&draft_editor, cx))
        .expect("the draft's first line is painted");

    // Posted: the reply takes the slot the draft occupied.
    let mut a2 = fixture_user_post("a2", REPLY);
    a2.parent_action_id = Some("a1".into());
    vcx.update(|_, cx| {
        draft_editor.update(cx, |e, cx| e.set_value(String::new(), cx));
        view.update(cx, |v, cx| {
            v.space().update(cx, |s, cx| {
                s.set_post_tree_for_test(vec![fixture_user_post("a1", ASK), a2.clone()], cx)
            });
        });
    });
    vcx.run_until_parked();
    vcx.update(|_, cx| view.update(cx, |v, _| v.scroll_page_to_top_for_test()));
    vcx.update(|window, _| window.refresh());
    vcx.run_until_parked();

    let post_editor = view
        .read_with(&vcx, |v, _| v.post_body_editor_for_test("a2"))
        .expect("the posted reply's body editor");
    let post_top = view
        .read_with(&vcx, |_, cx| first_painted_line_top(&post_editor, cx))
        .expect("the reply's first line is painted");

    assert!(
        (draft_top - post_top).abs() < 0.5,
        "a reply draft's first line must land exactly where the reply's does — \
         draft {draft_top}, post {post_top}"
    );
}

/// The window-space `y` of the first line an editor actually painted — the
/// geometric stand-in for "first ink", read off the editor's own layout rather
/// than recomputed from layout constants.
fn first_painted_line_top(
    editor: &Entity<gpui_markdown_editor::MarkdownEditorState>,
    cx: &gpui::App,
) -> Option<f32> {
    editor
        .read(cx)
        .debug_line_geometry()
        .first()
        .and_then(|(_, lines)| lines.first().map(|(_, y, _)| *y))
}

#[gpui::test]
fn space_blank_opens_with_composer_existing_does_not(cx: &mut TestAppContext) {
    // A brand-new (blank ⌘N) space opens with the composer ready.
    let stores = stub_stores_with_config(cx);
    let (_w, blank) = open_space(cx, &stores, None);
    blank.read_with(cx, |v, _| {
        assert!(
            v.has_active_draft_for_test(),
            "a blank space opens with the composer"
        );
        assert_eq!(v.active_draft_parent_for_test(), None, "root draft");
    });

    // A reopened space with history opens WITHOUT a composer (click "+" to start).
    let (_w2, existing) = open_space(cx, &stores, Some("has-history".into()));
    existing.read_with(cx, |v, _| {
        assert!(
            !v.has_active_draft_for_test(),
            "an existing space opens with no composer"
        );
    });
}

#[gpui::test]
fn space_submit_appends_user_streams_and_consumes_draft(cx: &mut TestAppContext) {
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, None);
    set_space_composer_text(&view, window, cx, "hello space");
    dispatch_space_action(&view, window, cx, Send);

    view.read_with(cx, |v, cx| {
        // The snapshot picked up the optimistic user turn, the space is
        // streaming (the stub leaves `streaming = Some`), and the draft was
        // consumed (it's now a persisted post).
        assert_eq!(v.post_count_for_test(), 1);
        assert_eq!(v.draft_count_for_test(), 0, "submit consumes the draft");
        assert!(!v.has_active_draft_for_test());
        let space = v.space().read(cx);
        assert!(space.is_streaming(), "submit enters the streaming state");
        assert_eq!(space.messages()[0].message.role, "user");
        assert_eq!(space.messages()[0].message.content, "hello space");
    });
}

#[gpui::test]
fn space_post_during_save_window_preserves_draft(cx: &mut TestAppContext) {
    // A Post landing while a prior Post's save/plan is still in flight (the
    // `post_runner` is busy but nothing is streaming yet) must be rejected with
    // the draft left intact and active — never consumed-then-dropped. The
    // composer consumes the draft only after the space *accepts* the submit.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, None);
    let space = view.read_with(cx, |v, _| v.space().clone());

    // Simulate the in-flight save window: the exclusive mutation slot is
    // occupied, but no turn is streaming (the exact window the old
    // `is_streaming`-only guard let a post through in).
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| s.arm_post_runner_for_test(cx));
    })
    .unwrap();
    view.read_with(cx, |v, cx| {
        assert!(
            !v.space().read(cx).is_streaming(),
            "the save window is not streaming"
        );
    });

    // Type into the active draft and try to Post it during that window.
    set_space_composer_text(&view, window, cx, "don't lose me");
    dispatch_space_action(&view, window, cx, Send);

    // The submit was rejected (space busy) — the draft survives, still active,
    // still carrying its content.
    view.read_with(cx, |v, cx| {
        assert_eq!(
            v.post_count_for_test(),
            0,
            "no optimistic post — the submit was rejected"
        );
        assert!(
            v.has_active_draft_for_test(),
            "the rejected draft stays active"
        );
        let editor = v
            .composer_state_for_test()
            .expect("the draft is still the active composer");
        assert_eq!(
            editor.read(cx).value().trim(),
            "don't lose me",
            "the typed content survives a rejected Post"
        );
    });
}

#[gpui::test]
fn space_post_only_appends_user_without_streaming(cx: &mut TestAppContext) {
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, None);
    set_space_composer_text(&view, window, cx, "just save this");
    dispatch_space_action(&view, window, cx, PostOnly);

    view.read_with(cx, |v, cx| {
        assert_eq!(v.post_count_for_test(), 1);
        let space = v.space().read(cx);
        assert!(!space.is_streaming(), "post-only does not stream");
        assert_eq!(space.messages()[0].message.content, "just save this");
    });
}

#[gpui::test]
fn space_empty_submit_is_a_noop(cx: &mut TestAppContext) {
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, None);
    // Composer left empty.
    dispatch_space_action(&view, window, cx, Send);
    view.read_with(cx, |v, cx| {
        assert_eq!(v.post_count_for_test(), 0);
        assert!(!v.space().read(cx).is_streaming());
    });
}

#[gpui::test]
fn space_reply_branches_at_target_and_clears_on_submit(cx: &mut TestAppContext) {
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, None);

    // Seed a persisted post tree so there's a real post to reply to.
    let space = view.read_with(cx, |v, _| v.space().clone());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("a1", "the root post")], cx)
        });
    })
    .unwrap();
    cx.run_until_parked();

    // The "+" on a1's band opens a draft replying to a1 (the prior empty root
    // draft is retired). The active draft's parent is a1.
    open_space_draft(&view, window, cx, Some("a1"));
    view.read_with(cx, |v, _| {
        assert_eq!(v.active_draft_parent_for_test().as_deref(), Some("a1"));
    });

    // Submitting consumes the draft.
    set_space_composer_text(&view, window, cx, "a branch reply");
    dispatch_space_action(&view, window, cx, Send);
    view.read_with(cx, |v, _| {
        assert!(!v.has_active_draft_for_test(), "draft consumed on submit");
    });
}

#[gpui::test]
fn space_post_in_a_new_branch_stays_on_that_branch(cx: &mut TestAppContext) {
    // Reply on a post that already has a committed reply → a fork draft on a
    // *second* branch, which `pending_select` brings onto the selected path.
    // Posting it must keep that branch selected. It used to snap back to the
    // first branch: the draft was consumed a moment before its post existed,
    // leaving the parent's strip with a single page, whose scroll offset gpui
    // then clamped to 0 — and the reload landed on the (now) first branch.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());

    // a1 (root) with one committed reply a2 — so a1's band offers Reply.
    let mut a2 = fixture_assistant_post("a2", "the first branch");
    a2.parent_action_id = Some("a1".into());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("a1", "the root post"), a2], cx)
        });
    })
    .unwrap();

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(560.)));
    vcx.run_until_parked();

    // Fork a new branch off a1 and post into it.
    vcx.update(|window, cx| {
        view.update(cx, |v, cx| {
            v.create_draft_for_test(Some("a1".into()), window, cx)
        });
    });
    vcx.run_until_parked();
    let editor = view
        .read_with(&vcx, |v, _| v.composer_state_for_test())
        .expect("the fork draft is the active composer");
    editor.update(&mut vcx, |e, cx| e.set_value("a second branch", cx));
    let focus = view.read_with(&vcx, |v, _| v.focus_handle());
    vcx.update(|window, cx| focus.dispatch_action(&Send, window, cx));
    vcx.run_until_parked();

    // The persist lands: a3 is a1's *second* child (the branch just posted
    // into). The view must still be on it.
    let mut a2b = fixture_assistant_post("a2", "the first branch");
    a2b.parent_action_id = Some("a1".into());
    let mut a3 = fixture_user_post("a3", "a second branch");
    a3.parent_action_id = Some("a1".into());
    space.update(&mut vcx, |s, cx| {
        s.set_post_tree_for_test(vec![fixture_user_post("a1", "the root post"), a2b, a3], cx)
    });
    vcx.run_until_parked();

    let leaf = vcx.update(|window, cx| view.read(cx).selected_leaf_for_test(window));
    assert_eq!(
        leaf.as_deref(),
        Some("a3"),
        "posting into a new branch stays on that branch, not the first one"
    );
}

#[gpui::test]
fn space_draft_rethreads_onto_an_edited_parents_current_tip(cx: &mut TestAppContext) {
    // A draft can outlive an edit of the very post it replies to — another
    // window on the same space commits a new generation, and the shared
    // entity's reloaded transcript then carries only that item's **current
    // tip**. The draft still names the superseded action, which is in no
    // current post, so the reply antecedent was silently dropped: the post
    // landed at the space tail (durably!) instead of beside its sibling on the
    // parent's branch. Reply threading follows *item* identity — so a draft
    // whose parent was superseded must rethread onto that item's tip.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());

    // a1 (root) with one committed reply a2 — so a1's band offers Reply.
    let mut a2 = fixture_assistant_post("a2", "the first branch");
    a2.parent_action_id = Some("a1".into());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("a1", "the root post"), a2], cx)
        });
    })
    .unwrap();

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(560.)));
    vcx.run_until_parked();

    // Fork a new branch off a1 and type into it (an unsent, non-empty draft).
    vcx.update(|window, cx| {
        view.update(cx, |v, cx| {
            v.create_draft_for_test(Some("a1".into()), window, cx)
        });
    });
    vcx.run_until_parked();
    let editor = view
        .read_with(&vcx, |v, _| v.composer_state_for_test())
        .expect("the fork draft is the active composer");
    editor.update(&mut vcx, |e, cx| e.set_value("a second branch", cx));
    vcx.run_until_parked();

    // Another window edits a1 while this draft is open: a new generation
    // (`a1b`, same item) supersedes it and every reply rethreads under the tip.
    let mut a1b = fixture_user_post("a1b", "the root post, edited");
    a1b.item_id = "item-a1".into();
    a1b.generation = 1;
    a1b.generation_count = 2;
    let mut a2b = fixture_assistant_post("a2", "the first branch");
    a2b.parent_action_id = Some("a1b".into());
    space.update(&mut vcx, |s, cx| {
        s.set_post_tree_for_test(vec![a1b, a2b], cx)
    });
    vcx.run_until_parked();

    // Post the draft.
    let focus = view.read_with(&vcx, |v, _| v.focus_handle());
    vcx.update(|window, cx| focus.dispatch_action(&Send, window, cx));
    vcx.run_until_parked();

    // The optimistic row's parent is the antecedent `Space::submit` received —
    // and therefore the one the durable post links to. It must be the item's
    // current tip, not `None` (the space tail) and not the superseded id.
    let parent = space.read_with(&vcx, |s, _| {
        s.messages()
            .last()
            .expect("the optimistic user turn was appended")
            .parent_action_id
            .clone()
    });
    assert_eq!(
        parent.as_deref(),
        Some("a1b"),
        "a draft whose parent was edited elsewhere replies to that item's \
         current tip (got {parent:?})"
    );
}

#[gpui::test]
fn space_auto_tail_draft_at_each_leaf(cx: &mut TestAppContext) {
    // Every branch leaf gets an always-present, *docked* tail draft (the
    // composer that replaces the leaf "+"); non-leaves do not.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());
    let mut a2 = fixture_user_post("a2", "a committed reply");
    a2.parent_action_id = Some("a1".into());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("a1", "root"), a2], cx)
        });
    })
    .unwrap();
    cx.run_until_parked();

    view.read_with(cx, |v, _| {
        let parents = v.draft_parents_for_test();
        assert!(
            parents.contains(&Some("a2".to_string())),
            "a tail draft sits at the leaf a2; parents = {parents:?}"
        );
        assert!(
            !parents.contains(&Some("a1".to_string())),
            "no tail draft at the non-leaf a1; parents = {parents:?}"
        );
        assert!(
            !v.has_active_draft_for_test(),
            "tail drafts are docked, not active"
        );
    });
}

#[gpui::test]
fn space_escape_keeps_empty_tail_draft(cx: &mut TestAppContext) {
    // The blank-page root draft is a *tail* draft (the always-present
    // end-of-branch composer), so Escape just docks it — it is NOT deleted.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, None);
    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| v.deactivate_for_test(cx));
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert_eq!(v.draft_count_for_test(), 1, "an empty tail draft persists");
        assert!(
            !v.has_active_draft_for_test(),
            "but is deactivated (docked)"
        );
    });
}

#[gpui::test]
fn space_escape_deletes_empty_fork_keeps_nonempty(cx: &mut TestAppContext) {
    // An existing space (no auto root draft). Seed a1 → a2 (a2 is a committed
    // reply to a1), so a draft on a1 is a *fork* (a1 already has a reply).
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());
    let mut a2 = fixture_user_post("a2", "a committed reply");
    a2.parent_action_id = Some("a1".into());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("a1", "root"), a2], cx)
        });
    })
    .unwrap();
    cx.run_until_parked();

    // An empty FORK draft on a1 → Escape deletes it (a transient new branch).
    // (The auto tail draft on the leaf a2 persists — assert by parent, not by
    // raw count.)
    open_space_draft(&view, window, cx, Some("a1"));
    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| v.deactivate_for_test(cx));
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        let parents = v.draft_parents_for_test();
        assert!(
            !parents.contains(&Some("a1".to_string())),
            "the empty fork on a1 was deleted; parents = {parents:?}"
        );
        assert!(!v.has_active_draft_for_test());
    });

    // A non-empty draft on a1 persists (deselected) when Escaped.
    open_space_draft(&view, window, cx, Some("a1"));
    set_space_composer_text(&view, window, cx, "keep me");
    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| v.deactivate_for_test(cx));
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        let parents = v.draft_parents_for_test();
        assert!(
            parents.contains(&Some("a1".to_string())),
            "a non-empty fork persists; parents = {parents:?}"
        );
        assert!(!v.has_active_draft_for_test(), "but is deactivated");
    });
}

#[gpui::test]
fn space_joins_shared_entity_for_same_id(cx: &mut TestAppContext) {
    // Two windows on one space share the same `Space` entity (the registry
    // join), so a submit in one appears in the other.
    let stores = stub_stores_with_config(cx);
    let (w1, v1) = open_space(cx, &stores, Some("shared".into()));
    let (_w2, v2) = open_space(cx, &stores, Some("shared".into()));
    let e1 = v1.read_with(cx, |v, _| v.space().clone());
    let e2 = v2.read_with(cx, |v, _| v.space().clone());
    assert_eq!(e1.entity_id(), e2.entity_id());

    // An existing space opens with no composer; open one, then submit.
    open_space_draft(&v1, w1, cx, None);
    set_space_composer_text(&v1, w1, cx, "from window one");
    dispatch_space_action(&v1, w1, cx, Send);
    // The second window's snapshot reflects the shared space.
    v2.read_with(cx, |v, _| assert_eq!(v.post_count_for_test(), 1));
}

#[gpui::test]
fn space_window_is_named_after_its_space(cx: &mut TestAppContext) {
    // The window's accessible name (and its Window-menu entry) tracks the
    // Library title: a blank ⌘N space is "New Space", a titled one carries its
    // title, and a rename re-titles the window. `TestWindow` doesn't implement
    // `get_title`, so the view's own record is the observable end of the call.
    //
    // This asserts the *values*, not the ordering. The ordering half — that
    // the title is written by the `stores.spaces` observer, ahead of the frame
    // its notify schedules, rather than from inside `render` — is **not
    // testable at this pin**: `App::flush_effects` draws every dirty window
    // inside the same update under `test-support`, so both placements look
    // identical from here, and the thing that would actually distinguish them
    // (the AccessKit root label built in `a11y.begin_frame`) is crate-private.
    // See `sync_window_title` for why the placement still matters in
    // production.
    let stores = stub_stores(cx, |s| {
        s.spaces = vec![stub_space("s1", Some("Tides and the moon"), None, 0)];
    });

    let (_blank_w, blank) = open_space(cx, &stores, None);
    blank.read_with(cx, |v, _| {
        assert_eq!(v.window_title_for_test(), Some("New Space"));
    });

    let (_w, view) = open_space(cx, &stores, Some("s1".into()));
    view.read_with(cx, |v, _| {
        assert_eq!(v.window_title_for_test(), Some("Tides and the moon"));
    });

    stores.spaces.update(cx, |s, cx| {
        s.rename("s1".into(), "The moon and tides".into(), cx)
    });
    view.read_with(cx, |v, _| {
        assert_eq!(v.window_title_for_test(), Some("The moon and tides"));
    });
}

// ---------------------------------------------------------------------------
// Separators — the band's Reply-or-Ask menu, and explicit asks (wave 3b).
//
// The composer's model picker + request panel are gone: who answers (and with
// what model) is Participants configuration, and explicit asks live on the
// separator bands. These tests drive the view's `set_band_menu_for_test` /
// `ask_participant` seams directly (the "+" click is a probe-tested affordance).
// ---------------------------------------------------------------------------

/// A stub space-owned agent participant (the separator Ask menus + cascade
/// notice read the space's agent set from `ParticipantsStore`).
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

/// Stub stores with two agent participants seeded for `space_id`.
fn stub_stores_with_agents(cx: &mut TestAppContext, space_id: &str) -> Stores {
    let sid = space_id.to_string();
    stub_stores(cx, move |s| {
        s.config_state = Some(config_state(true));
        s.eidola_trust = Some(eidola_trust());
        s.participants = Some((
            sid,
            vec![
                agent_participant("agent-b", "Ida"),
                agent_participant("agent-c", "Sage"),
            ],
        ));
    })
}

#[gpui::test]
fn space_band_ask_targets_participant_and_closes_menu(cx: &mut TestAppContext) {
    // Choosing "Ask <agent>" in a band's menu starts a streaming turn from
    // that participant targeting the band's post, and closes the menu.
    let stores = stub_stores_with_agents(cx, "s");
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());
    let mut a2 = fixture_assistant_post("a2", "an answer");
    a2.parent_action_id = Some("a1".into());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("a1", "root"), a2], cx)
        });
    })
    .unwrap();
    cx.run_until_parked();

    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| v.set_band_menu_for_test(Some("a1"), cx));
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert_eq!(v.band_menu_for_test().as_deref(), Some("a1"));
    });

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.ask_participant("agent-b".into(), "a1".into(), window, cx)
        });
    })
    .unwrap();
    view.read_with(cx, |v, cx| {
        assert!(v.band_menu_for_test().is_none(), "a choice closes the menu");
        let space = v.space().read(cx);
        let streams = space.streams();
        assert_eq!(streams.len(), 1, "the ask started one streaming turn");
        assert_eq!(streams[0].participant_id.as_deref(), Some("agent-b"));
        assert_eq!(streams[0].target_action_id.as_deref(), Some("a1"));
    });
}

#[gpui::test]
fn space_band_menu_closes_when_draft_deactivates(cx: &mut TestAppContext) {
    // The band menu belongs to the current interaction; retiring the draft
    // (Escape) must take it with it rather than leaving it floating.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, None);
    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| v.set_band_menu_for_test(Some("a1"), cx));
    })
    .unwrap();
    view.read_with(cx, |v, _| assert!(v.band_menu_for_test().is_some()));

    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| v.deactivate_for_test(cx));
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert!(
            v.band_menu_for_test().is_none(),
            "deactivating the draft closes the band menu"
        );
        assert!(!v.has_active_draft_for_test());
    });
}

#[gpui::test]
fn space_tail_ask_discards_empty_draft(cx: &mut TestAppContext) {
    // The tail-draft rule, empty half: asking at the end of a branch discards
    // an *empty* tail draft — the UI tracks the incoming response as the new
    // tail (`sync_tail_drafts` docks a fresh composer under it once it lands).
    let stores = stub_stores_with_agents(cx, "s");
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("a1", "the only post")], cx)
        });
    })
    .unwrap();
    cx.update_window(window, |_, window, _| window.refresh())
        .unwrap();
    cx.run_until_parked();

    // The leaf a1 grew its docked (empty) tail draft.
    view.read_with(cx, |v, _| {
        assert!(v.draft_parents_for_test().contains(&Some("a1".to_string())));
    });

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.ask_participant("agent-b".into(), "a1".into(), window, cx)
        });
    })
    .unwrap();
    view.read_with(cx, |v, cx| {
        assert!(
            !v.draft_parents_for_test().contains(&Some("a1".to_string())),
            "the empty tail draft is discarded — the response becomes the tail"
        );
        assert!(!v.has_active_draft_for_test());
        assert_eq!(v.space().read(cx).streams().len(), 1);
    });
}

#[gpui::test]
fn space_ask_selects_the_new_branch_the_reply_will_stream_in(cx: &mut TestAppContext) {
    // Asking about a post that has *already* been replied to forks a new
    // branch: the streaming leaf is a second child of the target, and the
    // target's strip is resting on the first. Selecting the target's path
    // alone left the strip where it was, so the reply streamed on a branch the
    // reader never saw (task 46, bug 3) — the ask must select the branch its
    // own turn lands on.
    let stores = stub_stores_with_agents(cx, "s");
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());
    let mut a2 = fixture_assistant_post("a2", "the first answer");
    a2.parent_action_id = Some("a1".into());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("a1", "a question"), a2], cx)
        });
    })
    .unwrap();
    cx.update_window(window, |_, window, _| window.refresh())
        .unwrap();
    cx.run_until_parked();

    // Resting on the existing reply.
    let before = cx
        .update_window(window, |_, window, cx| {
            view.read(cx).selected_effective_path_for_test(window, cx)
        })
        .unwrap();
    assert!(
        before.contains(&"a2".to_string()),
        "the reader starts on the existing reply's branch ({before:?})"
    );

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.ask_participant("agent-b".into(), "a1".into(), window, cx)
        });
    })
    .unwrap();
    cx.update_window(window, |_, window, _| window.refresh())
        .unwrap();
    cx.run_until_parked();

    let seq = view.read_with(cx, |v, cx| v.space().read(cx).streams()[0].seq);
    let path = cx
        .update_window(window, |_, window, cx| {
            view.read(cx).selected_effective_path_for_test(window, cx)
        })
        .unwrap();
    assert!(
        path.contains(&streaming_node_id(seq).to_string()),
        "the ask selects the branch its own turn streams in ({path:?})"
    );
    assert!(
        !path.contains(&"a2".to_string()),
        "and leaves the sibling branch it forked away from ({path:?})"
    );
}

#[gpui::test]
fn space_ask_selects_the_reply_when_the_turn_lands_before_the_next_render(cx: &mut TestAppContext) {
    // The ordering edge of the test above. Nothing sequences a render between
    // the ask and its turn's completion: gpui draws from the platform's frame
    // callback (`flush_effects` draws only in test builds), while a turn's
    // completion is an ordinary foreground-task update — so a turn that lands
    // fast (a decline, an immediate refusal, a cached answer) removes its
    // streaming entry and puts the persisted response in its place *before*
    // any frame runs. A selection deferred by streaming-leaf id then names a
    // node that no longer exists and is dropped: bug 3 again, one frame later.
    // Modelled by doing both in one update — gpui flushes effects, and only
    // then draws, when the outermost update unwinds.
    let stores = stub_stores_with_agents(cx, "s");
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());
    let mut a2 = fixture_assistant_post("a2", "the first answer");
    a2.parent_action_id = Some("a1".into());
    let mut a3 = fixture_assistant_post("a3", "the second answer");
    a3.parent_action_id = Some("a1".into());
    let before_tree = vec![fixture_user_post("a1", "a question"), a2.clone()];
    let after_tree = vec![fixture_user_post("a1", "a question"), a2, a3];
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| s.set_post_tree_for_test(before_tree, cx));
    })
    .unwrap();
    cx.update_window(window, |_, window, _| window.refresh())
        .unwrap();
    cx.run_until_parked();

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.ask_participant("agent-b".into(), "a1".into(), window, cx)
        });
        let seq = space.read(cx).streams()[0].seq;
        space.update(cx, |s, cx| {
            s.apply_turn_success_for_test(seq, after_tree, false, cx)
        });
    })
    .unwrap();
    cx.update_window(window, |_, window, _| window.refresh())
        .unwrap();
    cx.run_until_parked();

    let path = cx
        .update_window(window, |_, window, cx| {
            view.read(cx).selected_effective_path_for_test(window, cx)
        })
        .unwrap();
    assert!(
        path.contains(&"a3".to_string()),
        "the reply the ask started must be selected even when its turn \
         finished before the selection could be consumed ({path:?})"
    );
    assert!(
        !path.contains(&"a2".to_string()),
        "and the branch it forked away from is left behind ({path:?})"
    );
}

#[gpui::test]
fn space_ask_that_lands_pre_render_settles_at_the_end_of_the_answer(cx: &mut TestAppContext) {
    // The ask parks the reader at the end of the branch its turn lands on. When
    // the turn is still streaming that end is the answer's end — but a turn
    // that completed before the render is followed onto its *post*, and by the
    // time the request is consumed `sync_tail_drafts` has already docked a
    // fresh composer under it. Settling at the document end then scrolls
    // straight past the answer into the draft's window of runway: bug 2's
    // overshoot, reached through bug 3's path. "The end" has one definition —
    // the end of what was written.
    let stores = stub_stores_with_agents(cx, "s");
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());
    let long = "a long paragraph of the conversation so far. ".repeat(40);
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("a1", &long)], cx)
        });
    })
    .unwrap();
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(520.)));
    vcx.run_until_parked();

    // Ask, and the turn lands before the next frame.
    let mut a2 = fixture_assistant_post("a2", &long);
    a2.parent_action_id = Some("a1".into());
    vcx.update(|window, cx| {
        view.update(cx, |v, cx| {
            v.ask_participant("agent-b".into(), "a1".into(), window, cx)
        });
        let seq = space.read(cx).streams()[0].seq;
        space.update(cx, |s, cx| {
            s.apply_turn_success_for_test(seq, vec![fixture_user_post("a1", &long), a2], false, cx)
        });
    });
    vcx.run_until_parked();

    view.read_with(&vcx, |v, _| {
        assert!(
            v.draft_parents_for_test().contains(&Some("a2".to_string())),
            "the finished exchange grew its tail draft ({:?})",
            v.draft_parents_for_test()
        );
        let doc_end = v.scroll_min_y_for_test();
        let content_end = v.content_end_for_test();
        let offset = v.page_scroll_offset_y_for_test();
        assert!(
            content_end > doc_end + 1.0,
            "the trailing draft's runway is what separates the two (content \
             end {content_end}, document end {doc_end})"
        );
        assert!(
            (offset - content_end).abs() < 2.0,
            "the ask settles at the end of the answer, not past it (offset \
             {offset}, content end {content_end}, document end {doc_end})"
        );
    });
}

#[gpui::test]
fn space_a_newer_ask_outranks_a_completing_turns_retarget(cx: &mut TestAppContext) {
    // Two aims at once: the reader is parked on turn A's stream *and* has just
    // asked B, so B's leaf is the pending request. If A then lands before the
    // next render, following A onto its post must not overwrite that request —
    // a pending selection is the reader's latest ask, and the newest intent
    // wins. Otherwise the ask silently does nothing and the reader stays on
    // the answer they had already read.
    let stores = stub_stores_with_agents(cx, "s");
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("a1", "a question")], cx)
        });
    })
    .unwrap();
    cx.run_until_parked();

    // Parked on the first participant's stream.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.ask_participant("agent-b".into(), "a1".into(), window, cx)
        });
    })
    .unwrap();
    cx.update_window(window, |_, window, _| window.refresh())
        .unwrap();
    cx.run_until_parked();
    let first = view.read_with(cx, |v, cx| v.space().read(cx).streams()[0].seq);

    // Ask a second participant, and the first turn lands before either can
    // render.
    let mut a2 = fixture_assistant_post("a2", "the first agent's answer");
    a2.parent_action_id = Some("a1".into());
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.ask_participant("agent-c".into(), "a1".into(), window, cx)
        });
        space.update(cx, |s, cx| {
            s.apply_turn_success_for_test(
                first,
                vec![fixture_user_post("a1", "a question"), a2],
                false,
                cx,
            )
        });
    })
    .unwrap();
    cx.update_window(window, |_, window, _| window.refresh())
        .unwrap();
    cx.run_until_parked();

    let second = view.read_with(cx, |v, cx| v.space().read(cx).streams()[0].seq);
    let path = cx
        .update_window(window, |_, window, cx| {
            view.read(cx).selected_effective_path_for_test(window, cx)
        })
        .unwrap();
    assert!(
        path.contains(&streaming_node_id(second).to_string()),
        "the reader's newest ask wins over the older turn's landing ({path:?})"
    );
    assert!(
        !path.contains(&"a2".to_string()),
        "so the completed turn's post does not steal the selection ({path:?})"
    );
}

#[gpui::test]
fn space_selected_turn_keeps_its_branch_when_it_lands_before_its_sibling(cx: &mut TestAppContext) {
    // The same swap, one frame later: the reader is already *parked* on a
    // turn's streaming leaf when that turn lands. Branch selection is
    // positional (a strip's scroll offset → child index), and a turn's
    // persisted response is inserted among the target's *posts*, ahead of any
    // still-streaming sibling overlay. So in a fan-out where the selected
    // (later) turn completes first, the index the reader is resting on now
    // addresses the *other* participant's stream, and the view switches away
    // from the answer they were reading. Selection has to follow the turn
    // through its completion, the way drafts and tree focus follow item
    // identity across a rebuild.
    let stores = stub_stores_with_agents(cx, "s");
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("a1", "a question")], cx)
        });
    })
    .unwrap();
    cx.run_until_parked();

    // Two participants answer the same post: two concurrent turns, ordered
    // siblings under it.
    for agent in ["agent-b", "agent-c"] {
        cx.update_window(window, |_, window, cx| {
            view.update(cx, |v, cx| {
                v.ask_participant(agent.into(), "a1".into(), window, cx)
            });
        })
        .unwrap();
        cx.update_window(window, |_, window, _| window.refresh())
            .unwrap();
        cx.run_until_parked();
    }
    let (first, second) = view.read_with(cx, |v, cx| {
        let s = v.space().read(cx);
        (s.streams()[0].seq, s.streams()[1].seq)
    });
    let path = cx
        .update_window(window, |_, window, cx| {
            view.read(cx).selected_effective_path_for_test(window, cx)
        })
        .unwrap();
    assert!(
        path.contains(&streaming_node_id(second).to_string()),
        "the reader is parked on the second turn's stream ({path:?})"
    );

    // The turn they are reading lands first.
    let mut a3 = fixture_assistant_post("a3", "the second agent's answer");
    a3.parent_action_id = Some("a1".into());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.apply_turn_success_for_test(
                second,
                vec![fixture_user_post("a1", "a question"), a3],
                false,
                cx,
            )
        });
    })
    .unwrap();
    cx.update_window(window, |_, window, _| window.refresh())
        .unwrap();
    cx.run_until_parked();

    let path = cx
        .update_window(window, |_, window, cx| {
            view.read(cx).selected_effective_path_for_test(window, cx)
        })
        .unwrap();
    assert!(
        path.contains(&"a3".to_string()),
        "the reader stays on the reply they were reading once it persists \
         ({path:?})"
    );
    assert!(
        !path.contains(&streaming_node_id(first).to_string()),
        "and is not swapped onto the other participant's stream ({path:?})"
    );
}

#[gpui::test]
fn space_two_turns_landing_together_keep_a_scrolled_away_reader_still(cx: &mut TestAppContext) {
    // A settle mode records the *reader's* situation: parked means don't move
    // them, asked means take them there. When a sibling turn lands first, the
    // reader's own branch is preserved with `Stay` — and if their turn then
    // lands too, before any frame consumed that request, re-aiming it onto the
    // written post must forward the node identity **only**. Upgrading the
    // settle to `BranchEnd` would scroll a reader who had deliberately gone
    // back up the branch down to the tail of a reply they never asked to be
    // taken to.
    let stores = stub_stores_with_agents(cx, "s");
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());
    let long = "a long paragraph of the conversation so far. ".repeat(40);
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("a1", &long)], cx)
        });
    })
    .unwrap();
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(520.)));
    vcx.run_until_parked();

    // Two participants answer at once; the ask leaves the reader on the second.
    for agent in ["agent-b", "agent-c"] {
        vcx.update(|window, cx| {
            view.update(cx, |v, cx| {
                v.ask_participant(agent.into(), "a1".into(), window, cx)
            });
        });
        vcx.run_until_parked();
    }
    let (first, second) = view.read_with(&vcx, |v, cx| {
        let s = v.space().read(cx);
        (s.streams()[0].seq, s.streams()[1].seq)
    });

    // The reader goes back up their own branch to re-read the question.
    view.read_with(&vcx, |v, _| v.scroll_page_to_top_for_test());
    vcx.run_until_parked();
    let parked_at = view.read_with(&vcx, |v, _| v.page_scroll_offset_y_for_test());

    // Both turns land before the next frame — the sibling first.
    let mut a2 = fixture_assistant_post("a2", &long);
    a2.parent_action_id = Some("a1".into());
    let mut a3 = fixture_assistant_post("a3", &long);
    a3.parent_action_id = Some("a1".into());
    let question = fixture_user_post("a1", &long);
    vcx.update(|_, cx| {
        space.update(cx, |s, cx| {
            s.apply_turn_success_for_test(first, vec![question.clone(), a2.clone()], false, cx);
            s.apply_turn_success_for_test(second, vec![question.clone(), a2, a3], false, cx);
        });
    });
    vcx.run_until_parked();

    let path = vcx.update(|window, cx| view.read(cx).selected_effective_path_for_test(window, cx));
    assert!(
        path.contains(&"a3".to_string()),
        "the reader keeps the branch they were reading ({path:?})"
    );
    let offset = view.read_with(&vcx, |v, _| v.page_scroll_offset_y_for_test());
    let content_end = view.read_with(&vcx, |v, _| v.content_end_for_test());
    assert!(
        content_end < parked_at - 1.0,
        "the answers must have given the page somewhere to be dragged to \
         (parked at {parked_at}, content end {content_end})"
    );
    assert!(
        (offset - parked_at).abs() < 2.0,
        "and they are left where they scrolled to, not carried to the reply's \
         tail (offset {offset}, parked at {parked_at}, content end \
         {content_end})"
    );
}

#[gpui::test]
fn space_dot_chosen_branch_survives_its_turn_landing_at_once(cx: &mut TestAppContext) {
    // The instant arm of a dot switch (reduce-motion, or a strip already at the
    // target) writes the strip offset directly and returns — so nothing records
    // which child the reader chose. If that child's own turn then lands before
    // the next frame, the completion finds neither a pending request nor a
    // parked seq, and the reader is left on an index that now addresses the
    // *other* participant's stream: the same positional shift, entering through
    // the one switch that never told anyone where it went.
    cx.update(|cx| cx.set_reduce_motion(true));
    let stores = stub_stores_with_agents(cx, "s");
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("a1", "a question")], cx)
        });
    })
    .unwrap();
    cx.run_until_parked();
    // Three concurrent turns, so switching to the *middle* one is a real move
    // whose landing shifts the index under the reader (its post is inserted
    // ahead of every remaining stream).
    for agent in ["agent-b", "agent-c", "agent-d"] {
        cx.update_window(window, |_, window, cx| {
            view.update(cx, |v, cx| {
                v.ask_participant(agent.into(), "a1".into(), window, cx)
            });
        })
        .unwrap();
        cx.update_window(window, |_, window, _| window.refresh())
            .unwrap();
        cx.run_until_parked();
    }
    let (first, middle) = view.read_with(cx, |v, cx| {
        let s = v.space().read(cx);
        (s.streams()[0].seq, s.streams()[1].seq)
    });

    // Click to the middle participant, and let *its* turn land before the next
    // frame.
    let mut a2 = fixture_assistant_post("a2", "the middle agent's answer");
    a2.parent_action_id = Some("a1".into());
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.click_branch_dot_for_test("a1", 1, window, cx));
        space.update(cx, |s, cx| {
            s.apply_turn_success_for_test(
                middle,
                vec![fixture_user_post("a1", "a question"), a2],
                false,
                cx,
            )
        });
    })
    .unwrap();
    cx.update_window(window, |_, window, _| window.refresh())
        .unwrap();
    cx.run_until_parked();

    let path = cx
        .update_window(window, |_, window, cx| {
            view.read(cx).selected_effective_path_for_test(window, cx)
        })
        .unwrap();
    assert!(
        path.contains(&"a2".to_string()),
        "the branch the reader clicked to carries them onto its answer \
         ({path:?})"
    );
    assert!(
        !path.contains(&streaming_node_id(first).to_string()),
        "not onto the stream that slid into the index they were on ({path:?})"
    );
}

#[gpui::test]
fn space_dot_chosen_branch_survives_a_sibling_landing_mid_switch(cx: &mut TestAppContext) {
    // The animating arm of the same switch. The strip slides over several
    // frames, and each of those frames observes the *rounded* offset — which is
    // still the branch being left. A sibling turn landing mid-slide therefore
    // read the old child as "parked", re-selected it by identity, and cancelled
    // the snap: the reader's switch silently undone by a completion. What a
    // switch knows at click time is its destination, and that is what must
    // stand until it lands.
    let stores = stub_stores_with_agents(cx, "s");
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("a1", "a question")], cx)
        });
    })
    .unwrap();
    cx.run_until_parked();
    for agent in ["agent-b", "agent-c"] {
        cx.update_window(window, |_, window, cx| {
            view.update(cx, |v, cx| {
                v.ask_participant(agent.into(), "a1".into(), window, cx)
            });
        })
        .unwrap();
        cx.update_window(window, |_, window, _| window.refresh())
            .unwrap();
        cx.run_until_parked();
    }
    let (first, second) = view.read_with(cx, |v, cx| {
        let s = v.space().read(cx);
        (s.streams()[0].seq, s.streams()[1].seq)
    });

    // Switch back to the first participant. The snap animates (no test
    // dispatcher pumps its frame loop, which is exactly the mid-slide state).
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.click_branch_dot_for_test("a1", 0, window, cx));
    })
    .unwrap();
    cx.update_window(window, |_, window, _| window.refresh())
        .unwrap();
    cx.run_until_parked();

    // The branch they are *leaving* lands.
    let mut a2 = fixture_assistant_post("a2", "the second agent's answer");
    a2.parent_action_id = Some("a1".into());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.apply_turn_success_for_test(
                second,
                vec![fixture_user_post("a1", "a question"), a2],
                false,
                cx,
            )
        });
    })
    .unwrap();
    cx.update_window(window, |_, window, _| window.refresh())
        .unwrap();
    cx.run_until_parked();

    let path = cx
        .update_window(window, |_, window, cx| {
            view.read(cx).selected_effective_path_for_test(window, cx)
        })
        .unwrap();
    assert!(
        path.contains(&streaming_node_id(first).to_string()),
        "the reader arrives where they were heading ({path:?})"
    );
    assert!(
        !path.contains(&"a2".to_string()),
        "the landing turn does not pull them back to the branch they left \
         ({path:?})"
    );
}

#[gpui::test]
fn space_newer_branch_navigation_outranks_the_cached_turn_when_it_lands(cx: &mut TestAppContext) {
    let stores = stub_stores_with_agents(cx, "s");
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("a1", "a question")], cx)
        });
    })
    .unwrap();
    cx.run_until_parked();

    for agent in ["agent-b", "agent-c"] {
        cx.update_window(window, |_, window, cx| {
            view.update(cx, |v, cx| {
                v.ask_participant(agent.into(), "a1".into(), window, cx)
            });
        })
        .unwrap();
        cx.update_window(window, |_, window, _| window.refresh())
            .unwrap();
        cx.run_until_parked();
    }
    let (first, second) = view.read_with(cx, |v, cx| {
        let s = v.space().read(cx);
        (s.streams()[0].seq, s.streams()[1].seq)
    });

    // The last frame cached `second`; now navigate to `first` without drawing
    // another frame before `second` completes.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.select_effective_path_for_test(&streaming_node_id(first), window, cx)
        });
    })
    .unwrap();
    let mut response = fixture_assistant_post("a3", "the second agent's answer");
    response.parent_action_id = Some("a1".into());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.apply_turn_success_for_test(
                second,
                vec![fixture_user_post("a1", "a question"), response],
                false,
                cx,
            )
        });
    })
    .unwrap();
    cx.update_window(window, |_, window, _| window.refresh())
        .unwrap();
    cx.run_until_parked();

    let path = cx
        .update_window(window, |_, window, cx| {
            view.read(cx).selected_effective_path_for_test(window, cx)
        })
        .unwrap();
    assert!(
        path.contains(&streaming_node_id(first).to_string()),
        "the newer branch navigation survives the old turn landing ({path:?})"
    );
    assert!(
        !path.contains(&"a3".to_string()),
        "the stale selected-turn cache does not restore the completed reply ({path:?})"
    );
}

#[gpui::test]
fn space_tail_ask_keeps_nonempty_draft_as_its_own_branch(cx: &mut TestAppContext) {
    // The tail-draft rule, non-empty half: a draft with content is kept
    // exactly as it is — it becomes its own sibling branch beside the incoming
    // response, and stays active in the composer.
    let stores = stub_stores_with_agents(cx, "s");
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("a1", "the only post")], cx)
        });
    })
    .unwrap();
    cx.update_window(window, |_, window, _| window.refresh())
        .unwrap();
    cx.run_until_parked();

    open_space_draft(&view, window, cx, Some("a1"));
    set_space_composer_text(&view, window, cx, "a half-written thought");

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.ask_participant("agent-b".into(), "a1".into(), window, cx)
        });
    })
    .unwrap();
    view.read_with(cx, |v, cx| {
        assert!(
            v.draft_parents_for_test().contains(&Some("a1".to_string())),
            "the non-empty draft is kept as its own branch"
        );
        assert!(
            v.has_active_draft_for_test(),
            "the draft stays active in the composer"
        );
        assert_eq!(v.space().read(cx).streams().len(), 1);
    });
}

#[gpui::test]
fn space_turn_failure_leaves_sibling_streams_untouched(cx: &mut TestAppContext) {
    // Per-turn failure isolation: one turn of a fan-out failing surfaces the
    // recovery notice for that turn without disturbing sibling streams, and
    // the notice's Retry re-asks the same participant while siblings stream.
    let stores = stub_stores_with_agents(cx, "s");
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("a1", "the question")], cx);
            let _seq_b = s.push_streaming_turn_for_test(
                Some("agent-b".into()),
                Some("a1".into()),
                Default::default(),
                cx,
            );
            let seq_c = s.push_streaming_turn_for_test(
                Some("agent-c".into()),
                Some("a1".into()),
                Default::default(),
                cx,
            );
            s.push_content_delta_for_test(seq_c, "partial thoughts…", cx);
        });
    })
    .unwrap();
    cx.run_until_parked();

    // Fail agent-b's turn only.
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.apply_turn_failure_for_test(
                "agent-b",
                "a1",
                AppError::Network {
                    message: "connection reset".into(),
                },
                cx,
            )
        });
    })
    .unwrap();
    cx.run_until_parked();

    view.read_with(cx, |v, cx| {
        let space = v.space().read(cx);
        let streams = space.streams();
        assert_eq!(streams.len(), 1, "the sibling turn keeps streaming");
        assert_eq!(streams[0].participant_id.as_deref(), Some("agent-c"));
        assert_eq!(
            streams[0].response.content, "partial thoughts…",
            "the sibling's live buffer is untouched"
        );
        let msg = v
            .error_for_test(cx)
            .expect("the failed turn shows a notice");
        assert!(msg.contains("connection reset"));
        assert!(space.can_retry());
        let failed = space.failed_turn().expect("the failed turn is recorded");
        assert_eq!(failed.participant_id, "agent-b");
        assert_eq!(failed.target_action_id, "a1");
    });

    // Retry re-asks agent-b about the same post — the sibling still streams.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.retry_failed(window, cx));
    })
    .unwrap();
    view.read_with(cx, |v, cx| {
        assert!(v.error_for_test(cx).is_none(), "retry clears the notice");
        let space = v.space().read(cx);
        let streams = space.streams();
        assert_eq!(streams.len(), 2, "retry streams beside the sibling");
        assert!(
            streams
                .iter()
                .any(|s| s.participant_id.as_deref() == Some("agent-b")
                    && s.target_action_id.as_deref() == Some("a1")),
            "retry re-asks the same participant about the same post"
        );
        assert!(
            streams
                .iter()
                .any(|s| s.participant_id.as_deref() == Some("agent-c")),
            "the sibling stream survived the retry"
        );
    });
}

#[gpui::test]
fn space_sibling_success_keeps_failed_turn_notice(cx: &mut TestAppContext) {
    // A fan-out where one turn fails and a sibling *succeeds afterward*: the
    // sibling's StreamEnded must NOT hide the failed turn's recovery notice.
    // The notice's lifetime is owned by the Space's `failed_turn` record, so it
    // persists (with Retry) until that turn is retried or explicitly dismissed.
    let stores = stub_stores_with_agents(cx, "s");
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());
    let seq_c = cx
        .update_window(window, |_, _, cx| {
            space.update(cx, |s, cx| {
                s.set_post_tree_for_test(vec![fixture_user_post("a1", "the question")], cx);
                s.push_streaming_turn_for_test(
                    Some("agent-b".into()),
                    Some("a1".into()),
                    Default::default(),
                    cx,
                );
                s.push_streaming_turn_for_test(
                    Some("agent-c".into()),
                    Some("a1".into()),
                    Default::default(),
                    cx,
                )
            })
        })
        .unwrap();
    cx.run_until_parked();

    // agent-b's turn fails: the recovery notice + Retry appear.
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.apply_turn_failure_for_test(
                "agent-b",
                "a1",
                AppError::Network {
                    message: "connection reset".into(),
                },
                cx,
            )
        });
    })
    .unwrap();
    cx.run_until_parked();
    view.read_with(cx, |v, cx| {
        assert!(
            v.error_for_test(cx).is_some(),
            "the failed turn shows a notice"
        );
        assert!(v.space().read(cx).can_retry(), "Retry is available");
    });

    // agent-c's sibling turn now *succeeds* (StreamEnded) — the notice must
    // survive (the bug: the sibling's success silently removed the Retry path).
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| s.finish_streaming_turn_for_test(seq_c, cx));
    })
    .unwrap();
    cx.run_until_parked();
    view.read_with(cx, |v, cx| {
        assert!(
            v.error_for_test(cx).is_some(),
            "a sibling's success must not hide the failed turn's notice"
        );
        let space = v.space().read(cx);
        assert!(space.can_retry(), "Retry survives the sibling success");
        let failed = space
            .failed_turn()
            .expect("the failed turn is still recorded");
        assert_eq!(failed.participant_id, "agent-b");
        assert_eq!(failed.target_action_id, "a1");
    });

    // Dismiss ends the recovery: the notice is gone and nothing is retryable.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.dismiss_error(window, cx));
    })
    .unwrap();
    view.read_with(cx, |v, cx| {
        assert!(v.error_for_test(cx).is_none(), "dismiss clears the notice");
        assert!(
            !v.space().read(cx).can_retry(),
            "dismiss ends the recovery — nothing left to retry"
        );
    });
}

#[gpui::test]
fn space_cascade_notice_renders_dismisses_and_asks_to_continue(cx: &mut TestAppContext) {
    // A paused cascade surfaces the quiet, dismissible notice whose action is
    // an explicit ask (which bypasses the guard by construction — asserted in
    // app-core's orchestration tests; here we assert the render + routing).
    let stores = stub_stores_with_agents(cx, "s");
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("a1", "the question")], cx);
            s.emit_cascade_paused_for_test(4, 4, "a1".into(), cx);
        });
    })
    .unwrap();
    cx.run_until_parked();
    view.read_with(cx, |v, _| {
        assert_eq!(
            v.cascade_notice_for_test(),
            Some((4, 4, "a1".to_string())),
            "the paused plan surfaces the notice"
        );
    });

    // Dismissible (window-local; nothing else changes).
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.dismiss_cascade(window, cx));
    })
    .unwrap();
    view.read_with(cx, |v, _| assert!(v.cascade_notice_for_test().is_none()));

    // Re-announced on a later pause; "Ask <agent>" continues the conversation.
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.emit_cascade_paused_for_test(4, 4, "a1".into(), cx)
        });
    })
    .unwrap();
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.ask_participant("agent-c".into(), "a1".into(), window, cx)
        });
    })
    .unwrap();
    view.read_with(cx, |v, cx| {
        assert!(
            v.cascade_notice_for_test().is_none(),
            "asking clears the notice"
        );
        let space = v.space().read(cx);
        assert_eq!(space.streams().len(), 1);
        assert_eq!(
            space.streams()[0].participant_id.as_deref(),
            Some("agent-c")
        );
    });
}

/// An assistant (inference) post fixture — `fixture_user_post` with the agent
/// participant and action type, so role resolves to "assistant".
fn fixture_assistant_post(action_id: &str, text: &str) -> PostNode {
    let mut p = fixture_user_post(action_id, text);
    p.action_type = "inference".into();
    p.participant = PostParticipant {
        kind: "agent".into(),
        label: "kimi-k2".into(),
    };
    p.model = Some("kimi-k2".into());
    p
}

/// Seed an existing space with a user post (a1) and an assistant reply (a2),
/// forcing a frame so `sync_bodies` mints the per-post editors.
fn seed_space_pair(view: &Entity<SpaceView>, window: AnyWindowHandle, cx: &mut TestAppContext) {
    let space = view.read_with(cx, |v, _| v.space().clone());
    let mut a2 = fixture_assistant_post("a2", "the reply");
    a2.parent_action_id = Some("a1".into());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("a1", "original text"), a2], cx)
        });
    })
    .unwrap();
    cx.update_window(window, |_, window, _| window.refresh())
        .unwrap();
    cx.run_until_parked();
}

#[gpui::test]
fn space_composer_cmd_enter_routes_through_press_enter_to_submit(cx: &mut TestAppContext) {
    // The composer owns the ⌘↩ chord: dispatching the editor's
    // `Enter { secondary: true }` action (what `cmd-enter` binds to in the
    // `MarkdownEditor` context) must make the editor emit `PressEnter`, which
    // the draft's subscription (`create_draft_node`) routes to `submit`. This
    // exercises the full outward-event wiring, where the `&Send`-dispatch
    // tests bypass it. Ported from the retired ChatView suite.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, None);

    let editor = view
        .read_with(cx, |v, _| v.composer_state_for_test())
        .expect("blank space opens with the composer");
    cx.update_window(window, |_, _, cx| {
        editor.update(cx, |e, cx| e.set_value("via press-enter".to_string(), cx));
    })
    .unwrap();

    let editor_focus = editor.read_with(cx, |e, cx| e.focus_handle(cx));
    cx.update_window(window, |_, window, cx| {
        editor_focus.dispatch_action(
            &gpui_markdown_editor::Enter {
                secondary: true,
                shift: false,
            },
            window,
            cx,
        );
    })
    .unwrap();
    cx.run_until_parked();

    view.read_with(cx, |v, cx| {
        let space = v.space().read(cx);
        assert_eq!(space.messages().len(), 1);
        assert_eq!(space.messages()[0].message.content, "via press-enter");
        assert!(space.is_streaming(), "⌘↩ via PressEnter streams, like Send");
        assert_eq!(v.draft_count_for_test(), 0, "the draft was consumed");
    });
}

#[gpui::test]
fn space_composer_accessible_value_freezes_while_it_is_focused(cx: &mut TestAppContext) {
    // The composer's `aria_value` must not track keystrokes: assistive
    // technology re-reads a focused control's whole value on every change, so
    // a live binding would re-speak the entire draft per character (audit §4 —
    // the same reason Zed's own text field freezes). It refreshes only at
    // settled moments: a different draft becoming active, or focus leaving.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, None);
    draw_frame(cx, window);

    let editor = view
        .read_with(cx, |v, _| v.composer_state_for_test())
        .expect("blank space opens with the composer");
    let editor_focus = editor.read_with(cx, |e, cx| e.focus_handle(cx));
    cx.update_window(window, |_, window, cx| {
        editor_focus.focus(window, cx);
    })
    .unwrap();
    cx.run_until_parked();

    set_space_composer_text(&view, window, cx, "half a thought, still typing");
    draw_frame(cx, window);
    view.read_with(cx, |v, _| {
        assert_eq!(
            v.composer_aria_value_for_test().as_ref(),
            "",
            "typing must not move the accessible value"
        );
    });

    // Focus leaves the composer (the draft stays active): the value settles to
    // what is actually there, which is when a reader would ask for it.
    let root = view.read_with(cx, |v, _| v.focus_handle());
    cx.update_window(window, |_, window, cx| root.focus(window, cx))
        .unwrap();
    cx.run_until_parked();
    draw_frame(cx, window);
    view.read_with(cx, |v, _| {
        assert_eq!(
            v.composer_aria_value_for_test().as_ref(),
            "half a thought, still typing",
            "blur settles the value"
        );
    });
}

#[gpui::test]
fn space_composer_reopened_draft_reads_the_text_it_holds(cx: &mut TestAppContext) {
    // Escape retires the draft *before* any frame renders it unfocused, and
    // re-opening focuses the very same draft — so a rule that refreshes only on
    // "a different draft" or "an unfocused frame" never catches up, and the
    // reopened composer reports its pre-typing text forever. Activation is the
    // seam that fixes it: every editing session begins there.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, None);
    draw_frame(cx, window);

    let editor = view
        .read_with(cx, |v, _| v.composer_state_for_test())
        .expect("blank space opens with the composer");
    let editor_focus = editor.read_with(cx, |e, cx| e.focus_handle(cx));
    cx.update_window(window, |_, window, cx| editor_focus.focus(window, cx))
        .unwrap();
    cx.run_until_parked();
    set_space_composer_text(&view, window, cx, "a thought worth keeping");
    draw_frame(cx, window);

    // Escape: retire the draft and move focus to the view root — exactly what
    // the composer's key handler does, in that order.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.deactivate_for_test(cx));
        let root = view.read(cx).focus_handle();
        root.focus(window, cx);
    })
    .unwrap();
    cx.run_until_parked();
    draw_frame(cx, window);
    view.read_with(cx, |v, _| {
        assert_eq!(
            v.draft_count_for_test(),
            1,
            "the draft is kept — it has content"
        );
        assert!(
            !v.has_active_draft_for_test(),
            "Escape retires it before any frame renders the composer unfocused"
        );
        assert_eq!(
            v.composer_aria_value_for_test().as_ref(),
            "",
            "so the snapshot is still the stale, pre-typing one"
        );
    });

    // Click back into it — `activate_draft`, the same call the inactive
    // draft's click and the editor's own `Focus` event make. The id is
    // unchanged and the editor is focused again, so nothing the render-time
    // rule can see has moved; only the activation seed catches this up.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.activate_draft_for_test(0, cx));
        editor_focus.focus(window, cx);
    })
    .unwrap();
    cx.run_until_parked();
    draw_frame(cx, window);
    view.read_with(cx, |v, _| {
        assert!(v.has_active_draft_for_test(), "the draft is active again");
        assert_eq!(
            v.composer_aria_value_for_test().as_ref(),
            "a thought worth keeping",
            "a reopened draft must report the text it actually holds"
        );
    });
}

#[gpui::test]
fn space_composer_edit_arms_caret_scroll_into_view(cx: &mut TestAppContext) {
    // An edit that pushes the caret below the *floating* composer's visible
    // fold must scroll the composer so the caret stays visible. The full path
    // runs here: the active draft's `Change` arms the flag
    // (`create_draft_node`), and the composer body's `caret_into_view` canvas —
    // which paints under `TestAppContext`, reading the real laid-out caret
    // geometry — moves `composer_scroll` on the next draw. (The pure offset
    // math is additionally unit-tested by `composer::caret_scroll_offset`.)
    //
    // The composer only owns its own scroll when it *floats with overflow*
    // (capped at COMPOSER_MAX_FRACTION); a docked / fit-height composer has
    // `scroll_max == 0` and the page owns scrolling instead. So the scene is a
    // tall conversation with the page scrolled to the top, which pushes the
    // active tail draft's slot far below the fold and floats the composer.
    use eidola_app_core::{PostBlock, PostNode, PostParticipant};
    use gpui_markdown_editor::EditorEvent;
    let post = |aid: &str, parent: Option<&str>, user: bool| PostNode {
        action_id: aid.into(),
        item_id: format!("item-{aid}"),
        parent_action_id: parent.map(Into::into),
        participant: PostParticipant {
            kind: if user { "human".into() } else { "agent".into() },
            label: if user { "You".into() } else { "kimi".into() },
        },
        action_type: if user {
            "user_input".into()
        } else {
            "inference".into()
        },
        generation: 0,
        generation_count: 1,
        is_current: true,
        model: None,
        credits_consumed: None,
        relation: parent.map(|_| "reply".to_string()),
        depth: 0,
        is_branch: false,
        blocks: vec![PostBlock {
            id: String::new(),
            block_type: "text".into(),
            text: Some(
                "A few sentences of body text so each post has a realistic \
                 measured height, tall enough that the transcript overflows."
                    .into(),
            ),
            tool_name: None,
            tool_call_id: None,
            data: None,
        }],
        references: Vec::new(),
        created_at: 0,
    };
    let nodes: Vec<PostNode> = (0..8)
        .map(|i| {
            let parent = (i > 0).then(|| format!("a{}", i - 1));
            post(&format!("a{i}"), parent.as_deref(), i % 2 == 0)
        })
        .collect();

    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("caret".into()));
    view.update(cx, |v, cx| {
        v.space()
            .update(cx, |s, cx| s.set_post_tree_for_test(nodes, cx));
    });
    cx.run_until_parked();
    // Activate a tail draft replying to the last post (its slot is at the
    // bottom of the tall document).
    open_space_draft(&view, window, cx, Some("a7"));

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(620.)));
    vcx.run_until_parked();

    let editor = view
        .read_with(&vcx, |v, _| v.composer_state_for_test())
        .expect("the tail draft is the active composer");

    // Scroll the page to the top **and settle a frame** so the composer's slot
    // sits far below the fold and the composer renders floating (capped)
    // before the edit lands. Order matters: the caret canvas branches on the
    // frame's own docked/floating decision, and a fresh draft is docked at its
    // home — arming the flag while that stale docked frame is still current
    // routes the caret to the *page*, which is docked behavior, not the
    // floating behavior under test.
    view.read_with(&vcx, |v, _| v.scroll_page_to_top_for_test());
    view.update(&mut vcx, |_, cx| cx.notify());
    vcx.run_until_parked();

    // Type a draft far taller than the ~310px floating viewport. `InsertText`
    // leaves the caret at the end of the inserted text — below the fold.
    let long = "line of the draft that carries some words\n".repeat(40);
    editor.update(&mut vcx, |e, cx| {
        e.apply_event_for_test(EditorEvent::InsertText(long), cx)
    });
    vcx.run_until_parked();

    view.read_with(&vcx, |v, _| {
        assert!(
            v.composer_scroll_offset_y_for_test() < -1.0,
            "the caret ran below the floating composer's fold, so it scrolled \
             down to keep the caret visible (offset {} should be negative)",
            v.composer_scroll_offset_y_for_test()
        );
        assert!(
            !v.caret_scroll_pending_for_test(),
            "the caret-into-view canvas consumed the pending flag"
        );
    });
}

#[gpui::test]
fn space_scrolled_floating_composer_glides_to_its_top_by_the_dock(cx: &mut TestAppContext) {
    // The float→dock transition for an internally-scrolled composer (the
    // pre-dock glide, `SpaceView::glide_composer_toward_dock`). A floating
    // composer scrolled off its own top must NOT carry that scroll into the
    // dock: while the dock threshold sits under the floating bar (the last
    // ≤half-window of page travel before docking), each increment of page
    // scroll unwinds a proportional share of the internal offset, so the
    // content sits at exactly its top the moment the composer docks. Page
    // scrolling outside that zone must never move the internal content, the
    // unwind must be monotone, and reversing the page mid-zone must leave the
    // internal offset where it is (the glide never reverses) while a resumed
    // descent still lands exactly at the top. The per-step math is unit-tested
    // (`composer::approach_glide_offset`); this drives it through real renders
    // at arbitrary stop/reverse/restart points.
    use eidola_app_core::{PostBlock, PostNode, PostParticipant};
    use gpui_markdown_editor::EditorEvent;
    let post = |aid: &str, parent: Option<&str>, user: bool| PostNode {
        action_id: aid.into(),
        item_id: format!("item-{aid}"),
        parent_action_id: parent.map(Into::into),
        participant: PostParticipant {
            kind: if user { "human".into() } else { "agent".into() },
            label: if user { "You".into() } else { "kimi".into() },
        },
        action_type: if user {
            "user_input".into()
        } else {
            "inference".into()
        },
        generation: 0,
        generation_count: 1,
        is_current: true,
        model: None,
        credits_consumed: None,
        relation: parent.map(|_| "reply".to_string()),
        depth: 0,
        is_branch: false,
        blocks: vec![PostBlock {
            id: String::new(),
            block_type: "text".into(),
            text: Some(
                "A few sentences of body text so each post has a realistic \
                 measured height, tall enough that the transcript overflows."
                    .into(),
            ),
            tool_name: None,
            tool_call_id: None,
            data: None,
        }],
        references: Vec::new(),
        created_at: 0,
    };
    let nodes: Vec<PostNode> = (0..8)
        .map(|i| {
            let parent = (i > 0).then(|| format!("a{}", i - 1));
            post(&format!("a{i}"), parent.as_deref(), i % 2 == 0)
        })
        .collect();

    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("glide".into()));
    view.update(cx, |v, cx| {
        v.space()
            .update(cx, |s, cx| s.set_post_tree_for_test(nodes, cx));
    });
    cx.run_until_parked();
    open_space_draft(&view, window, cx, Some("a7"));

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(620.)));
    vcx.run_until_parked();

    let editor = view
        .read_with(&vcx, |v, _| v.composer_state_for_test())
        .expect("the tail draft is the active composer");

    // Park the page at the top (and settle a frame) so the slot sits far below
    // the fold and the composer renders floating **before** the edit lands —
    // the caret canvas branches on the frame's own docked/floating decision,
    // and the fresh draft is otherwise still docked at its home. Then type a
    // draft far taller than the ~310px floating viewport; the caret-into-view
    // path scrolls the floating composer deep off its own top.
    view.read_with(&vcx, |v, _| v.scroll_page_to_top_for_test());
    view.update(&mut vcx, |_, cx| cx.notify());
    vcx.run_until_parked();
    let long = "line of the draft that carries some words\n".repeat(40);
    editor.update(&mut vcx, |e, cx| {
        e.apply_event_for_test(EditorEvent::InsertText(long), cx)
    });
    vcx.run_until_parked();
    // A settle frame so the glide's runway tracking has a baseline before the
    // stepping begins (the first tracked frame only records, never steps).
    view.update(&mut vcx, |_, cx| cx.notify());
    vcx.run_until_parked();

    let step = |vcx: &mut VisualTestContext, view: &Entity<SpaceView>, dy: f32| -> (bool, f32) {
        view.update(vcx, |v, cx| {
            v.scroll_page_by_for_test(dy);
            cx.notify();
        });
        vcx.run_until_parked();
        view.read_with(vcx, |v, _| {
            (
                v.composer_overlayed_for_test(),
                v.composer_scroll_offset_y_for_test(),
            )
        })
    };

    let off0 = view.read_with(&vcx, |v, _| v.composer_scroll_offset_y_for_test());
    assert!(
        off0 < -100.0,
        "precondition: the floating composer is scrolled well off its top \
         (offset {off0})"
    );

    // Outside the approach zone (the slot is several hundred px below the
    // window bottom here) page scrolling must not touch the internal offset.
    for _ in 0..2 {
        let (overlayed, off) = step(&mut vcx, &view, -20.0);
        assert!(overlayed, "still floating just below the page top");
        assert!(
            (off - off0).abs() < 0.5,
            "page scroll outside the approach zone moved the internal offset \
             ({off0} -> {off})"
        );
    }

    // Walk the page toward the dock. The offset must unwind monotonically,
    // begin unwinding while still floating, survive a mid-zone reversal
    // untouched, and read ~0 on the very first docked frame.
    let mut last_off = off0;
    let mut reversed = false;
    let mut docked_off = None;
    for i in 0..200 {
        let (overlayed, off) = step(&mut vcx, &view, -40.0);
        assert!(
            off >= last_off - 0.01,
            "the glide must never deepen the internal scroll (step {i}: \
             {last_off} -> {off})"
        );
        last_off = off;
        if !overlayed {
            docked_off = Some(off);
            break;
        }
        // Once the glide has visibly engaged (still floating, partially
        // unwound), reverse the page mid-zone: the offset must hold — the
        // glide never reverses — and the later descent still lands at 0.
        if !reversed && off > off0 + 5.0 {
            reversed = true;
            for _ in 0..3 {
                let (_, back_off) = step(&mut vcx, &view, 40.0);
                assert!(
                    (back_off - off).abs() < 0.5,
                    "reversing the page must leave the internal offset where \
                     it was ({off} -> {back_off})"
                );
            }
            last_off = off;
        }
    }
    let docked_off = docked_off.expect("the page walk reached the dock threshold");
    assert!(
        reversed,
        "the glide must engage while the composer still floats (the offset \
         never moved off {off0} before docking)"
    );
    assert!(
        docked_off.abs() < 1.0,
        "the internal scroll must sit at exactly its top the moment the \
         composer docks (was {docked_off})"
    );

    // And it stays at the top through the rest of the docked traversal.
    view.update(&mut vcx, |v, cx| {
        v.scroll_page_to_end_for_test();
        cx.notify();
    });
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert!(
            v.composer_scroll_offset_y_for_test().abs() < 1.0,
            "the docked composer stays at its top to the end of the document \
             (offset {})",
            v.composer_scroll_offset_y_for_test()
        );
    });
}

#[gpui::test]
fn space_composer_resize_drag_pins_exact_height_and_reverts_on_deactivate(cx: &mut TestAppContext) {
    // The separator resize handle's state machine. Grabbing the handle
    // switches the window-local sizing to **Exact** at the bar's *current*
    // ratio (never a jump under the grab), dragging follows the pointer as a
    // delta clamped to the fraction bounds — sizing the bar in excess of its
    // (unchanged) content, which Max could never do — releasing keeps the
    // pin, and deactivating (Escape) reverts the pin to Max while the
    // fraction itself survives as the window's cap.
    let (_window, view, mut vcx) = open_floating_composer_scene(cx, "resize");
    const WIN: f32 = 620.0;

    let (overlayed, fraction, exact, natural) = view.read_with(&vcx, |v, _| {
        (
            v.composer_overlayed_for_test(),
            v.composer_fraction_for_test(),
            v.composer_sizing_is_exact_for_test(),
            v.composer_float_bar_h_for_test(WIN),
        )
    });
    assert!(overlayed, "the scene's tail draft floats");
    assert!(
        (fraction - 0.5).abs() < 1e-6 && !exact,
        "every window opens at the default: fraction 0.5, Max sizing \
         (fraction {fraction}, exact {exact})"
    );
    // The empty draft floats at its natural height, well under the cap —
    // which is what makes "in excess of the content" observable below.
    assert!(
        natural < 0.5 * WIN - 1.0,
        "precondition: the empty draft's natural bar ({natural}) rests under \
         the 50% cap"
    );

    // Grab the handle: Exact immediately, pinned at the current height.
    view.update(&mut vcx, |v, cx| {
        v.begin_composer_resize_for_test(400.0, WIN, cx)
    });
    view.read_with(&vcx, |v, _| {
        assert!(
            v.composer_sizing_is_exact_for_test(),
            "grabbing the handle switches to Exact immediately"
        );
        let bar = v.composer_float_bar_h_for_test(WIN);
        assert!(
            (bar - natural).abs() < 1.0,
            "the grab pins the bar where it rests — no jump ({natural} -> {bar})"
        );
    });

    // Drag 150px up: the bar is exactly 150 taller — in excess of content.
    view.update(&mut vcx, |v, cx| {
        v.move_composer_resize_for_test(250.0, WIN, cx)
    });
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        let bar = v.composer_float_bar_h_for_test(WIN);
        assert!(
            (bar - (natural + 150.0)).abs() < 1.0,
            "the bar edge follows the pointer ({} vs {})",
            bar,
            natural + 150.0
        );
        assert!(
            v.composer_overlayed_for_test(),
            "still floating at the pinned height"
        );
    });

    // Wild drags clamp to the fraction bounds (deltas are from the grab, so
    // these don't accumulate).
    view.update(&mut vcx, |v, cx| {
        v.move_composer_resize_for_test(-10_000.0, WIN, cx)
    });
    view.read_with(&vcx, |v, _| {
        assert!(
            (v.composer_fraction_for_test() - 0.85).abs() < 1e-6,
            "dragging past the top clamps to the max fraction (got {})",
            v.composer_fraction_for_test()
        );
    });
    view.update(&mut vcx, |v, cx| {
        v.move_composer_resize_for_test(10_000.0, WIN, cx)
    });
    view.read_with(&vcx, |v, _| {
        let floor = v.composer_min_fraction_for_test(WIN);
        assert!(
            floor >= 0.1,
            "the effective floor never dips below the nominal minimum ({floor})"
        );
        assert!(
            (v.composer_fraction_for_test() - floor).abs() < 1e-6,
            "dragging past the bottom clamps to the effective floor (got {}, floor {floor})",
            v.composer_fraction_for_test()
        );
    });

    // Settle mid-range and release: the pin survives the drag's end.
    view.update(&mut vcx, |v, cx| {
        v.move_composer_resize_for_test(250.0, WIN, cx);
        v.end_composer_resize_for_test(cx);
    });
    vcx.run_until_parked();
    let pinned = view.read_with(&vcx, |v, _| {
        assert!(
            v.composer_sizing_is_exact_for_test(),
            "Exact survives releasing the handle"
        );
        v.composer_fraction_for_test()
    });

    // Deactivate (Escape's path): the pin reverts to Max; the fraction is
    // window state and survives as the cap until re-dragged.
    view.update(&mut vcx, |v, cx| v.deactivate_for_test(cx));
    view.read_with(&vcx, |v, _| {
        assert!(
            !v.composer_sizing_is_exact_for_test(),
            "deactivation reverts the sizing to Max"
        );
        assert!(
            (v.composer_fraction_for_test() - pinned).abs() < 1e-6,
            "the fraction survives deactivation ({} vs {pinned})",
            v.composer_fraction_for_test()
        );
    });
}

#[gpui::test]
fn compact_floating_composer_uses_complete_natural_height(cx: &mut TestAppContext) {
    let (_window, view, mut vcx) = open_floating_composer_scene(cx, "compact-natural-height");
    const WIN: f32 = 620.0;

    // A non-empty draft, so the actions reveal and the bottom bar holds its
    // reservation alongside the docked byline row.
    let editor = view
        .read_with(&vcx, |v, _| v.composer_state_for_test())
        .expect("the scene's draft is active");
    editor.update(&mut vcx, |editor, cx| {
        editor.set_value("a line of draft", cx)
    });
    vcx.run_until_parked();
    for _ in 0..2 {
        vcx.update(|window, _| window.refresh());
        vcx.run_until_parked();
    }

    view.read_with(&vcx, |v, _| {
        let (natural, base, gutters) = v.composer_height_contract_for_test();
        let (byline_row, total) = v.composer_gutter_contract_for_test();
        let action_bar = total - byline_row;
        assert!(
            byline_row > 0.0 && action_bar > 0.0,
            "the compact composer reserves its docked byline row and its bottom action bar"
        );
        assert!(
            (natural - base - gutters).abs() < 0.01,
            "docked natural height {natural} includes base {base} and both gutters {gutters}"
        );
        let floating = v.composer_floating_natural_height_for_test();
        assert!(
            (floating - base - action_bar).abs() < 0.01,
            "the floating bar carries the action bar but never the byline row \
             (floating {floating}, base {base}, action bar {action_bar})"
        );
        assert!(
            (v.composer_float_bar_h_for_test(WIN) - floating).abs() < 0.5,
            "the uncapped floating bar uses the complete floating natural height"
        );
    });
}

#[gpui::test]
fn space_composer_resize_reverts_to_max_when_the_draft_posts(cx: &mut TestAppContext) {
    // Posting consumes the draft — a deactivation — so an exact-height resize
    // reverts to Max with it (the `send_active_draft` reset path, distinct
    // from Escape's `retire_active_draft`).
    let (window, view, mut vcx) = open_floating_composer_scene(cx, "resize-post");
    const WIN: f32 = 620.0;

    view.update(&mut vcx, |v, cx| {
        v.begin_composer_resize_for_test(400.0, WIN, cx);
        v.move_composer_resize_for_test(250.0, WIN, cx);
        v.end_composer_resize_for_test(cx);
    });
    view.read_with(&vcx, |v, _| {
        assert!(v.composer_sizing_is_exact_for_test());
    });

    set_space_composer_text(&view, window, &mut vcx, "a resized draft, posted");
    dispatch_space_action(&view, window, &mut vcx, Send);

    view.read_with(&vcx, |v, _| {
        assert!(
            !v.has_active_draft_for_test(),
            "the accepted post consumed the draft"
        );
        assert!(
            !v.composer_sizing_is_exact_for_test(),
            "posting reverts the sizing to Max"
        );
    });
}

#[gpui::test]
fn space_docked_composer_edit_scrolls_page_into_view(cx: &mut TestAppContext) {
    // The docked/page counterpart of the floating test above. A blank ⌘N
    // notebook's composer is **docked**: it owns no internal scroll and expands
    // to full height, growing *below* the window. Typing a first message taller
    // than the window runs the caret off the bottom, and bringing it back is a
    // `page_scroll` concern, not `composer_scroll` — so the docked branch of
    // `caret_into_view` follows the caret with the page.
    use gpui_markdown_editor::EditorEvent;
    let stores = stub_stores_with_config(cx);
    // A blank (id-less) space opens with its root tail draft already the active,
    // docked composer.
    let (window, view) = open_space(cx, &stores, None);
    view.read_with(cx, |v, _| {
        assert!(
            v.has_active_draft_for_test(),
            "a blank space opens with its (docked) composer"
        );
    });

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(560.)));
    vcx.run_until_parked();

    let editor = view
        .read_with(&vcx, |v, _| v.composer_state_for_test())
        .expect("the blank space's root draft is the active composer");

    // Type a first message far taller than the 560px window. `InsertText`
    // leaves the caret at the end of the inserted text — below the window fold.
    let long = "line of the first message in a blank notebook\n".repeat(40);
    editor.update(&mut vcx, |e, cx| {
        e.apply_event_for_test(EditorEvent::InsertText(long), cx)
    });
    vcx.run_until_parked();

    view.read_with(&vcx, |v, _| {
        // The docked composer never owns internal scroll: composer_scroll stays 0.
        assert!(
            v.composer_scroll_offset_y_for_test().abs() < 0.5,
            "a docked composer owns no internal scroll (composer_scroll should \
             stay 0, was {})",
            v.composer_scroll_offset_y_for_test()
        );
        // Instead the whole page scrolled down to reveal the caret.
        assert!(
            v.page_scroll_offset_y_for_test() < -1.0,
            "the caret ran below the window, so the PAGE scrolled down to keep it \
             visible (page offset {} should be negative)",
            v.page_scroll_offset_y_for_test()
        );
        assert!(
            !v.caret_scroll_pending_for_test(),
            "the caret-into-view canvas consumed the pending flag"
        );
    });

    // Full-reveal assertion. The docked
    // branch computes the caret's DOCUMENT position as `page_slot_doc_top +
    // editor_top_offset + caret_content_bottom`, then scrolls the page to reveal
    // it. The final page-scroll value is gpui-clamped against a frame-lagged
    // content size (and races under parallel test load), so we assert the
    // frame-independent piece the branch recorded: the slot-relative offset it
    // folded in (`caret_doc_bottom − caret_content_bottom` = `page_slot_doc_top +
    // editor_top_offset`). For a blank ⌘N space the sole node is the draft leaf,
    // so `page_slot_doc_top` is just the document's top reserve and this must
    // equal `reserve + POST_PAD_Y + compact_top` — the editor's content-top
    // offset within the slot. The top metadata line precedes the editor in
    // compact layout, while total top+bottom occupancy still sizes the runway.
    let (slot_offset, reserve, compact_top, compact_total) = view.read_with(&vcx, |v, _| {
        let (top, total) = v.composer_gutter_contract_for_test();
        (
            v.docked_caret_slot_offset_for_test(),
            v.doc_reserve_for_test(),
            top,
            total,
        )
    });
    let post_pad_y = 40.0_f32;
    assert!(compact_top > 0.0, "the compact composer has top metadata");
    assert!(
        compact_total > compact_top,
        "the nonempty compact composer also has bottom actions"
    );
    let expected = reserve + post_pad_y + compact_top;
    assert!(
        (slot_offset - expected).abs() < 1.0,
        "the docked reveal must fold the {compact_top}px compact metadata line \
         into the editor's document position (slot-relative offset {slot_offset}, \
         expected {expected}); omitting it under-scrolls the lower-fold caret \
         by exactly that occupancy",
    );
}

#[gpui::test]
fn space_stale_initial_load_does_not_replace_submitted_prompt(cx: &mut TestAppContext) {
    // The load-vs-submit race is serialized inside the `Space` entity: a
    // reopened space's initial load completing *after* a local submit is
    // stale and must be dropped, not clobber the just-submitted prompt.
    // Ported from the retired ChatView suite (the entity logic is shared).
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("space-123".into()));

    open_space_draft(&view, window, cx, None);
    set_space_composer_text(&view, window, cx, "new prompt");
    dispatch_space_action(&view, window, cx, Send);

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
        let space = v.space().read(cx);
        assert_eq!(space.messages().len(), 1);
        assert_eq!(space.messages()[0].message.content, "new prompt");
        assert!(space.is_streaming());
    });
}

#[gpui::test]
fn space_edit_and_regenerate_supersede_in_flight_load(cx: &mut TestAppContext) {
    // Same stale-fetch class as the Record listing race (the codex finding on
    // PR #179), on the transcript: a reload in flight (e.g. bus-driven, from a
    // CLI write to the same space) when the user commits an edit or regenerate
    // must be *cancelled* at that moment — the mutation's own post-commit
    // reload is the authoritative truth, and a superseded load must never land
    // late around it. The structural fix is the shared mutation prologue
    // (`Space::supersede_load_for_mutation`, replace-cancels on the load
    // slot), which `submit`/`post_only` already ran and `edit`/
    // `regenerate_post` previously skipped.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_space_pair(&view, window, cx);
    let space = view.read_with(cx, |v, _| v.space().clone());

    space.update(cx, |s, cx| {
        s.arm_load_for_test(cx);
        assert!(s.edit("a1".into(), "edited".into(), Vec::new(), cx));
        assert!(
            !s.has_pending_load_for_test(),
            "committing an edit must supersede the in-flight transcript load"
        );
    });

    space.update(cx, |s, cx| {
        s.arm_load_for_test(cx);
        assert!(s.regenerate_post("a2".into(), "gemma4-31b".into(), cx));
        assert!(
            !s.has_pending_load_for_test(),
            "regenerate must supersede the in-flight transcript load"
        );
    });
}

#[gpui::test]
fn space_declined_turn_does_not_attach_its_reasoning_to_another_post(cx: &mut TestAppContext) {
    // The agent-side decline checkpoint: a declined turn wrote no post, so its
    // `response_action_id` is `None`. `merge_from_db`'s `None`-action fallback
    // attaches captured reasoning to the *last assistant message*, which would
    // put the declining agent's private thinking under another agent's reply.
    // A decline must drop the capture instead.
    let stores = stub_stores_with_agents(cx, "s");
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());
    let mut a2 = fixture_assistant_post("a2", "agent-c already answered");
    a2.parent_action_id = Some("a1".into());
    let tree = vec![fixture_user_post("a1", "the question"), a2];

    // A turn streams reasoning and then declines.
    let seq = cx
        .update_window(window, |_, _, cx| {
            space.update(cx, |s, cx| {
                s.set_post_tree_for_test(tree.clone(), cx);
                s.push_streaming_turn_for_test(
                    Some("agent-b".into()),
                    Some("a1".into()),
                    eidola_gui::space::StreamingResponse {
                        reasoning: "agent-b's private deliberation".into(),
                        ..Default::default()
                    },
                    cx,
                )
            })
        })
        .unwrap();
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.apply_turn_success_for_test(seq, tree.clone(), true, cx)
        });
    })
    .unwrap();
    cx.run_until_parked();

    view.read_with(cx, |v, _| {
        assert_eq!(
            v.post_reasoning_for_test(1),
            None,
            "a decline must not hang its reasoning on another agent's post"
        );
    });

    // Control: the *same* finalize on an ordinary (non-declined) turn does
    // attach — so the assertion above is about the decline, not about the
    // capture never landing.
    let seq = cx
        .update_window(window, |_, _, cx| {
            space.update(cx, |s, cx| {
                s.push_streaming_turn_for_test(
                    Some("agent-b".into()),
                    Some("a1".into()),
                    eidola_gui::space::StreamingResponse {
                        reasoning: "agent-b's private deliberation".into(),
                        ..Default::default()
                    },
                    cx,
                )
            })
        })
        .unwrap();
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.apply_turn_success_for_test(seq, tree.clone(), false, cx)
        });
    })
    .unwrap();
    cx.run_until_parked();
    view.read_with(cx, |v, _| {
        assert_eq!(
            v.post_reasoning_for_test(1),
            Some(("agent-b's private deliberation".to_string(), false)),
            "an ordinary turn still attaches its reasoning to the post it wrote"
        );
    });
}

#[gpui::test]
fn space_post_reasoning_projection_toggles(cx: &mut TestAppContext) {
    // Reasoning re-attached to a finalized post survives into the render
    // snapshot, and `Space::toggle_message_reasoning` flips the disclosure —
    // the "Thinking…" toggle on a finished reply (the ChatView feature the
    // space view previously dropped at finalize).
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_space_pair(&view, window, cx);
    let space = view.read_with(cx, |v, _| v.space().clone());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_reasoning_for_test(1, "chain of thought".into(), false, cx)
        });
    })
    .unwrap();
    cx.run_until_parked();

    view.read_with(cx, |v, _| {
        assert_eq!(
            v.post_reasoning_for_test(1),
            Some(("chain of thought".to_string(), false)),
            "reasoning flows into the render snapshot"
        );
    });
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| s.toggle_message_reasoning(1, cx));
    })
    .unwrap();
    cx.run_until_parked();
    view.read_with(cx, |v, _| {
        assert_eq!(
            v.post_reasoning_for_test(1),
            Some(("chain of thought".to_string(), true)),
            "the disclosure toggle reaches the snapshot"
        );
    });
}

/// Seed a space whose branch is far taller than the window and start one
/// streaming turn on it, returning the turn's seq. Shared by the two
/// tail-following cases.
fn seed_streaming_tall_space(
    view: &Entity<SpaceView>,
    window: AnyWindowHandle,
    cx: &mut TestAppContext,
) -> u64 {
    let space = view.read_with(cx, |v, _| v.space().clone());
    let long = "a long paragraph of the conversation so far. ".repeat(40);
    let mut a2 = fixture_assistant_post("a2", &long);
    a2.parent_action_id = Some("a1".into());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("a1", &long), a2], cx);
            s.push_streaming_turn_for_test(
                Some("agent-b".into()),
                Some("a2".into()),
                Default::default(),
                cx,
            )
        })
    })
    .unwrap()
}

#[gpui::test]
fn space_streaming_tail_follows_when_parked_at_the_end(cx: &mut TestAppContext) {
    // Parked at the end of the branch while a turn streams: each delta grows
    // the document, and the page stays pinned to the new end — the answer
    // writes itself into view instead of running off the bottom.
    let stores = stub_stores_with_agents(cx, "s");
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());
    let seq = seed_streaming_tall_space(&view, window, cx);

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(520.)));
    vcx.run_until_parked();

    // Park the reader at the tail.
    view.read_with(&vcx, |v, _| v.scroll_page_to_end_for_test());
    vcx.run_until_parked();
    let before = view.read_with(&vcx, |v, _| v.scroll_min_y_for_test());
    assert!(
        before < -1.0,
        "the seeded branch must overflow the window (scroll_min_y {before})"
    );

    // The turn produces a long reply.
    space.update(&mut vcx, |s, cx| {
        s.push_content_delta_for_test(seq, &"streamed answer line\n".repeat(60), cx)
    });
    vcx.run_until_parked();

    view.read_with(&vcx, |v, _| {
        let after = v.scroll_min_y_for_test();
        assert!(
            after < before - 1.0,
            "the streamed reply must grow the document ({before} -> {after})"
        );
        assert!(
            (v.page_scroll_offset_y_for_test() - after).abs() < 2.0,
            "the page follows the producing tail (offset {} should track the new \
             end {after})",
            v.page_scroll_offset_y_for_test()
        );
    });
}

#[gpui::test]
fn space_streaming_tail_yields_to_a_navigation_glide(cx: &mut TestAppContext) {
    // A reader parked at the tail who asks to be taken somewhere is on their
    // way — the glide owns the page until it lands. Following and the glide are
    // the only two motions that span frames, and both write the offset every
    // frame (each delta moves "the end"), so without the stand-down the
    // follower takes the page straight back — through the very seam that lets
    // an instant scroll retire a glide, on the first frame after the
    // navigation.
    let stores = stub_stores_with_agents(cx, "s");
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());
    let seq = seed_streaming_tall_space(&view, window, cx);

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(520.)));
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| v.scroll_page_to_end_for_test());
    vcx.run_until_parked();
    let end = view.read_with(&vcx, |v, _| v.scroll_min_y_for_test());

    // Taken to the top of the conversation, mid-stream.
    vcx.update(|window, cx| {
        view.update(cx, |v, cx| v.navigate_to_action("a1".into(), window, cx));
    });
    vcx.run_until_parked();
    assert!(
        view.read_with(&vcx, |v, _| v.page_glide_target_for_test())
            .is_some(),
        "the navigation armed a glide"
    );
    view.update(&mut vcx, |v, _| v.drive_page_glide_for_test(0.4));
    let mid = view.read_with(&vcx, |v, _| v.page_scroll_offset_y_for_test());

    // The reply keeps coming while the reader travels.
    space.update(&mut vcx, |s, cx| {
        s.push_content_delta_for_test(seq, &"streamed answer line\n".repeat(60), cx)
    });
    vcx.run_until_parked();

    view.read_with(&vcx, |v, _| {
        let grown = v.scroll_min_y_for_test();
        assert!(
            grown < end - 1.0,
            "the streamed reply grew the document ({end} -> {grown})"
        );
        assert!(
            (v.page_scroll_offset_y_for_test() - mid).abs() < 2.0,
            "the traveling reader is not dragged back to the tail (offset {}, \
             on the glide at {mid}, new end {grown})",
            v.page_scroll_offset_y_for_test()
        );
    });
}

#[gpui::test]
fn space_streaming_tail_does_not_yank_a_reader_who_scrolled_away(cx: &mut TestAppContext) {
    // The other half of the contract: a reader who has scrolled back up to
    // re-read something must be left exactly where they are, however much the
    // streaming reply grows.
    let stores = stub_stores_with_agents(cx, "s");
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());
    let seq = seed_streaming_tall_space(&view, window, cx);

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(520.)));
    vcx.run_until_parked();

    // Scroll away from the tail (all the way back to the top).
    view.read_with(&vcx, |v, _| v.scroll_page_to_top_for_test());
    vcx.run_until_parked();
    let before = view.read_with(&vcx, |v, _| v.scroll_min_y_for_test());

    space.update(&mut vcx, |s, cx| {
        s.push_content_delta_for_test(seq, &"streamed answer line\n".repeat(60), cx)
    });
    vcx.run_until_parked();

    view.read_with(&vcx, |v, _| {
        assert!(
            v.scroll_min_y_for_test() < before - 1.0,
            "the streamed reply must grow the document (else this proves nothing)"
        );
        assert!(
            v.page_scroll_offset_y_for_test().abs() < 2.0,
            "a reader who scrolled away is never yanked to the tail (offset {})",
            v.page_scroll_offset_y_for_test()
        );
    });
}

#[gpui::test]
fn space_streaming_tail_ignores_a_sibling_branchs_stream(cx: &mut TestAppContext) {
    // Tail-following is scoped to the branch the reader is on. A fan-out can
    // stream on a *sibling* branch, and while it does, the selected branch's
    // own document still grows for the ordinary non-stream reasons the design
    // deliberately excludes (its composer's runway, a post measuring for the
    // first time, a post arriving from another window). None of those is a
    // producing tail, so none of them may move the reader — even though some
    // turn, somewhere in the space, is streaming.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());

    let long = "a long paragraph of the conversation so far. ".repeat(40);
    // a1 forks: a2 is the (selected) first branch, a3 the second.
    let branch = |aid: &str| {
        let mut p = fixture_assistant_post(aid, &long);
        p.parent_action_id = Some("a1".into());
        p
    };
    let seq = cx
        .update_window(window, |_, _, cx| {
            space.update(cx, |s, cx| {
                s.set_post_tree_for_test(
                    vec![fixture_user_post("a1", &long), branch("a2"), branch("a3")],
                    cx,
                );
                // The turn streams on the *other* branch.
                s.push_streaming_turn_for_test(
                    Some("agent-b".into()),
                    Some("a3".into()),
                    Default::default(),
                    cx,
                )
            })
        })
        .unwrap();

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(520.)));
    vcx.run_until_parked();

    // The reader is on the first branch (the default active page) and parked at
    // its end.
    let leaf = vcx.update(|window, cx| view.read(cx).selected_leaf_for_test(window));
    assert_eq!(
        leaf.as_deref(),
        Some("a2"),
        "the reader is on the first branch — the one *not* streaming"
    );
    view.read_with(&vcx, |v, _| v.scroll_page_to_end_for_test());
    vcx.run_until_parked();
    let (parked, before) = view.read_with(&vcx, |v, _| {
        (v.page_scroll_offset_y_for_test(), v.scroll_min_y_for_test())
    });
    assert!(
        before < -1.0,
        "the selected branch must overflow the window (scroll_min_y {before})"
    );

    // The sibling's stream keeps producing — it must be irrelevant here.
    space.update(&mut vcx, |s, cx| {
        s.push_content_delta_for_test(seq, &"streamed answer line\n".repeat(60), cx)
    });
    // …while the *selected* branch grows for a non-stream reason: a post lands
    // on it (another window's write, or this one's own reload).
    space.update(&mut vcx, |s, cx| {
        let mut a4 = fixture_user_post("a4", &long);
        a4.parent_action_id = Some("a2".into());
        s.set_post_tree_for_test(
            vec![
                fixture_user_post("a1", &long),
                branch("a2"),
                branch("a3"),
                a4,
            ],
            cx,
        )
    });
    vcx.run_until_parked();

    view.read_with(&vcx, |v, _| {
        let after = v.scroll_min_y_for_test();
        assert!(
            after < before - 1.0,
            "the selected branch's document must have grown ({before} -> {after})"
        );
        assert!(
            (v.page_scroll_offset_y_for_test() - parked).abs() < 2.0,
            "a stream on a sibling branch must not make the selected branch's \
             own growth follow the reader (offset {} should still be {parked})",
            v.page_scroll_offset_y_for_test()
        );
    });
}

#[gpui::test]
fn space_following_reader_stops_at_the_end_of_the_streamed_content(cx: &mut TestAppContext) {
    // When the stream ends, the finished exchange grows a fresh tail draft — a
    // whole window of empty runway. A reader who was following the tail must
    // come to rest at the end of what was *written*, not be carried down into
    // the speculative reply below it (task 46, bug 2). The window is real: the
    // turn runner reloads the tree and re-plans the cascade after its stream
    // entry is gone, so the space is still busy (and the reader still pinned)
    // while `sync_tail_drafts` docks the new composer.
    let stores = stub_stores_with_agents(cx, "s");
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());

    let long = "a long paragraph of the conversation so far. ".repeat(40);
    let mut a2 = fixture_assistant_post("a2", &long);
    a2.parent_action_id = Some("a1".into());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("a1", &long), a2], cx)
        });
    })
    .unwrap();

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(520.)));
    vcx.run_until_parked();

    // Post, then watch the answer stream in (the pin is armed by the post).
    vcx.update(|window, cx| {
        view.update(cx, |v, cx| {
            v.create_draft_for_test(Some("a2".into()), window, cx)
        });
    });
    vcx.run_until_parked();
    let editor = view
        .read_with(&vcx, |v, _| v.composer_state_for_test())
        .expect("the draft is the active composer");
    editor.update(&mut vcx, |e, cx| e.set_value("a new post", cx));
    let focus = view.read_with(&vcx, |v, _| v.focus_handle());
    vcx.update(|window, cx| focus.dispatch_action(&Send, window, cx));
    vcx.run_until_parked();

    let seq = view.read_with(&vcx, |v, cx| v.space().read(cx).streams()[0].seq);
    space.update(&mut vcx, |s, cx| {
        s.push_content_delta_for_test(seq, &"streamed answer line\n".repeat(60), cx)
    });
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        let offset = v.page_scroll_offset_y_for_test();
        let end = v.scroll_min_y_for_test();
        assert!(
            (offset - end).abs() < 2.0,
            "while streaming, the end of content *is* the end of the document ({offset} vs {end})"
        );
    });

    // The stream closes; the runner is still settling (the post runner stands
    // in for it), so the pin still holds while the tail draft appears.
    let mut a2b = fixture_assistant_post("a2", &long);
    a2b.parent_action_id = Some("a1".into());
    let mut a3 = fixture_user_post("a3", "a new post");
    a3.parent_action_id = Some("a2".into());
    let mut a4 = fixture_assistant_post("a4", &"streamed answer line\n".repeat(60));
    a4.parent_action_id = Some("a3".into());
    space.update(&mut vcx, |s, cx| {
        s.finish_streaming_turn_for_test(seq, cx);
        s.set_post_tree_for_test(vec![fixture_user_post("a1", &long), a2b, a3, a4], cx);
        s.arm_post_runner_for_test(cx);
    });
    vcx.run_until_parked();

    view.read_with(&vcx, |v, _| {
        assert!(
            v.tail_pin_for_test(),
            "the pin outlives the stream while the turn settles"
        );
        assert!(
            v.draft_parents_for_test().contains(&Some("a4".to_string())),
            "the finished exchange grew its tail draft"
        );
        let doc_end = v.scroll_min_y_for_test();
        let content_end = v.content_end_for_test();
        let offset = v.page_scroll_offset_y_for_test();
        assert!(
            content_end > doc_end + 1.0,
            "the trailing draft's runway is what separates the two \
             (content end {content_end}, document end {doc_end})"
        );
        assert!(
            (offset - content_end).abs() < 2.0,
            "the reader rests at the end of the streamed content, not at the \
             end of the document (offset {offset}, content end {content_end}, \
             document end {doc_end})"
        );
    });
}

#[gpui::test]
fn a_reader_scroll_during_convergence_takes_the_viewport(cx: &mut TestAppContext) {
    // The post-submit pin's forcing phase (`TailPin::Converging`) exists to
    // hold the reader at the tail while the just-posted rows converge from
    // estimates to measured heights — but a reader who scrolls (or navigates)
    // away during those frames has taken the viewport, and the pin must
    // demote to observation instead of snapping them back on the next render.
    let stores = stub_stores_with_agents(cx, "s");
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());

    let long = "a long paragraph of the conversation so far. ".repeat(40);
    let mut a2 = fixture_assistant_post("a2", &long);
    a2.parent_action_id = Some("a1".into());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("a1", &long), a2], cx)
        })
    })
    .unwrap();

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(520.)));
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| v.scroll_page_to_top_for_test());
    vcx.run_until_parked();

    vcx.update(|window, cx| {
        view.update(cx, |v, cx| {
            v.create_draft_for_test(Some("a2".into()), window, cx)
        });
    });
    vcx.run_until_parked();
    let editor = view
        .read_with(&vcx, |v, _| v.composer_state_for_test())
        .expect("the draft is the active composer");
    editor.update(&mut vcx, |e, cx| e.set_value("a new post", cx));
    let focus = view.read_with(&vcx, |v, _| v.focus_handle());
    vcx.update(|window, cx| focus.dispatch_action(&Send, window, cx));
    vcx.run_until_parked();

    // Mid-convergence — the pin is still in its forcing phase — the reader
    // scrolls up to reread. The wheel seam demotes the pin, and no later
    // frame drags them back to the tail.
    view.read_with(&vcx, |v, _| {
        assert!(
            v.tail_pin_forced_for_test(),
            "precondition: the submit's pin is still converging"
        );
    });
    view.update(&mut vcx, |v, cx| {
        v.reader_scroll_page_by_for_test(180.0, cx)
    });
    vcx.run_until_parked();
    let taken = view.read_with(&vcx, |v, _| {
        assert!(
            !v.tail_pin_forced_for_test(),
            "the reader's own scroll demotes the pin to observation"
        );
        v.page_scroll_offset_y_for_test()
    });
    for _ in 0..3 {
        vcx.update(|window, _| window.refresh());
        vcx.run_until_parked();
    }
    view.read_with(&vcx, |v, _| {
        let offset = v.page_scroll_offset_y_for_test();
        assert!(
            (offset - taken).abs() < 2.0,
            "no later frame snaps the reader back to the tail \
             (offset {offset}, where they scrolled to {taken})"
        );
    });
}

#[gpui::test]
fn space_post_parks_the_reader_at_the_tail_and_holds_it_there(cx: &mut TestAppContext) {
    // Posting from a reader parked anywhere (here: the top of a long space)
    // must land them at the *end* of the branch the post joined — including the
    // post itself, which the pre-post snapshot does not carry — and hold them
    // there while the exchange settles, so tail-following picks up when the
    // answer starts arriving.
    //
    // Both halves used to be missing. The scroll measured the document before
    // the optimistic turn reached the view's snapshot (the `MessagesChanged`
    // emission is delivered after the submit handler returns), leaving the page
    // a post short of the end; and the growth between the save landing and the
    // response's first delta — the persisted post re-keyed and re-measured —
    // moved the end out from under the reader while nothing was streaming yet.
    // Either one leaves `at_tail` false, and the answer then writes itself off
    // the bottom of the window.
    let stores = stub_stores_with_agents(cx, "s");
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());

    let long = "a long paragraph of the conversation so far. ".repeat(40);
    let mut a2 = fixture_assistant_post("a2", &long);
    a2.parent_action_id = Some("a1".into());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("a1", &long), a2], cx)
        });
    })
    .unwrap();

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(520.)));
    vcx.run_until_parked();

    // The reader is up at the top of the transcript when they post.
    view.read_with(&vcx, |v, _| v.scroll_page_to_top_for_test());
    vcx.run_until_parked();

    vcx.update(|window, cx| {
        view.update(cx, |v, cx| {
            v.create_draft_for_test(Some("a2".into()), window, cx)
        });
    });
    vcx.run_until_parked();
    let editor = view
        .read_with(&vcx, |v, _| v.composer_state_for_test())
        .expect("the draft is the active composer");
    editor.update(&mut vcx, |e, cx| e.set_value("a new post", cx));
    let focus = view.read_with(&vcx, |v, _| v.focus_handle());
    vcx.update(|window, cx| focus.dispatch_action(&Send, window, cx));
    vcx.run_until_parked();
    // The headless dispatcher does not advance `on_next_frame`; draw the
    // settled extent the production tail pin reasserts against.
    for _ in 0..3 {
        vcx.update(|window, _| window.refresh());
        vcx.run_until_parked();
    }

    let (parked, end) = view.read_with(&vcx, |v, _| {
        assert!(v.tail_pin_for_test(), "the post arms the tail pin");
        (v.page_scroll_offset_y_for_test(), v.scroll_min_y_for_test())
    });
    assert!(
        end < -1.0,
        "the seeded branch must overflow the window (scroll_min_y {end})"
    );
    assert!(
        (parked - end).abs() < 2.0,
        "posting parks the reader at the end of the branch, post included \
         (offset {parked} should be the document end {end})"
    );

    // Once the optimistic post has measured and the initial landing has
    // converged, the pin keeps the pre-stream gap eligible for following but
    // no longer overrides the reader's observed position. Re-reading during
    // the live exchange must therefore remain possible.
    view.read_with(&vcx, |v, _| {
        assert!(
            v.tail_pin_for_test(),
            "the exchange still owns the follow gate"
        );
        assert!(
            !v.tail_pin_forced_for_test(),
            "initial measured-floor convergence releases forced authority"
        );
        v.scroll_page_by_for_test(180.0);
    });
    vcx.run_until_parked();
    let reader_offset = view.read_with(&vcx, |v, _| v.page_scroll_offset_y_for_test());
    let seq = view.read_with(&vcx, |v, cx| v.space().read(cx).streams()[0].seq);
    space.update(&mut vcx, |s, cx| {
        s.push_content_delta_for_test(seq, &"a growing answer\n".repeat(20), cx)
    });
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert!(
            (v.page_scroll_offset_y_for_test() - reader_offset).abs() < 2.0,
            "stream growth leaves a reader who moved away where they chose to read"
        );
        v.scroll_page_to_end_for_test();
    });
    vcx.run_until_parked();

    // Now the production gap the pin exists for: the save is in flight and
    // nothing is streaming yet (the stub's synthetic turn stands in for the
    // real one, which only starts once the post has persisted). Both steps run
    // in one update so no frame observes a settled space and retires the pin.
    space.update(&mut vcx, |s, cx| {
        s.finish_streaming_turn_for_test(seq, cx);
        s.arm_post_runner_for_test(cx);
    });
    vcx.run_until_parked();
    view.read_with(&vcx, |v, cx| {
        assert!(!v.space().read(cx).is_streaming(), "the save window");
        assert!(v.tail_pin_for_test(), "the pin outlives the save window");
    });

    // The persisted transcript lands: the post comes back re-keyed (a fresh
    // node id, so it re-measures from its estimate) and the document end moves.
    let mut a2b = fixture_assistant_post("a2", &long);
    a2b.parent_action_id = Some("a1".into());
    let mut a3 = fixture_user_post("a3", &long);
    a3.parent_action_id = Some("a2".into());
    space.update(&mut vcx, |s, cx| {
        s.set_post_tree_for_test(vec![fixture_user_post("a1", &long), a2b, a3], cx)
    });
    vcx.run_until_parked();

    view.read_with(&vcx, |v, _| {
        let after = v.scroll_min_y_for_test();
        assert!(
            after < end - 1.0,
            "the persisted post must have moved the document end ({end} -> {after})"
        );
        // "The end" the pin holds is the end of the *written* content: by now
        // the exchange has grown its tail draft, whose runway is not something
        // to follow into (see `space_following_reader_stops_at_the_end_of_the_
        // streamed_content`).
        let content_end = v.content_end_for_test();
        assert!(
            (v.page_scroll_offset_y_for_test() - content_end).abs() < 2.0,
            "the reader is held at the end across the save (offset {} should \
             track the new content end {content_end}; document end {after})",
            v.page_scroll_offset_y_for_test()
        );
    });
}

#[gpui::test]
fn space_persisted_thinking_block_renders_the_disclosure(cx: &mut TestAppContext) {
    // Reasoning is durable: an inference's `thinking` content block comes back
    // from the post tree, so a *reloaded* space shows the disclosure without
    // any live stream having run in this process. The thinking text must stay
    // out of the reading column (it's a disclosure, not prose) and out of the
    // quotable block spans.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());

    let mut reply = fixture_user_post("a2", "The answer.");
    reply.parent_action_id = Some("a1".into());
    reply.action_type = "inference".into();
    reply.participant = PostParticipant {
        kind: "agent".into(),
        label: "gemma4-31b".into(),
    };
    // The persisted pair, in the order `persist_turn` writes them.
    reply.blocks = vec![
        PostBlock {
            id: "cb-think".into(),
            block_type: "thinking".into(),
            text: Some("chain of thought".into()),
            tool_name: None,
            tool_call_id: None,
            data: None,
        },
        PostBlock {
            id: "cb-text".into(),
            block_type: "text".into(),
            text: Some("The answer.".into()),
            tool_name: None,
            tool_call_id: None,
            data: None,
        },
    ];
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("a1", "the question"), reply], cx)
        });
    })
    .unwrap();
    cx.run_until_parked();

    view.read_with(cx, |v, _| {
        assert_eq!(
            v.post_reasoning_for_test(1),
            Some(("chain of thought".to_string(), false)),
            "a persisted thinking block feeds the disclosure on reload"
        );
    });
    space.read_with(cx, |s, _| {
        let reply = &s.messages()[1];
        assert_eq!(
            reply.message.content, "The answer.",
            "the reading column carries only the text blocks"
        );
        assert_eq!(
            reply.blocks.len(),
            1,
            "only text blocks are quotable; got {:?}",
            reply.blocks
        );
        assert_eq!(reply.blocks[0].block_id, "cb-text");
    });
}

#[gpui::test]
fn space_edit_commits_and_escape_restores(cx: &mut TestAppContext) {
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_space_pair(&view, window, cx);

    // Begin editing the user post: the session records the target and the
    // post's own editor becomes the buffer.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.begin_edit("a1".into(), window, cx));
    })
    .unwrap();
    let editor = view
        .read_with(cx, |v, _| v.post_body_editor_for_test("a1"))
        .expect("the post's body editor exists after a frame");
    view.read_with(cx, |v, _| {
        assert_eq!(v.editing_action_id_for_test(), Some("a1".to_string()));
    });

    // Cancel restores the pre-edit text.
    cx.update_window(window, |_, window, cx| {
        editor.update(cx, |e, cx| e.set_value("mangled".to_string(), cx));
        view.update(cx, |v, cx| v.cancel_edit(window, cx));
    })
    .unwrap();
    cx.update_window(window, |_, _, cx| {
        assert_eq!(editor.read(cx).value(), "original text");
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert_eq!(
            v.editing_action_id_for_test(),
            None,
            "cancel ends the session"
        );
    });

    // Commit accepts the new text (stub Space::edit returns accepted).
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.begin_edit("a1".into(), window, cx));
        editor.update(cx, |e, cx| e.set_value("edited text".to_string(), cx));
        view.update(cx, |v, cx| v.commit_edit(window, cx));
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert_eq!(
            v.editing_action_id_for_test(),
            None,
            "commit ends the session"
        );
    });

    // Non-user posts refuse to enter an edit session.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.begin_edit("a2".into(), window, cx));
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert_eq!(
            v.editing_action_id_for_test(),
            None,
            "assistant posts are not editable"
        );
    });
}

#[gpui::test]
fn space_edit_buffer_survives_transcript_sync(cx: &mut TestAppContext) {
    // `sync_bodies` re-syncs each post editor to the persisted content every
    // frame — but the editor holding an in-progress edit must keep its
    // divergent buffer (a bus-driven reload mid-edit must not clobber typing).
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_space_pair(&view, window, cx);

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.begin_edit("a1".into(), window, cx));
    })
    .unwrap();
    let editor = view
        .read_with(cx, |v, _| v.post_body_editor_for_test("a1"))
        .unwrap();
    cx.update_window(window, |_, _, cx| {
        editor.update(cx, |e, cx| e.set_value("half-typed edit".to_string(), cx));
    })
    .unwrap();

    // Force frames (each render runs sync_bodies against the "original text"
    // persisted content).
    cx.update_window(window, |_, window, _| window.refresh())
        .unwrap();
    cx.run_until_parked();
    cx.update_window(window, |_, _, cx| {
        assert_eq!(
            editor.read(cx).value(),
            "half-typed edit",
            "an in-progress edit buffer must survive sync_bodies"
        );
    })
    .unwrap();
}

// ---------------------------------------------------------------------------
// Failed-ask recovery — the space must never brick on a failed request. A
// failure surfaces a *dismissible* recovery notice, and all three recovery
// paths (edit the message, re-request a response, add a follow-up) stay live.
// ---------------------------------------------------------------------------

/// Seed a space with a single saved user post (a1, no reply) — exactly what a
/// failed ask leaves behind — then drive one **turn's** failure through the
/// shared turn-failure completion (as a real turn runner's error arm does), so
/// the view's recovery notice appears and the failed turn (who was asked,
/// about what) is recorded for Retry.
fn seed_failed_ask(view: &Entity<SpaceView>, window: AnyWindowHandle, cx: &mut TestAppContext) {
    let space = view.read_with(cx, |v, _| v.space().clone());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(
                vec![fixture_user_post("a1", "Hello, what is your name?")],
                cx,
            )
        });
    })
    .unwrap();
    let wrapped = AppError::Network {
        message: "dns error: failed to look up address".into(),
    };
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.apply_turn_failure_for_test("agent-b", "a1", wrapped, cx)
        });
    })
    .unwrap();
    cx.update_window(window, |_, window, _| window.refresh())
        .unwrap();
    cx.run_until_parked();
}

#[gpui::test]
fn space_failed_ask_notice_is_dismissible(cx: &mut TestAppContext) {
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_failed_ask(&view, window, cx);

    // The failure surfaced the recovery notice (with the source's text).
    view.read_with(cx, |v, cx| {
        let msg = v
            .error_for_test(cx)
            .expect("a failure shows the recovery notice");
        assert!(msg.contains("dns error"), "notice carries the error: {msg}");
    });

    // Dismissing clears only the notice — the space is untouched.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.dismiss_error(window, cx));
    })
    .unwrap();
    view.read_with(cx, |v, cx| {
        assert!(v.error_for_test(cx).is_none(), "notice dismissed");
        assert_eq!(v.post_count_for_test(), 1, "the saved post remains");
        assert!(!v.space().read(cx).is_streaming());
    });
}

#[gpui::test]
fn space_failed_ask_can_edit_original(cx: &mut TestAppContext) {
    // Recovery path 1: the failed ask's saved user post is still editable.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_failed_ask(&view, window, cx);

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.begin_edit("a1".into(), window, cx));
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert_eq!(
            v.editing_action_id_for_test(),
            Some("a1".to_string()),
            "a failed ask does not block editing the original message"
        );
    });
    let editor = view
        .read_with(cx, |v, _| v.post_body_editor_for_test("a1"))
        .expect("the post's body editor exists");
    cx.update_window(window, |_, window, cx| {
        editor.update(cx, |e, cx| {
            e.set_value("Actually, who are you?".to_string(), cx)
        });
        view.update(cx, |v, cx| v.commit_edit(window, cx));
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert_eq!(
            v.editing_action_id_for_test(),
            None,
            "the edit committed (stub Space::edit accepts)"
        );
    });
}

#[gpui::test]
fn space_failed_ask_can_re_request(cx: &mut TestAppContext) {
    // Recovery path 2: re-request a response against the saved user post — the
    // notice's Retry action, routed through `Space::retry` (no re-post).
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_failed_ask(&view, window, cx);

    view.read_with(cx, |v, cx| {
        assert!(
            v.space().read(cx).can_retry(),
            "a recorded failed turn can be re-asked"
        );
    });

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.retry_failed(window, cx));
    })
    .unwrap();
    view.read_with(cx, |v, cx| {
        assert!(v.error_for_test(cx).is_none(), "retry clears the notice");
        let space = v.space().read(cx);
        assert!(
            space.is_streaming(),
            "retry re-enters the streaming state (a real send would request a response)"
        );
        assert_eq!(
            space.streams()[0].participant_id.as_deref(),
            Some("agent-b"),
            "retry re-asks the same participant"
        );
    });
}

#[gpui::test]
fn space_failed_ask_can_add_followup(cx: &mut TestAppContext) {
    // Recovery path 3: a normal follow-up at the tail composer still works.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_failed_ask(&view, window, cx);

    // The tail composer reappeared at the leaf a1 after the failure.
    view.read_with(cx, |v, _| {
        assert!(
            v.draft_parents_for_test().contains(&Some("a1".to_string())),
            "a docked tail draft sits at the leaf a1 after a failed ask"
        );
    });

    // Typing a follow-up and Ask-ing it is accepted (not bricked).
    open_space_draft(&view, window, cx, Some("a1"));
    set_space_composer_text(&view, window, cx, "never mind — how are you?");
    dispatch_space_action(&view, window, cx, Send);
    view.read_with(cx, |v, cx| {
        assert!(
            !v.has_active_draft_for_test(),
            "the follow-up draft was consumed"
        );
        assert!(
            v.space().read(cx).is_streaming(),
            "the follow-up entered the streaming state"
        );
        assert_eq!(
            v.space().read(cx).messages().len(),
            2,
            "the follow-up appended a second user turn"
        );
    });
}

#[gpui::test]
fn space_failed_ask_retry_selects_the_failed_posts_branch(cx: &mut TestAppContext) {
    // PR #218 review (comment 2): in a branched space, Retry must select the
    // failed post's branch before streaming, so the response streams under the
    // right post — not whatever branch is currently selected (which would
    // stream under the wrong post until the DB reload snapped it over).
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());

    // a1 (user root) with two replies: a2 (assistant) is the default-selected
    // spine leaf; a3 (user) is a second branch — a saved user post whose ask
    // failed (the recorded failed turn targets a3).
    let mut a2 = fixture_assistant_post("a2", "an answer");
    a2.parent_action_id = Some("a1".into());
    let mut a3 = fixture_user_post("a3", "the failed ask");
    a3.parent_action_id = Some("a1".into());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("a1", "root"), a2, a3], cx);
            s.apply_turn_failure_for_test(
                "agent-b",
                "a3",
                AppError::Network {
                    message: "connection reset".into(),
                },
                cx,
            );
        });
    })
    .unwrap();
    cx.update_window(window, |_, window, _| window.refresh())
        .unwrap();
    cx.run_until_parked();

    // Default selection is the first branch (a2), NOT the retry target (a3).
    let before = cx
        .update_window(window, |_, window, cx| {
            view.read(cx).selected_leaf_for_test(window)
        })
        .unwrap();
    assert_eq!(
        before.as_deref(),
        Some("a2"),
        "default selected leaf is the spine branch, not the failed post"
    );
    view.read_with(cx, |v, cx| {
        assert!(v.space().read(cx).can_retry());
        assert_eq!(
            v.space()
                .read(cx)
                .failed_turn()
                .map(|f| f.target_action_id.clone())
                .as_deref(),
            Some("a3")
        );
    });

    // Retry re-selects the failed post's branch, so the streaming node attaches
    // under a3.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.retry_failed(window, cx));
    })
    .unwrap();
    let after = cx
        .update_window(window, |_, window, cx| {
            view.read(cx).selected_leaf_for_test(window)
        })
        .unwrap();
    assert_eq!(
        after.as_deref(),
        Some("a3"),
        "retry selects the failed post's branch so streaming attaches under it"
    );
    view.read_with(cx, |v, cx| assert!(v.space().read(cx).is_streaming()));
}

#[gpui::test]
fn space_ask_other_post_keeps_unrelated_failed_turn(cx: &mut TestAppContext) {
    // An explicit ask of participant P about post B must NOT clear a failed turn
    // recorded for P about post A — the clearing requires BOTH the participant
    // and the target to match (only a real Retry re-asks P@A). Matching P alone
    // orphaned A's Retry.
    let stores = stub_stores_with_agents(cx, "s");
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());

    // Two user posts; agent-b's ask about a1 failed.
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(
                vec![
                    fixture_user_post("a1", "first question"),
                    fixture_user_post("a2", "second question"),
                ],
                cx,
            );
            s.apply_turn_failure_for_test(
                "agent-b",
                "a1",
                AppError::Network {
                    message: "connection reset".into(),
                },
                cx,
            );
        });
    })
    .unwrap();
    cx.run_until_parked();
    view.read_with(cx, |v, cx| {
        let f = v
            .space()
            .read(cx)
            .failed_turn()
            .expect("a1's failed turn recorded");
        assert_eq!(f.target_action_id, "a1");
    });

    // Explicitly ask agent-b about a DIFFERENT post (a2). This starts a turn on
    // a2 but must leave a1's recorded failure intact.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.ask_participant("agent-b".into(), "a2".into(), window, cx)
        });
    })
    .unwrap();
    cx.run_until_parked();
    view.read_with(cx, |v, cx| {
        let space = v.space().read(cx);
        assert!(
            space.can_retry(),
            "asking P about another post keeps P@A's Retry"
        );
        let f = space
            .failed_turn()
            .expect("a1's failed turn survives the unrelated ask");
        assert_eq!(f.participant_id, "agent-b");
        assert_eq!(
            f.target_action_id, "a1",
            "the surviving failure still targets the originally-failed post"
        );
        assert!(
            space
                .streams()
                .iter()
                .any(|s| s.target_action_id.as_deref() == Some("a2")),
            "the new ask started a streaming turn against a2"
        );
    });
}

#[gpui::test]
fn space_regenerate_uses_posts_own_model(cx: &mut TestAppContext) {
    // With the composer's model picker gone, regenerating re-asks the model
    // that answered — the post's own recorded model, not any global choice.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_space_pair(&view, window, cx);

    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| {
            v.regenerate(&"a2".into(), cx);
        });
    })
    .unwrap();
    view.read_with(cx, |v, cx| {
        assert_eq!(
            v.space().read(cx).last_submitted_model(),
            Some("kimi-k2"),
            "regenerate resolves the post's own recorded model"
        );
    });
}

#[gpui::test]
fn space_post_hover_survives_out_of_order_leave(cx: &mut TestAppContext) {
    // Moving the cursor up the page, the row being left can fire hover-false
    // after the row being entered fired hover-true — the stale leave must not
    // wipe the new row's affordances (the Library lesson).
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_space_pair(&view, window, cx);

    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| {
            v.set_post_hover_for_test("a2", true, cx);
            v.set_post_hover_for_test("a1", true, cx);
            v.set_post_hover_for_test("a2", false, cx); // stale leave
        });
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert_eq!(v.hovered_post_for_test(), Some("a1".to_string()));
    });
}

#[gpui::test]
fn space_alt_modifiers_reach_window_input(cx: &mut TestAppContext) {
    // The root's single `on_modifiers_changed` listener mirrors platform
    // modifier events into the shared `WindowInput` — the ⌥ reveal for the
    // composer's action gutter (Post + keyboard hints) reads from it.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, None);
    view.read_with(cx, |v, cx| assert!(!v.alt_held_for_test(cx)));

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_modifiers_change(Modifiers {
        alt: true,
        ..Modifiers::default()
    });
    view.read_with(&vcx, |v, cx| {
        assert!(
            v.alt_held_for_test(cx),
            "the root listener must mirror ⌥ into WindowInput"
        );
    });
    vcx.simulate_modifiers_change(Modifiers::default());
    view.read_with(&vcx, |v, cx| assert!(!v.alt_held_for_test(cx)));
}

// ---------------------------------------------------------------------------
// Onboarding window
// ---------------------------------------------------------------------------

fn open_onboarding(
    cx: &mut TestAppContext,
    stores: &Stores,
) -> (AnyWindowHandle, Entity<OnboardingView>) {
    let stores = stores.clone();
    open_view(cx, |window, cx| {
        cx.new(|cx| OnboardingView::new(stores.clone(), window, cx))
    })
}

/// Advance the flow through several slides by driving `reveal` directly (the
/// same call each CTA's click handler makes).
fn reveal(view: &Entity<OnboardingView>, cx: &mut TestAppContext, after: Slide, next: Slide) {
    view.update(cx, |v, cx| v.reveal(after, next, cx));
}

#[gpui::test]
fn onboarding_checkout_will_not_fund_an_account_linked_over(cx: &mut TestAppContext) {
    // The Purchase slide's checkout is the same money decision the Account
    // pane's is, and onboarding's back-chevron is a documented way to reach
    // the existing-account slide again: press a plan, go back, link a
    // different account, and the link that lands would fund the one the
    // reader has just walked away from.
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
    });
    let (_w, view) = open_onboarding(cx, &stores);

    let minted_for = config_state(true).account_id;

    view.update(cx, |v, cx| {
        v.finish_checkout(
            minted_for.clone(),
            Ok("https://checkout.example/session/current".into()),
            cx,
        )
    });
    assert_eq!(
        cx.opened_url().as_deref(),
        Some("https://checkout.example/session/current"),
        "the configured account's checkout must open"
    );

    // A different account is linked while the next request is in flight.
    stores.config.update(cx, |c, _| {
        let mut state = config_state(true);
        state.account_id = Some("00000000-0000-7000-8000-000000000333".into());
        state.account_secret = Some("a-linked-accounts-secret".into());
        c.set_state_for_test(Some(state));
    });

    view.update(cx, |v, cx| {
        v.finish_checkout(
            minted_for,
            Ok("https://checkout.example/session/stale".into()),
            cx,
        )
    });
    assert_eq!(
        cx.opened_url().as_deref(),
        Some("https://checkout.example/session/current"),
        "a checkout funding the account that was linked over must not open"
    );
    view.read_with(cx, |v, _| {
        assert!(
            v.checkout_error()
                .is_some_and(|e| e.contains("account changed"))
        );
    });
}

#[gpui::test]
fn onboarding_starts_on_first_slide(cx: &mut TestAppContext) {
    let stores = stub_stores(cx, |_| {});
    let (_w, view) = open_onboarding(cx, &stores);
    view.read_with(cx, |v, _| {
        assert_eq!(v.revealed(), vec![Slide::Pause]);
    });
}

#[gpui::test]
fn onboarding_skip_account_disables_eidola_and_hands_off_to_a_space(cx: &mut TestAppContext) {
    // The GetStarted slide's quiet third choice: no account at all. It
    // disables the eidola backend (optimistically visible in the store; the
    // stub has no core, so the op stops at the guard) and leaves onboarding —
    // which, being the only window (as at launch, where onboarding opens
    // *instead of* a blank space), must hand off to a space rather than
    // leaving the user with no window at all.
    let stores = stub_stores(cx, |s| {
        s.backends = vec![eidola_app_core::BackendInfo {
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
        }];
    });
    let (window, view) = open_onboarding(cx, &stores);

    stores
        .backends
        .read_with(cx, |b, _| assert!(b.is_enabled("eidola")));
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.skip_account(window, cx));
    })
    .unwrap();
    cx.run_until_parked();

    stores.backends.read_with(cx, |b, _| {
        assert!(
            !b.is_enabled("eidola"),
            "skip must disable the eidola backend"
        );
    });
    // The onboarding window is gone, replaced by exactly one space window —
    // not zero (macOS would linger dock-only; Linux would quit outright on
    // the very choice to keep using the app on-device).
    let windows = cx.update(|cx| cx.windows());
    assert_eq!(windows.len(), 1, "skip must hand off to a space window");
    assert!(
        !windows.contains(&window),
        "skip must close the onboarding window"
    );
}

#[gpui::test]
fn onboarding_leaving_with_a_window_behind_opens_no_extra_space(cx: &mut TestAppContext) {
    // The hand-off is for the launch case, where onboarding is the only
    // window. Reached from the Eidola menu there is already a window behind
    // it, and leaving must not conjure a second blank space on top of it.
    let stores = stub_stores(cx, |_| {});
    let (behind, _) = open_space(cx, &stores, None);
    let (window, view) = open_onboarding(cx, &stores);
    assert_eq!(cx.update(|cx| cx.windows().len()), 2);

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.skip_account(window, cx));
    })
    .unwrap();
    cx.run_until_parked();

    let windows = cx.update(|cx| cx.windows());
    assert_eq!(
        windows,
        vec![behind],
        "leaving must close onboarding and leave the window behind it untouched"
    );
}

#[gpui::test]
fn onboarding_reveal_advances_and_is_idempotent(cx: &mut TestAppContext) {
    let stores = stub_stores(cx, |_| {});
    let (_w, view) = open_onboarding(cx, &stores);

    reveal(&view, cx, Slide::Pause, Slide::Tool);
    view.read_with(cx, |v, _| {
        assert_eq!(v.revealed(), vec![Slide::Pause, Slide::Tool]);
    });

    // Re-revealing the same next slide must not duplicate it.
    reveal(&view, cx, Slide::Pause, Slide::Tool);
    view.read_with(cx, |v, _| {
        assert_eq!(v.revealed(), vec![Slide::Pause, Slide::Tool]);
    });
}

#[gpui::test]
fn onboarding_back_arrow_glides_to_previous_slide(cx: &mut TestAppContext) {
    // Each slide past the first shows an up-arrow that glides to the previous
    // slide — the same `scroll_to_slide` path the arrow's click drives. The
    // resting offset (`pinned_y`) is set synchronously, so no frame-pumping.
    // A prior paint populates the per-slide child bounds the glide measures.
    let stores = stub_stores(cx, |_| {});
    let (window, view) = open_onboarding(cx, &stores);
    reveal(&view, cx, Slide::Pause, Slide::Tool);
    reveal(&view, cx, Slide::Tool, Slide::Control);
    draw_window(cx, window);

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            // Back from the Control slide (index 2) to the Tool slide (index 1):
            // the page pins above the top (a negative offset = the height of
            // slide 0, whatever it measured to).
            v.scroll_to_slide(1, window, cx);
            let one_up = v
                .pinned_y_for_test()
                .expect("glide must pin a resting offset");
            assert!(
                one_up < 0.0,
                "gliding to slide 1 must pin the page below the top, got {one_up}"
            );

            // Back all the way to the first slide pins exactly at the top.
            v.scroll_to_slide(0, window, cx);
            assert_eq!(
                v.pinned_y_for_test(),
                Some(0.0),
                "gliding to the first slide must pin at the top"
            );
        });
    })
    .unwrap();
}

#[gpui::test]
fn onboarding_slides_size_to_content_and_grow_on_small_windows(cx: &mut TestAppContext) {
    // The construction contract that replaced the old fixed-window-height
    // slides: each slide is *at least* one window tall (short slides read as a
    // full page) but **grows** past the window when its content is longer than
    // the window — so a long slide's prose and its trailing CTA are laid out in
    // full and reachable by scrolling, never clipped or overlapped. On a small
    // window the long narrative slides must therefore be taller than the window
    // while the short opening slide is exactly one window tall.
    let stores = stub_stores(cx, |_| {});
    // A deliberately small window so the long narrative slides overflow it (the
    // default test window is far taller than any slide's content).
    let stores2 = stores.clone();
    let (window, view) = cx.update(|cx| {
        gpui_component::init(cx);
        eidola_gui::theme::install(cx);
        let mut inner: Option<Entity<OnboardingView>> = None;
        let bounds = gpui::Bounds {
            origin: gpui::point(px(0.), px(0.)),
            size: gpui::size(px(680.), px(440.)),
        };
        let window = cx
            .open_window(
                WindowOptions {
                    window_bounds: Some(gpui::WindowBounds::Windowed(bounds)),
                    ..Default::default()
                },
                |window, cx| {
                    let view = cx.new(|cx| OnboardingView::new(stores2.clone(), window, cx));
                    inner = Some(view.clone());
                    cx.new(|cx| Root::new(view, window, cx))
                },
            )
            .expect("open small onboarding window");
        (window.into(), inner.expect("view"))
    });
    reveal(&view, cx, Slide::Pause, Slide::Tool);
    reveal(&view, cx, Slide::Tool, Slide::Control);
    reveal(&view, cx, Slide::Control, Slide::Responsibility);
    draw_window(cx, window);

    cx.update_window(window, |_, window, cx| {
        view.read_with(cx, |v, _| {
            let content_h = eidola_gui::chrome::content_height_for_test(window);
            let tops = v.slide_tops_for_test(window);
            assert_eq!(tops.len(), 4, "four slides revealed");
            assert_eq!(tops[0], 0.0, "the first slide starts at the content top");

            let height = |i: usize| tops[i + 1] - tops[i];
            // The short opening slide is exactly one window tall (min-height, no
            // growth).
            assert!(
                (height(0) - content_h).abs() < 1.0,
                "the short Pause slide should be exactly one window tall: \
                 height {} vs window {content_h}",
                height(0),
            );
            // The long narrative slides grow past the small window.
            assert!(
                height(1) > content_h + 1.0,
                "the long Tool slide should grow past the small window: \
                 height {} vs window {content_h}",
                height(1),
            );
            assert!(
                height(2) > content_h + 1.0,
                "the long Control slide should grow past the small window: \
                 height {} vs window {content_h}",
                height(2),
            );
        });
    })
    .unwrap();
}

#[gpui::test]
fn onboarding_rechoosing_branch_truncates_downstream(cx: &mut TestAppContext) {
    let stores = stub_stores(cx, |_| {});
    let (_w, view) = open_onboarding(cx, &stores);

    // Walk to the branch point and down the new-account path.
    reveal(&view, cx, Slide::Pause, Slide::Tool);
    reveal(&view, cx, Slide::Tool, Slide::Control);
    reveal(&view, cx, Slide::Control, Slide::Responsibility);
    reveal(&view, cx, Slide::Responsibility, Slide::GetStarted);
    reveal(&view, cx, Slide::GetStarted, Slide::CreateAccount);
    reveal(&view, cx, Slide::CreateAccount, Slide::NewAccount);
    view.read_with(cx, |v, _| {
        assert_eq!(v.revealed().last(), Some(&Slide::NewAccount));
    });

    // Re-choosing the *other* branch at "Get started" replaces the whole
    // downstream tail with the existing-account slide.
    reveal(&view, cx, Slide::GetStarted, Slide::ExistingAccount);
    view.read_with(cx, |v, _| {
        let revealed = v.revealed();
        assert_eq!(
            revealed,
            vec![
                Slide::Pause,
                Slide::Tool,
                Slide::Control,
                Slide::Responsibility,
                Slide::GetStarted,
                Slide::ExistingAccount,
            ],
            "the new-account slides must be gone after re-choosing"
        );
    });
}

#[gpui::test]
fn onboarding_verify_requires_both_fields(cx: &mut TestAppContext) {
    let stores = stub_stores(cx, |_| {});
    let (_w, view) = open_onboarding(cx, &stores);

    // Both inputs blank: verification refuses with a message, no request.
    view.update(cx, |v, cx| v.begin_verify(cx));
    view.read_with(cx, |v, _| {
        assert!(matches!(v.verify_result_for_test(), Some(Err(_))));
    });
}

#[gpui::test]
fn onboarding_verify_with_inputs_is_backend_gated_on_stub(cx: &mut TestAppContext) {
    let stores = stub_stores(cx, |_| {});
    let (window, view) = open_onboarding(cx, &stores);

    // Fill both credential inputs.
    let (id_input, secret_input) = view.read_with(cx, |v, _| v.existing_inputs_for_test());
    cx.update_window(window, |_, window, cx| {
        id_input.update(cx, |s, cx| s.set_value("acct-123", window, cx));
        secret_input.update(cx, |s, cx| s.set_value("shh", window, cx));
    })
    .unwrap();

    // With no backend the request is a no-op past the local guard: no error
    // message is produced (the empty-fields guard did not fire).
    view.update(cx, |v, cx| v.begin_verify(cx));
    cx.run_until_parked();
    view.read_with(cx, |v, _| {
        assert!(
            v.verify_result_for_test().is_none(),
            "a stub backend yields no verification result, and no field error"
        );
    });
}

#[gpui::test]
fn onboarding_create_on_stub_is_safe_noop(cx: &mut TestAppContext) {
    let stores = stub_stores(cx, |_| {});
    let (_w, view) = open_onboarding(cx, &stores);

    reveal(&view, cx, Slide::Pause, Slide::Tool);
    reveal(&view, cx, Slide::Tool, Slide::Control);
    reveal(&view, cx, Slide::Control, Slide::Responsibility);
    reveal(&view, cx, Slide::Responsibility, Slide::GetStarted);
    reveal(&view, cx, Slide::GetStarted, Slide::CreateAccount);

    // No backend: create marks in-flight and stops at the guard — it neither
    // reveals the next slide nor produces credentials, and does not panic.
    view.update(cx, |v, cx| v.begin_create(cx));
    cx.run_until_parked();
    view.read_with(cx, |v, _| {
        assert_eq!(v.revealed().last(), Some(&Slide::CreateAccount));
        assert!(v.created_for_test().is_none());
    });
}

// ---------------------------------------------------------------------------
// Read-only post selection — REGRESSION (content-dependent selection failure).
// ---------------------------------------------------------------------------

/// REGRESSION: pointer selection on a post whose markdown is *non-canonical*
/// for the editor (a table's `\n`-separated pipe rows, a heading tightly
/// followed by its paragraph, a paragraph directly followed by a blockquote —
/// shapes model output routinely produces) used to be impossible. The click's
/// `SetSelection` ran the editable pipeline's `enforce_invariants`, rewriting
/// the read-only buffer away from the post's persisted content; the next
/// frame's `sync_bodies` saw the divergence and re-seeded the editor
/// (`set_value` → selection reset to `Cursor(0)`), so every drag step was
/// wiped a frame later — while the autoscroll path, which re-extends *after*
/// the reset within the same render, appeared to work. Two fixes hold the
/// line: the editor's read-only dispatch never rewrites the buffer
/// (`update::update_readonly`), and `sync_bodies` re-seeds only when the
/// *post's* content changes (`body_seeds`), never because the live buffer
/// differs. The fixture mirrors the reported space's shape: a branched root
/// whose spine reply carries the heavy markdown (the strip is incidental —
/// the failure was content-dependent — but keeping it exercises selection
/// inside a branch scroller too).
#[gpui::test]
fn space_readonly_selection_sticks_on_noncanonical_post(cx: &mut TestAppContext) {
    let noncanonical = "### heading\nbody text directly under the heading\n\n\
        **Note:**\n> a quoted line right after a paragraph\n\n\
        | left | right |\n| --- | --- |\n| cell one | cell two |";

    let post =
        |action_id: &str, parent: Option<&str>, depth: usize, is_branch: bool, text: &str| {
            let mut n = fixture_user_post(action_id, text);
            n.parent_action_id = parent.map(String::from);
            n.relation = parent.map(|_| "reply".to_string());
            n.depth = depth;
            n.is_branch = is_branch;
            n
        };
    let nodes = vec![
        post(
            "s1",
            None,
            0,
            false,
            "a short root question\n\nwith a second paragraph of ordinary prose",
        ),
        post("s2", Some("s1"), 0, false, noncanonical),
        post("s3", Some("s1"), 1, true, "a branch aside"),
        post("s4", Some("s3"), 1, false, "a branch reply"),
    ];

    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("sel".into()));
    view.update(cx, |v, cx| {
        v.space()
            .update(cx, |s, cx| s.set_post_tree_for_test(nodes, cx));
    });
    cx.run_until_parked();

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(680.)));
    vcx.run_until_parked();

    // Aim the drag from the painted geometry of s2's first block, so the test
    // doesn't hardcode layout metrics — and assert the target is actually
    // within the viewport (a fully-in-view drag, the reported case).
    let (x, y, h) = view
        .read_with(&vcx, |v, cx| {
            v.post_body_editor_for_test("s2").and_then(|e| {
                e.read(cx)
                    .debug_line_geometry()
                    .first()
                    .and_then(|(_, lines)| lines.first().copied())
            })
        })
        .expect("s2's first painted line");
    assert!(
        y > 0.0 && y + 30.0 < 680.0,
        "fixture drift: s2's first line must be in view (y = {y})"
    );
    let mid = y + h.min(24.0) / 2.0;
    let start = gpui::point(px(x + 30.0), px(mid));
    let end = gpui::point(px(x + 180.0), px(mid + 25.0));

    vcx.simulate_event(gpui::MouseDownEvent {
        button: gpui::MouseButton::Left,
        position: start,
        modifiers: Modifiers::default(),
        click_count: 1,
        first_mouse: false,
    });
    vcx.simulate_event(gpui::MouseMoveEvent {
        position: end,
        pressed_button: Some(gpui::MouseButton::Left),
        modifiers: Modifiers::default(),
    });
    vcx.simulate_event(gpui::MouseUpEvent {
        button: gpui::MouseButton::Left,
        position: end,
        modifiers: Modifiers::default(),
        click_count: 1,
    });
    // Let further frames run: the old bug wiped the selection on the *next*
    // render (the sync_bodies re-seed), so surviving parked frames is the
    // regression's teeth.
    vcx.run_until_parked();

    view.read_with(&vcx, |v, cx| {
        let editor = v.post_body_editor_for_test("s2").expect("s2's editor");
        let e = editor.read(cx);
        assert_eq!(
            e.value(),
            noncanonical,
            "the read-only buffer must stay byte-identical to the post"
        );
        let sel = e.selection();
        assert!(
            sel.upper_bound() > sel.lower_bound(),
            "the in-view drag must leave a selection that survives subsequent \
             frames, got {sel:?}"
        );
    });
}

/// REGRESSION (task 32): a press on the transparent title band — the gesture
/// that drags the *window* — must not also land in the post scrolled under it.
/// It did: gpui's hit test reports every hitbox under the cursor, and the band
/// neither blocked the mouse nor painted after the page, so the press reached
/// the post's `MarkdownEditor` and the window move dragged out a text
/// selection with it. The fix is at the shared chrome layer
/// (`titlebar::make_draggable` → `block_mouse_except_scroll`, which only
/// suppresses what was painted *before* the strip) plus painting the space
/// view's band after the page it covers.
#[gpui::test]
fn space_title_band_press_does_not_select_the_post_beneath(cx: &mut TestAppContext) {
    let long = (1..=14)
        .map(|i| {
            format!(
                "Paragraph {i}. Sunlight is a fairly even mix across the visible spectrum, \
                 and as it crosses the atmosphere it meets molecules far smaller than its \
                 wavelength."
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    let nodes = vec![fixture_user_post("s1", &long)];

    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("sel".into()));
    view.update(cx, |v, cx| {
        v.space()
            .update(cx, |s, cx| s.set_post_tree_for_test(nodes, cx));
    });
    cx.run_until_parked();

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(680.)));
    vcx.run_until_parked();
    // Scroll the post up so its text runs under the 36px title band.
    view.update(&mut vcx, |v, cx| {
        v.scroll_page_by_for_test(-240.);
        cx.notify();
    });
    vcx.run_until_parked();

    // Aim at a point inside the band that real geometry says is painted post
    // text, rather than hardcoding metrics: `BAND_Y` sits within the 36px
    // reserve, and some painted line must span it.
    const BAND_Y: f32 = 18.0;
    let x = view
        .read_with(&vcx, |v, cx| {
            v.post_body_editor_for_test("s1").and_then(|e| {
                e.read(cx)
                    .debug_line_geometry()
                    .iter()
                    .flat_map(|(_, lines)| lines.iter().copied())
                    .find(|(_, y, h)| *y <= BAND_Y && y + h > BAND_Y)
                    .map(|(x, _, _)| x)
            })
        })
        .unwrap_or_else(|| {
            let g = view.read_with(&vcx, |v, cx| {
                v.post_body_editor_for_test("s1")
                    .map(|e| e.read(cx).debug_line_geometry())
            });
            panic!("fixture drift: no post text under the title band; geometry = {g:?}")
        });

    let start = gpui::point(px(x + 30.0), px(BAND_Y));
    let end = gpui::point(px(x + 200.0), px(300.0));
    vcx.simulate_event(gpui::MouseDownEvent {
        button: gpui::MouseButton::Left,
        position: start,
        modifiers: Modifiers::default(),
        click_count: 1,
        first_mouse: false,
    });
    vcx.simulate_event(gpui::MouseMoveEvent {
        position: end,
        pressed_button: Some(gpui::MouseButton::Left),
        modifiers: Modifiers::default(),
    });
    vcx.simulate_event(gpui::MouseUpEvent {
        button: gpui::MouseButton::Left,
        position: end,
        modifiers: Modifiers::default(),
        click_count: 1,
    });
    vcx.run_until_parked();

    view.read_with(&vcx, |v, cx| {
        let editor = v.post_body_editor_for_test("s1").expect("s1's editor");
        let sel = editor.read(cx).selection();
        assert_eq!(
            sel.lower_bound(),
            sel.upper_bound(),
            "a drag begun in the title band must leave the post unselected, got {sel:?}"
        );
    });
}

// ---------------------------------------------------------------------------
// Participants v1 — the Participants view + Space Templates pane (real core)
// ---------------------------------------------------------------------------
//
// These drive the views against a *real* tempdir-backed `AppCore` (the CRUD is
// local-DB work, so an unreachable base URL is fine) and `run_until_parked` to
// let each write-then-relist land — the established real-core idiom (see
// `record_refresh_supersedes_in_flight_fetch` and `tests/stores.rs`). Each
// joins the runtime's live tasks before returning so the last `Arc<AppCore>`
// always drops on the test thread.

/// A real, tempdir-backed core with an unreachable base URL, plus a freshly
/// created space (born from the default template: "You" + the seeded agent).
fn participants_scene(
    cx: &mut TestAppContext,
) -> (Stores, std::sync::Arc<AppCore>, tempfile::TempDir, String) {
    cx.executor().allow_parking();
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    let dir = tempfile::tempdir().unwrap();
    let core = std::sync::Arc::new(
        AppCore::new(dir.path().to_path_buf(), dir.path().join("data")).expect("open core"),
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
    (stores, core, dir, space)
}

fn drain_runtime(core: &std::sync::Arc<AppCore>) {
    while core.runtime().metrics().num_alive_tasks() > 0 {
        std::thread::yield_now();
    }
}

/// Poll `run_until_parked` until `pred` holds — a write-then-relist round-trips
/// through the tokio runtime, which `run_until_parked` alone can return before
/// (the tokio result lands after the gpui task parks). The DB op settles in
/// milliseconds; ~10s ceiling. (Mirrors `tests/stores.rs::wait_for_backends`.)
fn wait_until(
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

fn participant_labels(stores: &Stores, space: &str, cx: &mut TestAppContext) -> Vec<String> {
    stores.participants.read_with(cx, |s, _| {
        s.list(space).iter().map(|p| p.label.clone()).collect()
    })
}

/// Open a space window on a real space with the inspector already open — the
/// Participants section's live surface (wave 26.3).
fn open_participants_inspector(
    cx: &mut TestAppContext,
    stores: &Stores,
    space: &str,
) -> (AnyWindowHandle, Entity<SpaceView>) {
    let (window, view) = open_space(cx, stores, Some(space.to_string()));
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.set_inspector_open_for_test(true, window, cx));
    })
    .unwrap();
    (window, view)
}

#[gpui::test]
fn inspector_participants_add_and_remove(cx: &mut TestAppContext) {
    let (stores, core, _dir, space) = participants_scene(cx);
    let (window, view) = open_participants_inspector(cx, &stores, &space);
    wait_until(cx, "participants load", |cx| {
        participant_labels(&stores, &space, cx).len() == 2
    });

    // Add a participant: open the form, set its name, save.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_begin_add_participant(window, cx));
    })
    .unwrap();
    // The form arrives prefilled with the shared default charter — the same
    // starting point the Templates pane offers a new agent.
    assert_eq!(
        view.read_with(cx, |v, cx| v.inspector_adding_prompt(cx))
            .as_deref(),
        Some(eidola_gui::participants::DEFAULT_AGENT_SYSTEM_PROMPT),
        "a new participant starts from the shared default system prompt"
    );
    let label = view
        .read_with(cx, |v, _| v.inspector_adding_label_state())
        .expect("add form open");
    cx.update_window(window, |_, window, cx| {
        label.update(cx, |s, cx| s.set_value("Reviewer", window, cx));
    })
    .unwrap();
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_save_add_participant(window, cx));
    })
    .unwrap();
    wait_until(cx, "participant added", |cx| {
        participant_labels(&stores, &space, cx).contains(&"Reviewer".to_string())
    });
    assert_eq!(participant_labels(&stores, &space, cx).len(), 3);

    // The prompt it was created with round-trips into the durable row.
    let reviewer = stores.participants.read_with(cx, |s, _| {
        s.list(&space)
            .iter()
            .find(|p| p.label == "Reviewer")
            .unwrap()
            .clone()
    });
    assert_eq!(
        reviewer.system_prompt.as_deref(),
        Some(eidola_gui::participants::DEFAULT_AGENT_SYSTEM_PROMPT),
        "the default charter is what was written, not just what was shown"
    );

    // Remove it, from its own disclosure.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.inspector_remove_participant(&reviewer.id, window, cx)
        });
    })
    .unwrap();
    wait_until(cx, "participant removed", |cx| {
        participant_labels(&stores, &space, cx).len() == 2
    });

    drain_runtime(&core);
}

/// The system prompt is editable where the participant lives: open a member's
/// disclosure, rewrite its charter, save, and the durable row moves.
/// The whole point of creating a space when its window opens: the space's
/// configuration surface works from birth. A ⌘N window with **zero posts**
/// renders the Participants section over a real roster and edits it — the
/// membership exists because the space was instantiated from the default
/// template when the window opened, not when a first message was sent.
#[gpui::test]
fn inspector_participants_work_on_a_fresh_space_with_no_posts(cx: &mut TestAppContext) {
    let (stores, core, _dir, _seeded) = participants_scene(cx);

    // ⌘N — no space id; the window mints one and opens on it.
    let (window, view) = open_space(cx, &stores, None);
    let space = view.read_with(cx, |v, cx| v.space().read(cx).id().to_string());
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.set_inspector_open_for_test(true, window, cx));
    })
    .unwrap();

    // The window's own reads are issued in the frame it opened, which can
    // precede the insert they are reading behind — the instantiation's
    // announcements are what take an early "empty" answer back (app-core's
    // `create_space_with_id_emits_space_index_under_the_given_id` pins that it
    // emits them; here they arrive through the dispatch seam this file uses).
    wait_until(cx, "the space's row commits", |_| {
        core.runtime()
            .block_on(core.list_spaces(false))
            .is_ok_and(|spaces| spaces.iter().any(|s| s.id == space))
    });
    cx.update(|cx| {
        stores::dispatch_change_for_test(&stores, Some(Change::Participants), cx);
        stores::dispatch_change_for_test(&stores, Some(Change::Space(space.clone())), cx);
    });

    wait_until(cx, "the new space's roster loads", |cx| {
        participant_labels(&stores, &space, cx).len() == 2
    });
    assert!(
        view.read_with(cx, |v, cx| v.space().read(cx).messages().is_empty()),
        "and it has not been posted into"
    );

    // The section is renderable — it used to be withheld entirely here — and
    // its rows are the template's membership.
    draw_frame(cx, window);
    let agent = stores.participants.read_with(cx, |s, _| {
        s.list(&space)
            .iter()
            .find(|p| p.kind == "agent")
            .expect("the default template seeds one agent")
            .id
            .clone()
    });

    // …and it edits: the disclosure is the editor, and Save writes through.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.inspector_toggle_participant(&agent, window, cx)
        });
    })
    .unwrap();
    let name = view
        .read_with(cx, |v, _| v.inspector_editing_label_state())
        .expect("the disclosure is the editor");
    cx.update_window(window, |_, window, cx| {
        name.update(cx, |s, cx| s.set_value("Mara", window, cx));
    })
    .unwrap();
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_save_participant_edit(window, cx));
    })
    .unwrap();
    wait_until(cx, "the rename persists on a post-less space", |cx| {
        participant_labels(&stores, &space, cx).contains(&"Mara".to_string())
    });

    // The Space section's settings rows are live too — the space really is in
    // the database, so its settings are readable.
    wait_until(cx, "the space's settings load", |cx| {
        view.read_with(cx, |v, cx| v.inspector_cascade_for_test(cx))
            .is_some()
    });
}

#[gpui::test]
fn inspector_participant_system_prompt_round_trips(cx: &mut TestAppContext) {
    let (stores, core, _dir, space) = participants_scene(cx);
    let (window, view) = open_participants_inspector(cx, &stores, &space);
    wait_until(cx, "participants load", |cx| {
        participant_labels(&stores, &space, cx).len() == 2
    });

    let agent = stores.participants.read_with(cx, |s, _| {
        s.list(&space)
            .iter()
            .find(|p| p.kind == "agent")
            .expect("the default template seeds one agent")
            .id
            .clone()
    });
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.inspector_toggle_participant(&agent, window, cx)
        });
    })
    .unwrap();
    let prompt = view
        .read_with(cx, |v, _| v.inspector_editing_prompt_state())
        .expect("the disclosure is the editor");
    cx.update_window(window, |_, window, cx| {
        prompt.update(cx, |s, cx| {
            s.set_value("Answer only in questions.", window, cx)
        });
    })
    .unwrap();
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_save_participant_edit(window, cx));
    })
    .unwrap();
    wait_until(cx, "system prompt persisted", |cx| {
        stores.participants.read_with(cx, |s, _| {
            s.list(&space)
                .iter()
                .find(|p| p.id == agent)
                .and_then(|p| p.system_prompt.clone())
                .as_deref()
                == Some("Answer only in questions.")
        })
    });

    // Re-opening the disclosure seeds the field from what was stored.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.inspector_toggle_participant(&agent, window, cx)
        });
    })
    .unwrap();
    let reopened = view
        .read_with(cx, |v, _| v.inspector_editing_prompt_state())
        .expect("reopened");
    assert_eq!(
        reopened.read_with(cx, |s, _| s.value().to_string()),
        "Answer only in questions."
    );

    drain_runtime(&core);
}

/// The headline fork: editing a **referenced global** ("You") writes either the
/// shared config (edit everywhere) or a per-space override (override here). The
/// view routes to the right store method per its mode.
#[gpui::test]
fn inspector_participants_override_vs_edit_everywhere(cx: &mut TestAppContext) {
    let (stores, core, _dir, space) = participants_scene(cx);
    let (window, view) = open_participants_inspector(cx, &stores, &space);
    wait_until(cx, "participants load", |cx| {
        participant_labels(&stores, &space, cx).len() == 2
    });

    let you = eidola_app_core::HUMAN_PARTICIPANT_ID;

    // Override here: a referenced global defaults to the this-space-only mode.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_toggle_participant(you, window, cx));
    })
    .unwrap();
    assert_eq!(
        view.read_with(cx, |v, _| v.inspector_editing_mode()),
        Some(EditMode::OverrideHere)
    );
    let label = view
        .read_with(cx, |v, _| v.inspector_editing_label_state())
        .unwrap();
    cx.update_window(window, |_, window, cx| {
        label.update(cx, |s, cx| s.set_value("Me", window, cx));
    })
    .unwrap();
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_save_participant_edit(window, cx));
    })
    .unwrap();
    wait_until(cx, "override applied", |cx| {
        stores.participants.read_with(cx, |s, _| {
            s.list(&space)
                .iter()
                .find(|p| p.id == you)
                .map(|p| p.label == "Me")
                .unwrap_or(false)
        })
    });

    let (eff, base) = stores.participants.read_with(cx, |s, _| {
        let p = s.list(&space).iter().find(|p| p.id == you).unwrap().clone();
        (p.label.clone(), p.reference.unwrap().base_label)
    });
    assert_eq!(eff, "Me", "override changed the effective label");
    assert_ne!(base, "Me", "the shared global is untouched by an override");

    // Edit everywhere: switch mode, change the name, save — now the base moves.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_toggle_participant(you, window, cx));
        view.update(cx, |v, cx| {
            v.inspector_set_edit_mode(EditMode::Everywhere, window, cx)
        });
    })
    .unwrap();
    assert_eq!(
        view.read_with(cx, |v, _| v.inspector_editing_mode()),
        Some(EditMode::Everywhere)
    );
    let label = view
        .read_with(cx, |v, _| v.inspector_editing_label_state())
        .unwrap();
    cx.update_window(window, |_, window, cx| {
        label.update(cx, |s, cx| s.set_value("Myself", window, cx));
    })
    .unwrap();
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_save_participant_edit(window, cx));
    })
    .unwrap();
    wait_until(cx, "edit-everywhere applied", |cx| {
        stores.participants.read_with(cx, |s, _| {
            s.list(&space)
                .iter()
                .find(|p| p.id == you)
                .and_then(|p| p.reference.as_ref())
                .map(|r| r.base_label == "Myself")
                .unwrap_or(false)
        })
    });

    drain_runtime(&core);
}

/// The **promote affordance** (task 36): "Share this agent…" on a space-owned
/// agent's disclosure turns it into a shared identity *in place*.
///
/// Two things are asserted because both are the feature: the verb asks first
/// (sharing is one-way — "Not now" leaves the roster exactly as it was), and the
/// share itself moves ownership without moving configuration. The row comes back
/// **referenced** with the same id and the same effective persona — which is
/// also what makes the "shared" tag and the override fork appear with no view
/// code at all, since both are read off the store's own re-list.
#[gpui::test]
fn inspector_sharing_an_agent_promotes_it_in_place(cx: &mut TestAppContext) {
    let (stores, core, _dir, space) = participants_scene(cx);
    let (window, view) = open_participants_inspector(cx, &stores, &space);
    wait_until(cx, "participants load", |cx| {
        participant_labels(&stores, &space, cx).len() == 2
    });

    let before = stores.participants.read_with(cx, |s, _| {
        s.list(&space)
            .iter()
            .find(|p| p.kind == "agent")
            .expect("the default template seeds one agent")
            .clone()
    });
    assert_eq!(
        before.source, "owned",
        "the seeded agent starts space-owned"
    );
    let agent = before.id.clone();

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.inspector_toggle_participant(&agent, window, cx)
        });
    })
    .unwrap();

    // The confirmation is armed, then stood down — nothing is written.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_begin_promote(window, cx));
    })
    .unwrap();
    assert!(view.read_with(cx, |v, _| v.inspector_promote_confirming()));
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_cancel_promote(window, cx));
    })
    .unwrap();
    assert!(!view.read_with(cx, |v, _| v.inspector_promote_confirming()));
    cx.run_until_parked();
    assert_eq!(
        stores.participants.read_with(cx, |s, _| {
            s.list(&space)
                .iter()
                .find(|p| p.id == agent)
                .map(|p| p.source.clone())
        }),
        Some("owned".to_string()),
        "declining the confirmation shares nothing"
    );

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_begin_promote(window, cx));
    })
    .unwrap();
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_confirm_promote(window, cx));
    })
    .unwrap();
    // What the editor was editing has changed shape, so it closes.
    assert!(
        view.read_with(cx, |v, _| v.inspector_editing_participant().is_none()),
        "the disclosure closes on a share"
    );

    wait_until(cx, "the share lands", |cx| {
        stores.participants.read_with(cx, |s, _| {
            s.list(&space)
                .iter()
                .any(|p| p.id == agent && p.source == "referenced")
        })
    });
    let after = stores.participants.read_with(cx, |s, _| {
        s.list(&space)
            .iter()
            .find(|p| p.id == agent)
            .expect("the shared agent is still a member")
            .clone()
    });
    assert_eq!(after.scope, "global");
    assert_eq!(after.label, before.label);
    assert_eq!(after.model_ref, before.model_ref);
    assert_eq!(after.system_prompt, before.system_prompt);
    assert_eq!(after.notify_policy, before.notify_policy);
    assert!(
        after.reference.is_some(),
        "a referenced global carries the override fork's detail"
    );
    assert!(
        stores
            .participants
            .read_with(cx, |s, _| s.op_errors_for(&space).is_empty()),
        "the share was accepted"
    );

    drain_runtime(&core);
}

/// **The confirmation is about the persona on screen.** "Share this agent…" is
/// pressed from inside the open editor, whose fields stay visible behind the
/// confirmation — so "keeps this space's persona exactly as it is" is read
/// against the values the reader is looking at, which are the ones they just
/// typed. Sharing therefore saves the draft and *then* promotes; discarding it
/// would make the reassurance false about everything the reader could see
/// (Codex review, PR #279).
#[gpui::test]
fn inspector_sharing_a_dirty_editor_shares_the_persona_on_screen(cx: &mut TestAppContext) {
    let (stores, core, _dir, space) = participants_scene(cx);
    let (window, view) = open_participants_inspector(cx, &stores, &space);
    wait_until(cx, "participants load", |cx| {
        participant_labels(&stores, &space, cx).len() == 2
    });

    let before = stores.participants.read_with(cx, |s, _| {
        s.list(&space)
            .iter()
            .find(|p| p.kind == "agent")
            .expect("the default template seeds one agent")
            .clone()
    });
    let agent = before.id.clone();
    assert_ne!(
        before.notify_policy, "explicit",
        "the notify change below has to be a change"
    );

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.inspector_toggle_participant(&agent, window, cx)
        });
    })
    .unwrap();
    let label = view
        .read_with(cx, |v, _| v.inspector_editing_label_state())
        .expect("the disclosure is the editor");
    let prompt = view
        .read_with(cx, |v, _| v.inspector_editing_prompt_state())
        .expect("the disclosure is the editor");
    cx.update_window(window, |_, window, cx| {
        label.update(cx, |s, cx| s.set_value("Cartographer", window, cx));
        prompt.update(cx, |s, cx| {
            s.set_value("Draw the map before arguing about it.", window, cx)
        });
    })
    .unwrap();
    view.update(cx, |v, cx| {
        v.inspector_select_participant_model("gemma4-31b@eidola", cx);
        v.inspector_set_edit_notify("explicit", cx);
    });

    // Share, without touching Save first.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_begin_promote(window, cx));
    })
    .unwrap();
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_confirm_promote(window, cx));
    })
    .unwrap();

    wait_until(cx, "the share lands", |cx| {
        stores.participants.read_with(cx, |s, _| {
            s.list(&space)
                .iter()
                .any(|p| p.id == agent && p.source == "referenced")
        })
    });
    let after = stores.participants.read_with(cx, |s, _| {
        s.list(&space)
            .iter()
            .find(|p| p.id == agent)
            .expect("the shared agent is still a member")
            .clone()
    });
    assert_eq!(after.scope, "global");
    assert_eq!(
        after.label, "Cartographer",
        "the name on screen is the name that was shared"
    );
    assert_eq!(
        after.system_prompt.as_deref(),
        Some("Draw the map before arguing about it."),
        "the charter on screen is the charter that was shared"
    );
    assert_eq!(after.model_ref.as_deref(), Some("gemma4-31b@eidola"));
    assert_eq!(after.notify_policy, "explicit");
    // The membership keeps NULL overrides, so the persona above is the shared
    // agent's own — not this space papering over a stale one.
    let reference = after.reference.expect("a referenced global carries detail");
    assert_eq!(reference.base_label, "Cartographer");
    assert_eq!(
        reference.base_system_prompt.as_deref(),
        Some("Draw the map before arguing about it.")
    );
    assert!(
        reference.override_label.is_none() && reference.override_system_prompt.is_none(),
        "promotion writes no per-space override"
    );
    assert!(
        stores
            .participants
            .read_with(cx, |s, _| s.op_errors_for(&space).is_empty()),
        "the share was accepted"
    );

    drain_runtime(&core);
}

/// The inspector's half of the handback rule (Codex review, PR #279): the
/// roster-driven retire has always restored the keyboard, but the verbs a reader
/// actually presses — Cancel, Save, and the ones that close a form on the way to
/// doing something else — dropped their focused fields and left the window
/// holding a dead handle.
#[gpui::test]
fn inspector_hands_the_keyboard_back_when_a_form_closes(cx: &mut TestAppContext) {
    let (stores, core, _dir, space) = participants_scene(cx);
    let (window, view) = open_participants_inspector(cx, &stores, &space);
    wait_until(cx, "participants load", |cx| {
        participant_labels(&stores, &space, cx).len() == 2
    });
    let agent = stores.participants.read_with(cx, |s, _| {
        s.list(&space)
            .iter()
            .find(|p| p.kind == "agent")
            .expect("the default template seeds one agent")
            .id
            .clone()
    });

    let view_holds_the_keyboard = |cx: &mut TestAppContext| {
        cx.update_window(window, |_, window, cx| {
            view.read(cx).focus_handle().is_focused(window)
        })
        .unwrap()
    };
    // Opening a disclosure focuses its name field (the reveal-focuses rule), so
    // every case below starts with the form holding the keyboard.
    let open_the_disclosure = |cx: &mut TestAppContext| {
        cx.update_window(window, |_, window, cx| {
            view.update(cx, |v, cx| {
                v.inspector_toggle_participant(&agent, window, cx)
            });
        })
        .unwrap();
        assert!(!view_holds_the_keyboard(cx), "the name field has it");
    };

    open_the_disclosure(cx);
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_cancel_participant_edit(window, cx));
    })
    .unwrap();
    assert!(view_holds_the_keyboard(cx), "Cancel hands it back");

    // The same containment question here: focus inside the disclosure that is
    // not one of its inputs still owes the keyboard back.
    open_the_disclosure(cx);
    let subtree = view
        .read_with(cx, |v, _| v.inspector_editing_focus_handle())
        .expect("the disclosure is open");
    cx.update_window(window, |_, window, cx| {
        window.focus(&subtree, cx);
    })
    .unwrap();
    assert!(!view_holds_the_keyboard(cx), "the form subtree holds it");
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_cancel_participant_edit(window, cx));
    })
    .unwrap();
    assert!(
        view_holds_the_keyboard(cx),
        "containment, not an enumeration of inputs"
    );

    open_the_disclosure(cx);
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_save_participant_edit(window, cx));
    })
    .unwrap();
    assert!(view_holds_the_keyboard(cx), "Save hands it back");

    // **The share confirmation's buttons are tab stops too**, and arming or
    // standing down unmounts the one that was pressed while the form itself
    // survives — so the keyboard goes back *to the form*, where the reader
    // still is (Codex review, PR #279).
    open_the_disclosure(cx);
    let form = view
        .read_with(cx, |v, _| v.inspector_editing_focus_handle())
        .expect("the disclosure is open");
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_begin_promote(window, cx));
    })
    .unwrap();
    assert!(
        cx.update_window(window, |_, window, _cx| form.is_focused(window))
            .unwrap(),
        "arming the share leaves the keyboard on the form it replaced a button in"
    );
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_cancel_promote(window, cx));
    })
    .unwrap();
    assert!(
        cx.update_window(window, |_, window, _cx| form.is_focused(window))
            .unwrap(),
        "and so does standing it down"
    );
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_cancel_participant_edit(window, cx));
    })
    .unwrap();

    // The inspector's own sync path: a row that leaves the roster takes the
    // whole editor — **including an armed share confirmation** — and the
    // round-5 handback fires for it, because the question is containment of the
    // form and the confirm lives inside it (Codex review, PR #279 asked; this
    // is the answer, pinned).
    open_the_disclosure(cx);
    let form = view
        .read_with(cx, |v, _| v.inspector_editing_focus_handle())
        .expect("the disclosure is open");
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_begin_promote(window, cx));
        window.focus(&form, cx);
    })
    .unwrap();
    assert!(!view_holds_the_keyboard(cx), "the form holds the keyboard");
    stores
        .participants
        .update(cx, |s, cx| s.remove(space.clone(), agent.clone(), cx));
    wait_until(cx, "the row leaves the roster", |cx| {
        stores
            .participants
            .read_with(cx, |s, _| !s.list(&space).iter().any(|p| p.id == agent))
    });
    draw_window(cx, window);
    assert!(
        view.read_with(cx, |v, _| v.inspector_editing_participant().is_none()),
        "the editor went with its row"
    );
    assert!(
        !view.read_with(cx, |v, _| v.inspector_promote_confirming()),
        "and the armed confirmation with it"
    );
    assert!(
        view_holds_the_keyboard(cx),
        "the roster-driven unmount hands the keyboard back too"
    );

    // The add form is the same shape, and its Cancel is the same door.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_begin_add_participant(window, cx));
    })
    .unwrap();
    assert!(
        !view_holds_the_keyboard(cx),
        "the add form's name field has it"
    );
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_cancel_add_participant(window, cx));
    })
    .unwrap();
    assert!(
        view_holds_the_keyboard(cx),
        "the add form hands it back too"
    );

    drain_runtime(&core);
}

/// **An open editor is valid only while its row still answers to the shape it
/// was seeded from** (Codex review, PR #279).
///
/// Promotion keeps the participant's id, so the roster's re-list carries the
/// same row — with `source` flipped to `referenced`. An editor seeded on the
/// *owned* shape then goes on painting the owned form: no Everyone/This-space
/// fork, a live "Share this agent…" over an agent that already is one, and — the
/// harm — a **Save** that routes to `update_everywhere`, publishing the draft to
/// every space the shared agent joins without ever showing the reader the choice
/// they were entitled to make.
#[gpui::test]
fn inspector_editor_retires_when_its_row_stops_being_what_it_was_seeded_as(
    cx: &mut TestAppContext,
) {
    let (stores, core, _dir, space) = participants_scene(cx);
    let (window, view) = open_participants_inspector(cx, &stores, &space);
    wait_until(cx, "participants load", |cx| {
        participant_labels(&stores, &space, cx).len() == 2
    });

    let before = stores.participants.read_with(cx, |s, _| {
        s.list(&space)
            .iter()
            .find(|p| p.kind == "agent")
            .expect("the default template seeds one agent")
            .clone()
    });
    let agent = before.id.clone();
    assert_eq!(before.source, "owned");

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.inspector_toggle_participant(&agent, window, cx)
        });
    })
    .unwrap();
    let label = view
        .read_with(cx, |v, _| v.inspector_editing_label_state())
        .expect("the disclosure is the editor");
    cx.update_window(window, |_, window, cx| {
        label.update(cx, |s, cx| {
            s.set_value("Published to every space", window, cx)
        });
    })
    .unwrap();
    // With the share armed, too: an irreversible verb aimed at a row that has
    // since become something else must not survive either.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_begin_promote(window, cx));
    })
    .unwrap();

    // Another window shares the same agent. Its id does not move — only what it
    // *is* — so the roster-leave rule alone sees nothing.
    core.runtime()
        .block_on(core.promote_participant(agent.clone(), None, None))
        .expect("the other window's share");
    stores
        .participants
        .update(cx, |s, cx| s.refresh(space.clone(), cx));
    wait_until(cx, "the roster re-lists it as shared", |cx| {
        stores.participants.read_with(cx, |s, _| {
            s.list(&space)
                .iter()
                .any(|p| p.id == agent && p.source == "referenced")
        })
    });
    draw_window(cx, window);

    assert!(
        view.read_with(cx, |v, _| v.inspector_editing_participant().is_none()),
        "an editor seeded on an owned agent must not go on describing a shared one"
    );
    assert!(
        !view.read_with(cx, |v, _| v.inspector_promote_confirming()),
        "the armed share dies with the editor that owned it"
    );

    // The verb the stale editor was still offering writes nothing now. Read the
    // database, not the cached roster: the claim is about what was durably
    // published, and `drain_runtime` is what makes "no write was issued" and "a
    // write landed" distinguishable in one bounded step.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_save_participant_edit(window, cx));
    })
    .unwrap();
    cx.run_until_parked();
    drain_runtime(&core);
    cx.run_until_parked();
    let durable = core
        .runtime()
        .block_on(core.list_space_participants(space.clone()))
        .expect("participants")
        .into_iter()
        .find(|p| p.id == agent)
        .expect("still a member");
    assert_eq!(
        durable.label, before.label,
        "the draft was never published to every space that follows the shared row"
    );

    // Re-opening seeds the editor the row now deserves: the fork, in its safe
    // mode, with nothing armed.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.inspector_toggle_participant(&agent, window, cx)
        });
    })
    .unwrap();
    assert_eq!(
        view.read_with(cx, |v, _| v.inspector_editing_mode()),
        Some(EditMode::OverrideHere),
        "a referenced global opens in this-space-only"
    );

    drain_runtime(&core);
}

/// A shared agent's **notebook** is a real space, and the Library is the list of
/// the human's conversations — so promotion must not put a new row in it.
/// `AppCore::list_spaces` excludes notebooks unconditionally; this is the GUI's
/// end of that promise, read where a person would read it.
#[gpui::test]
fn a_shared_agents_notebook_stays_out_of_the_library(cx: &mut TestAppContext) {
    let (stores, core, _dir, space) = participants_scene(cx);
    let (_window, _view) = open_view(cx, |window, cx| {
        cx.new(|cx| LibraryView::new(stores.clone(), window, cx))
    });
    stores
        .participants
        .update(cx, |s, cx| s.ensure(space.clone(), cx));
    wait_until(cx, "participants load", |cx| {
        participant_labels(&stores, &space, cx).len() == 2
    });
    wait_until(cx, "the library lists the space", |cx| {
        stores.spaces.read_with(cx, |s, _| s.list().len() == 1)
    });

    let agent = stores.participants.read_with(cx, |s, _| {
        s.list(&space)
            .iter()
            .find(|p| p.kind == "agent")
            .expect("the default template seeds one agent")
            .id
            .clone()
    });
    stores.participants.update(cx, |s, cx| {
        s.promote(space.clone(), agent.clone(), None, cx)
    });
    wait_until(cx, "the share lands", |cx| {
        stores.participants.read_with(cx, |s, _| {
            s.list(&space)
                .iter()
                .any(|p| p.id == agent && p.source == "referenced")
        })
    });

    // Re-list from the database, not from the cached index: promotion emits
    // `Change::Participants` only (the Library provably didn't change), so this
    // asks the question the exclusion is the answer to.
    stores.spaces.update(cx, |s, cx| s.refresh(cx));
    wait_until(cx, "the library re-lists", |cx| {
        stores
            .spaces
            .read_with(cx, |s, _| !s.index().is_loading() && s.list().len() == 1)
    });
    assert_eq!(
        stores.spaces.read_with(cx, |s, _| s
            .list()
            .iter()
            .map(|r| r.id.clone())
            .collect::<Vec<_>>()),
        vec![space],
        "the notebook created by the share is not a conversation"
    );

    drain_runtime(&core);
}

/// Promote the seeded agent and hand back its id — the precondition of every
/// Agents-pane test, since an agent reaches the library only by being shared.
fn share_the_seeded_agent(cx: &mut TestAppContext, stores: &Stores, space: &str) -> String {
    stores
        .participants
        .update(cx, |s, cx| s.ensure(space.to_string(), cx));
    wait_until(cx, "participants load", |cx| {
        participant_labels(stores, space, cx).len() == 2
    });
    let agent = stores.participants.read_with(cx, |s, _| {
        s.list(space)
            .iter()
            .find(|p| p.kind == "agent")
            .expect("the default template seeds one agent")
            .id
            .clone()
    });
    stores.participants.update(cx, |s, cx| {
        s.promote(space.to_string(), agent.clone(), None, cx)
    });
    // Behavior tests run bus-less (`Stores::for_test` installs no bridge), so
    // the library is re-listed here the way `Change::Participants` does it in
    // the app — the dispatch itself is pinned by
    // `a_participants_change_reaches_the_agent_library` (`tests/stores.rs`).
    wait_until(cx, "the promotion lands", |cx| {
        stores.participants.read_with(cx, |s, _| {
            s.list(space)
                .iter()
                .any(|p| p.id == agent && p.source == "referenced")
        })
    });
    stores.agents.update(cx, |s, cx| s.refresh(cx));
    wait_until(cx, "the share lands", |cx| {
        stores
            .agents
            .read_with(cx, |s, _| s.list().iter().any(|a| a.id == agent))
    });
    agent
}

/// The **Agents pane** end to end (task 36): a shared agent appears in the
/// library with its notebook, its editor writes the agent's *own* config (the
/// edit-everywhere half, which the space it came from then follows), and its
/// notebook opens as an ordinary space window.
#[gpui::test]
fn agents_pane_lists_edits_and_opens_the_notebook(cx: &mut TestAppContext) {
    let (stores, core, _dir, space) = participants_scene(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| AgentsSettingsView::new(stores.clone(), window, cx))
    });
    // Nothing is shared yet, so the library is empty — the pane's own store is
    // what says so (the seeded human and Eidola are globals it must not list).
    wait_until(cx, "the empty library answers", |cx| {
        stores
            .agents
            .read_with(cx, |s, _| s.agents().has_value() && s.list().is_empty())
    });

    let agent = share_the_seeded_agent(cx, &stores, &space);
    let listed = stores.agents.read_with(cx, |s, _| s.list()[0].clone());
    assert_eq!(listed.id, agent);
    let notebook = listed
        .notebook_space_id
        .clone()
        .expect("promotion created a notebook, and the roster carries its id");

    // Edit everywhere: the name written here is the agent's own.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.toggle_edit(&agent, window, cx));
    })
    .unwrap();
    let label = view
        .read_with(cx, |v, _| v.editing_label_state())
        .expect("the row's editor is open");
    cx.update_window(window, |_, window, cx| {
        label.update(cx, |s, cx| s.set_value("Ada", window, cx));
    })
    .unwrap();
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.save_edit(window, cx));
    })
    .unwrap();
    wait_until(cx, "the rename lands", |cx| {
        stores
            .agents
            .read_with(cx, |s, _| s.list().iter().any(|a| a.label == "Ada"))
    });
    // …and the space it came from reads the new name, because its membership
    // overrides nothing.
    stores
        .participants
        .update(cx, |s, cx| s.refresh(space.clone(), cx));
    wait_until(cx, "the space follows", |cx| {
        stores.participants.read_with(cx, |s, _| {
            s.list(&space)
                .iter()
                .any(|p| p.id == agent && p.label == "Ada")
        })
    });

    // The notebook door: a real space, through the ordinary window path.
    view.update(cx, |v, cx| v.open_notebook(notebook, cx));
    cx.run_until_parked();
    assert_eq!(
        view.read_with(cx, |v, _| v.notebooks_opened_for_test()),
        1,
        "the notebook opens as a space window"
    );

    drain_runtime(&core);
}

/// **A form that closes hands the keyboard back** (Codex review, PR #279).
///
/// The editor's fields are focusable elements owned by the draft, and every way
/// the editor closes — Cancel, Save, arming Retire, or a roster refresh that
/// drops the agent — drops them. Whatever the window was focused on is then a
/// handle to something nobody paints: no keystroke reaches anything, and Tab
/// restarts from the window root instead of resuming beside the row the reader
/// was working on. The `held` observation has to be taken **before** the drop,
/// since a dead input's handle is gone with it.
#[gpui::test]
fn agents_pane_hands_the_keyboard_back_when_its_editor_closes(cx: &mut TestAppContext) {
    let (stores, core, _dir, space) = participants_scene(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| AgentsSettingsView::new(stores.clone(), window, cx))
    });
    let agent = share_the_seeded_agent(cx, &stores, &space);

    let pane_holds_the_keyboard = |cx: &mut TestAppContext| {
        cx.update_window(window, |_, window, cx| {
            view.read(cx).focus_handle().is_focused(window)
        })
        .unwrap()
    };
    let open_and_focus_the_charter = |cx: &mut TestAppContext| {
        cx.update_window(window, |_, window, cx| {
            view.update(cx, |v, cx| v.toggle_edit(&agent, window, cx));
        })
        .unwrap();
        let prompt = view
            .read_with(cx, |v, _| v.editing_prompt_state())
            .expect("the editor is open");
        cx.update_window(window, |_, window, cx| {
            prompt.update(cx, |s, cx| s.focus(window, cx));
        })
        .unwrap();
        assert!(
            !pane_holds_the_keyboard(cx),
            "the charter holds the keyboard while the editor is open"
        );
    };

    // **Focus inside the editor that is not one of its two inputs.** The editor
    // is a subtree — verbs, chips, a model dropdown — and a handback gated on an
    // enumeration of its text fields answers "not held" for everything else in
    // it, dropping the keyboard on the floor for exactly the controls a future
    // edit adds (Codex review, PR #279).
    open_and_focus_the_charter(cx);
    let subtree = view
        .read_with(cx, |v, _| v.editing_focus_handle())
        .expect("the editor is open");
    cx.update_window(window, |_, window, cx| {
        window.focus(&subtree, cx);
    })
    .unwrap();
    assert!(
        !pane_holds_the_keyboard(cx),
        "the editor subtree holds the keyboard, not the pane"
    );
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.cancel_edit(window, cx));
    })
    .unwrap();
    assert!(
        pane_holds_the_keyboard(cx),
        "containment, not an enumeration of inputs, is what the handback asks"
    );

    // Cancel.
    open_and_focus_the_charter(cx);
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.cancel_edit(window, cx));
    })
    .unwrap();
    assert!(pane_holds_the_keyboard(cx), "Cancel hands it back");

    // Save.
    open_and_focus_the_charter(cx);
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.save_edit(window, cx));
    })
    .unwrap();
    assert!(pane_holds_the_keyboard(cx), "Save hands it back");

    // **The confirmation's own buttons are tab stops** (`probe(Role::Button)`
    // derives `focusable()` + `tab_index(0)`), so Keep and Retire unmount a
    // focused control just as an editor's fields do — and owe the keyboard back
    // the same way (Codex review, PR #279).
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.arm_retire(&agent, window, cx));
    })
    .unwrap();
    let confirm = view.read_with(cx, |v, _| v.retire_focus_handle());
    cx.update_window(window, |_, window, cx| {
        window.focus(&confirm, cx);
    })
    .unwrap();
    assert!(
        !pane_holds_the_keyboard(cx),
        "the confirmation holds the keyboard"
    );
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.cancel_retire(window, cx));
    })
    .unwrap();
    assert!(pane_holds_the_keyboard(cx), "Keep hands it back");

    // Arming a retirement, which replaces the editor.
    open_and_focus_the_charter(cx);
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.arm_retire(&agent, window, cx));
    })
    .unwrap();
    assert!(pane_holds_the_keyboard(cx), "arming Retire hands it back");
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.cancel_retire(window, cx));
    })
    .unwrap();

    // The same unmount, over an armed **confirmation** rather than an editor:
    // its Keep/Retire are tab stops too, and a roster refresh that drops the
    // agent takes them away without anyone pressing anything. A second shared
    // agent, so the roster still has a row for the editor case below.
    let core_handle = stores.app_core().expect("a real core");
    let space_b = core_handle
        .runtime()
        .block_on(core_handle.create_space(Some("B".into())))
        .expect("a space")
        .id;
    let agent_b = core_handle
        .runtime()
        .block_on(core_handle.list_space_participants(space_b.clone()))
        .expect("participants")
        .into_iter()
        .find(|p| p.kind == "agent")
        .expect("the seeded agent")
        .id;
    core_handle
        .runtime()
        .block_on(core_handle.promote_participant(agent_b.clone(), None, None))
        .expect("share the second agent");
    stores.agents.update(cx, |s, cx| s.refresh(cx));
    wait_until(cx, "the library lists both", |cx| {
        stores.agents.read_with(cx, |s, _| s.list().len() == 2)
    });
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.arm_retire(&agent_b, window, cx));
    })
    .unwrap();
    let confirm_focus = view.read_with(cx, |v, _| v.retire_focus_handle());
    cx.update_window(window, |_, window, cx| {
        window.focus(&confirm_focus, cx);
    })
    .unwrap();
    assert!(!pane_holds_the_keyboard(cx), "the confirmation holds it");
    core_handle
        .runtime()
        .block_on(core_handle.retire_participant(agent_b.clone()))
        .expect("the other window's retirement");
    stores.agents.update(cx, |s, cx| s.refresh(cx));
    wait_until(cx, "the roster drops it", |cx| {
        stores.agents.read_with(cx, |s, _| s.list().len() == 1)
    });
    draw_window(cx, window);
    assert!(
        view.read_with(cx, |v, _| v.retiring_agent().is_none()),
        "the confirmation went with its row"
    );
    assert!(
        pane_holds_the_keyboard(cx),
        "and so did the keyboard it was holding"
    );

    // And the unmount no verb pressed: the agent leaves the roster under an
    // open editor, and `sync_open_forms` retires it at the head of `render`.
    open_and_focus_the_charter(cx);
    let handle = stores.app_core().expect("a real core");
    handle
        .runtime()
        .block_on(handle.retire_participant(agent.clone()))
        .expect("the other window's retirement");
    stores.agents.update(cx, |s, cx| s.refresh(cx));
    wait_until(cx, "the roster drops it", |cx| {
        stores.agents.read_with(cx, |s, _| s.list().is_empty())
    });
    draw_window(cx, window);
    assert!(
        view.read_with(cx, |v, _| v.editing_agent().is_none()),
        "the editor went with its row"
    );
    assert!(
        pane_holds_the_keyboard(cx),
        "and so did the keyboard it was holding"
    );

    drain_runtime(&core);
}

/// **Retirement** asks first, then takes the agent out of the library and
/// archives its notebook — and is not an unshare: the participant survives, so
/// the space it worked in still resolves it.
#[gpui::test]
fn agents_pane_retires_behind_a_confirm(cx: &mut TestAppContext) {
    let (stores, core, _dir, space) = participants_scene(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| AgentsSettingsView::new(stores.clone(), window, cx))
    });
    let agent = share_the_seeded_agent(cx, &stores, &space);

    // Armed from over an open editor, the retirement takes that editor with it —
    // and unlike the share, discarding the draft is the honest reading: the
    // confirmation *replaces* the editor on screen rather than standing under
    // its still-visible fields, it promises nothing about configuration, and a
    // config edit to an agent leaving the library is a value nothing will read
    // again.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.toggle_edit(&agent, window, cx));
    })
    .unwrap();
    let label = view
        .read_with(cx, |v, _| v.editing_label_state())
        .expect("the row's editor is open");
    cx.update_window(window, |_, window, cx| {
        label.update(cx, |s, cx| s.set_value("Renamed in passing", window, cx));
    })
    .unwrap();
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.arm_retire(&agent, window, cx));
    })
    .unwrap();
    assert!(
        view.read_with(cx, |v, _| v.editing_agent().is_none()),
        "arming the retirement closes the editor it replaces"
    );
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.cancel_retire(window, cx));
    })
    .unwrap();
    cx.run_until_parked();
    assert!(
        stores.agents.read_with(cx, |s, _| s
            .list()
            .iter()
            .all(|a| a.label != "Renamed in passing")),
        "the draft the retirement discarded was never written"
    );

    // Armed, then stood down — nothing is written.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.arm_retire(&agent, window, cx));
    })
    .unwrap();
    assert_eq!(
        view.read_with(cx, |v, _| v.retiring_agent().map(str::to_string)),
        Some(agent.clone())
    );
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.cancel_retire(window, cx));
    })
    .unwrap();
    assert!(view.read_with(cx, |v, _| v.retiring_agent().is_none()));
    cx.run_until_parked();
    assert_eq!(
        stores.agents.read_with(cx, |s, _| s.list().len()),
        1,
        "declining the confirmation retires nothing"
    );

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.arm_retire(&agent, window, cx));
    })
    .unwrap();
    // The confirmation's Retire is a real tab stop, and pressing it unmounts
    // the button — so it owes the keyboard back like every other closing path
    // (Codex review, PR #279).
    let confirm = view.read_with(cx, |v, _| v.retire_focus_handle());
    cx.update_window(window, |_, window, cx| {
        window.focus(&confirm, cx);
    })
    .unwrap();
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.confirm_retire(window, cx));
    })
    .unwrap();
    assert!(
        cx.update_window(window, |_, window, cx| {
            view.read(cx).focus_handle().is_focused(window)
        })
        .unwrap(),
        "Retire hands the keyboard back to the pane"
    );
    wait_until(cx, "the retirement lands", |cx| {
        stores.agents.read_with(cx, |s, _| s.list().is_empty())
    });
    assert!(
        stores
            .agents
            .read_with(cx, |s, _| s.op_error(&agent).map(str::to_string))
            .is_none(),
        "the retirement was accepted"
    );
    // The notebook went with it (archived in the same transaction).
    let core_handle = stores.app_core().expect("a real core");
    assert!(
        core_handle
            .runtime()
            .block_on(core_handle.list_global_agents())
            .expect("library")
            .is_empty()
    );
    // Not an unshare: the space still has the agent as a referenced member.
    stores
        .participants
        .update(cx, |s, cx| s.refresh(space.clone(), cx));
    cx.run_until_parked();

    drain_runtime(&core);
}

/// A failed *initial* read must not read as an empty library, and a refused
/// write must say so — the two honest-state rules, over this pane's cell.
#[gpui::test]
fn agents_pane_failed_load_and_refused_write_are_visible(cx: &mut TestAppContext) {
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.agents = Some(Vec::new());
    });
    let (_window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| AgentsSettingsView::new(stores.clone(), window, cx))
    });
    let _ = view;

    stores.agents.read_with(cx, |s, _| {
        assert!(s.agents().has_value(), "an empty library has answered");
        assert!(s.list().is_empty());
    });

    stores
        .agents
        .update(cx, |s, _| s.set_failed_for_test("boom"));
    stores.agents.read_with(cx, |s, _| {
        assert!(
            !s.agents().has_value(),
            "a failed initial read holds nothing"
        );
        assert!(s.agents().error().is_some());
    });
}

#[gpui::test]
fn templates_pane_crud_and_set_default(cx: &mut TestAppContext) {
    let (stores, core, _dir, _space) = participants_scene(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| TemplatesSettingsView::new(stores.clone(), window, cx))
    });
    wait_until(cx, "templates load", |cx| {
        stores.templates.read_with(cx, |s, _| !s.list().is_empty())
    });

    // Seeded with the built-in "Default", which is the default.
    let titles = stores.templates.read_with(cx, |s, _| {
        s.list().iter().map(|t| t.title.clone()).collect::<Vec<_>>()
    });
    assert!(
        titles.iter().any(|t| t == "Default"),
        "seeded Default: {titles:?}"
    );

    // Create a template.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.begin_create(window, cx));
    })
    .unwrap();
    assert_eq!(view.read_with(cx, |v, _| v.draft_cascade()), Some(4));
    let title = view.read_with(cx, |v, _| v.draft_title_state()).unwrap();
    cx.update_window(window, |_, window, cx| {
        title.update(cx, |s, cx| s.set_value("Research", window, cx));
        view.update(cx, |v, cx| v.cascade_inc(1, cx));
    })
    .unwrap();
    assert_eq!(view.read_with(cx, |v, _| v.draft_cascade()), Some(5));
    view.update(cx, |v, cx| v.save(cx));
    wait_until(cx, "template created", |cx| {
        stores
            .templates
            .read_with(cx, |s, _| s.list().iter().any(|t| t.title == "Research"))
    });

    let research = stores
        .templates
        .read_with(cx, |s, _| {
            s.list().iter().find(|t| t.title == "Research").cloned()
        })
        .expect("Research template created");
    assert_eq!(research.cascade_limit, 5, "cascade limit persisted");

    // Set it as default (config write-through).
    view.update(cx, |v, cx| v.set_default(&research.id, cx));
    wait_until(cx, "default set", |cx| {
        stores.config.read_with(cx, |c, _| c.default_template()) == Some(research.id.clone())
    });

    // Remove it (soft — leaves the listing).
    view.update(cx, |v, cx| v.remove_template(&research.id, cx));
    wait_until(cx, "template removed", |cx| {
        stores
            .templates
            .read_with(cx, |s, _| !s.list().iter().any(|t| t.title == "Research"))
    });

    drain_runtime(&core);
}

/// The router picker writes through `set_template_router_model`, and **Off**
/// round-trips back to NULL — the default is a real choice, not a one-way door.
#[gpui::test]
fn templates_pane_router_model_writes_through_and_off_round_trips(cx: &mut TestAppContext) {
    let (stores, core, _dir, _space) = participants_scene(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| TemplatesSettingsView::new(stores.clone(), window, cx))
    });
    wait_until(cx, "templates load", |cx| {
        stores.templates.read_with(cx, |s, _| !s.list().is_empty())
    });
    let default_id = eidola_app_core::DEFAULT_TEMPLATE_ID;

    // The seeded template's router is unset — the draft reads Off.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.begin_edit(default_id, window, cx));
    })
    .unwrap();
    assert_eq!(
        view.read_with(cx, |v, _| v.draft_router_model()),
        Some(None),
        "an unset router opens as Off"
    );

    // Pick a model and save: the setting lands on the template.
    view.update(cx, |v, cx| v.set_router_model(Some("gemma4-31b"), cx));
    view.update(cx, |v, cx| v.save(cx));
    wait_until(cx, "router model persisted", |cx| {
        stores.templates.read_with(cx, |s, _| {
            s.list()
                .iter()
                .find(|t| t.id == default_id)
                .and_then(|t| t.router_model.clone())
                .is_some()
        })
    });
    let stored = stores.templates.read_with(cx, |s, _| {
        s.list()
            .iter()
            .find(|t| t.id == default_id)
            .and_then(|t| t.router_model.clone())
    });
    assert_eq!(
        stored.as_deref(),
        Some("gemma4-31b"),
        "the picked reference is what was written"
    );

    // Re-open: the draft is seeded from the stored value, and Off clears it.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.begin_edit(default_id, window, cx));
    })
    .unwrap();
    assert_eq!(
        view.read_with(cx, |v, _| v.draft_router_model()),
        Some(Some("gemma4-31b".to_string())),
        "the editor reads back what it wrote"
    );
    view.update(cx, |v, cx| v.set_router_model(None, cx));
    view.update(cx, |v, cx| v.save(cx));
    wait_until(cx, "router model cleared", |cx| {
        stores.templates.read_with(cx, |s, _| {
            s.list()
                .iter()
                .find(|t| t.id == default_id)
                .map(|t| t.router_model.is_none())
                .unwrap_or(false)
        })
    });

    drain_runtime(&core);
}

/// Setting a template's router is a **second** core call after the create, and
/// it validates its backend — so a pick that went stale (its backend disabled
/// or removed while the editor was open) must be refused **before** anything is
/// written, not after a template has already landed without the router it was
/// created with.
#[gpui::test]
fn a_stale_router_pick_refuses_before_creating_the_template(cx: &mut TestAppContext) {
    let (stores, core, _dir, _space) = participants_scene(cx);
    stores.templates.update(cx, |s, cx| s.refresh(cx));
    wait_until(cx, "templates load", |cx| {
        stores.templates.read_with(cx, |s, _| !s.list().is_empty())
    });

    // The pick goes stale: its backend is disabled after the editor opened.
    core.runtime()
        .block_on(core.set_backend_enabled("eidola".into(), false))
        .expect("disable eidola");

    stores.templates.update(cx, |s, cx| {
        s.create(
            "Doomed".into(),
            4,
            Vec::new(),
            Some("gemma4-31b".into()),
            cx,
        );
    });
    wait_until(cx, "the refusal surfaces", |cx| {
        stores
            .templates
            .read_with(cx, |s, _| s.op_error().is_some())
    });

    // Zero trace: the template itself was never created.
    let titles: Vec<String> = core
        .runtime()
        .block_on(core.list_space_templates())
        .expect("list templates")
        .into_iter()
        .map(|t| t.title)
        .collect();
    assert!(
        !titles.iter().any(|t| t == "Doomed"),
        "a refused router must leave no template behind: {titles:?}"
    );

    drain_runtime(&core);
}

/// A `SpaceTemplateInfo` embeds its referenced globals' **effective** config, so
/// an "edit everywhere" of a shared participant — which emits only
/// `Change::Participants` — must reach the templates snapshot too.
#[gpui::test]
fn everywhere_edit_of_a_shared_participant_reaches_the_templates_snapshot(cx: &mut TestAppContext) {
    let (stores, core, _dir, space) = participants_scene(cx);
    // A template projected from a space carries the shared "You" by reference.
    core.runtime()
        .block_on(core.template_from_space(space.clone(), "Projected".into()))
        .expect("project template");
    stores.templates.update(cx, |s, cx| s.refresh(cx));

    let referenced_label = |cx: &mut TestAppContext| -> Option<String> {
        stores.templates.read_with(cx, |s, _| {
            s.list()
                .iter()
                .find(|t| t.title == "Projected")
                .and_then(|t| t.referenced.first())
                .map(|r| r.label.clone())
        })
    };
    wait_until(cx, "projected template lists its shared global", |cx| {
        stores.templates.read_with(cx, |s, _| {
            s.list()
                .iter()
                .find(|t| t.title == "Projected")
                .map(|t| !t.referenced.is_empty())
                .unwrap_or(false)
        })
    });
    assert_eq!(referenced_label(cx).as_deref(), Some("User"));

    // Edit everywhere: the shared global's own config moves.
    stores.participants.update(cx, |s, cx| {
        s.update_everywhere(
            space.clone(),
            eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
            eidola_app_core::ParticipantUpdate {
                label: Some("Myself".into()),
                ..Default::default()
            },
            eidola_app_core::ExpectedScope::Any,
            cx,
        );
    });
    wait_until(cx, "the edit lands", |cx| {
        stores.participants.read_with(cx, |s, _| {
            s.list(&space).iter().any(|p| p.label == "Myself")
        })
    });

    // The bus bridge isn't installed in this scene, so drive the dispatch it
    // would have: this is the routing under test.
    cx.update(|cx| stores::dispatch_change_for_test(&stores, Some(Change::Participants), cx));
    wait_until(cx, "the templates snapshot follows", |cx| {
        referenced_label(cx).as_deref() == Some("Myself")
    });

    drain_runtime(&core);
}

/// A new agent participant arrives with the shared default system prompt, and
/// that prompt survives the save.
#[gpui::test]
fn templates_pane_new_agent_carries_the_default_system_prompt(cx: &mut TestAppContext) {
    use eidola_gui::participants::DEFAULT_AGENT_SYSTEM_PROMPT;

    let (stores, core, _dir, _space) = participants_scene(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| TemplatesSettingsView::new(stores.clone(), window, cx))
    });
    wait_until(cx, "templates load", |cx| {
        stores.templates.read_with(cx, |s, _| !s.list().is_empty())
    });

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.begin_create(window, cx));
        // A second agent takes the same starting point.
        view.update(cx, |v, cx| v.add_participant(window, cx));
    })
    .unwrap();
    for idx in [0, 1] {
        assert_eq!(
            view.read_with(cx, |v, cx| v.draft_participant_prompt(idx, cx))
                .as_deref(),
            Some(DEFAULT_AGENT_SYSTEM_PROMPT),
            "new agent {idx} is prefilled with the default system prompt"
        );
    }

    let title = view.read_with(cx, |v, _| v.draft_title_state()).unwrap();
    cx.update_window(window, |_, window, cx| {
        title.update(cx, |s, cx| s.set_value("Prompted", window, cx));
    })
    .unwrap();
    view.update(cx, |v, cx| v.save(cx));
    wait_until(cx, "template created", |cx| {
        stores
            .templates
            .read_with(cx, |s, _| s.list().iter().any(|t| t.title == "Prompted"))
    });
    let prompts = stores.templates.read_with(cx, |s, _| {
        s.list()
            .iter()
            .find(|t| t.title == "Prompted")
            .map(|t| {
                t.participants
                    .iter()
                    .map(|p| p.system_prompt.clone())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    });
    assert_eq!(
        prompts,
        vec![
            Some(DEFAULT_AGENT_SYSTEM_PROMPT.to_string()),
            Some(DEFAULT_AGENT_SYSTEM_PROMPT.to_string())
        ],
        "the default prompt round-trips through the template update path"
    );

    drain_runtime(&core);
}

/// An edited system prompt round-trips through the template update path.
#[gpui::test]
fn templates_pane_system_prompt_round_trips(cx: &mut TestAppContext) {
    let (stores, core, _dir, _space) = participants_scene(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| TemplatesSettingsView::new(stores.clone(), window, cx))
    });
    wait_until(cx, "templates load", |cx| {
        stores.templates.read_with(cx, |s, _| !s.list().is_empty())
    });
    let default_id = eidola_app_core::DEFAULT_TEMPLATE_ID;

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.begin_edit(default_id, window, cx));
    })
    .unwrap();
    let prompt = view
        .read_with(cx, |v, _| v.draft_participant_prompt_state(0))
        .expect("the seeded template owns one agent");
    cx.update_window(window, |_, window, cx| {
        prompt.update(cx, |s, cx| {
            s.set_value("Answer only in questions.", window, cx)
        });
    })
    .unwrap();
    view.update(cx, |v, cx| v.save(cx));
    wait_until(cx, "system prompt persisted", |cx| {
        stores.templates.read_with(cx, |s, _| {
            s.list()
                .iter()
                .find(|t| t.id == default_id)
                .and_then(|t| t.participants.first())
                .and_then(|p| p.system_prompt.clone())
                .as_deref()
                == Some("Answer only in questions.")
        })
    });

    // Re-opening the editor seeds the field from what was stored.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.begin_edit(default_id, window, cx));
    })
    .unwrap();
    assert_eq!(
        view.read_with(cx, |v, cx| v.draft_participant_prompt(0, cx))
            .as_deref(),
        Some("Answer only in questions.")
    );

    drain_runtime(&core);
}

/// A failed initial participant load must render Retry (not a phantom-empty
/// roster), and Retry must actually re-fetch. `ensure` declines once a `Failed`
/// cell exists, so `retry_load` is the only path back.
#[gpui::test]
fn inspector_participants_retry_refetches_after_failed_load(cx: &mut TestAppContext) {
    let (stores, core, _dir, space) = participants_scene(cx);
    let (_window, view) = open_participants_inspector(cx, &stores, &space);
    wait_until(cx, "participants load", |cx| {
        participant_labels(&stores, &space, cx).len() == 2
    });

    // Simulate a failed refresh: the cell goes Failed with no prior.
    let space_for_fail = space.clone();
    stores.participants.update(cx, move |s, _| {
        s.set_failed_for_test(&space_for_fail, "boom")
    });
    stores.participants.read_with(cx, |s, _| {
        let cell = s.participants(&space);
        assert!(cell.error().is_some() && !cell.has_value(), "failed, blank");
    });

    // Retry re-fetches; the real list lands again.
    view.update(cx, |v, cx| v.inspector_retry_participants(cx));
    wait_until(cx, "retry reloads", |cx| {
        participant_labels(&stores, &space, cx).len() == 2
    });

    drain_runtime(&core);
}

/// The per-space error keying: a failure on space A must not surface under
/// space B, and starting B's op must not clear A's error.
#[gpui::test]
fn participants_store_op_error_is_per_space(cx: &mut TestAppContext) {
    let (stores, core, _dir, space_a) = participants_scene(cx);
    let space_b = core
        .runtime()
        .block_on(core.create_space(None))
        .expect("space b")
        .id;

    // An empty label is rejected by app-core → a per-space op_error.
    let bad = || eidola_app_core::NewParticipant {
        label: "".into(),
        ..Default::default()
    };
    let a = space_a.clone();
    stores
        .participants
        .update(cx, move |s, cx| s.add(a, bad(), cx));
    wait_until(cx, "A op_error", |cx| {
        stores
            .participants
            .read_with(cx, |s, _| !s.op_errors_for(&space_a).is_empty())
    });
    stores.participants.read_with(cx, |s, _| {
        assert!(
            s.op_errors_for(&space_b).is_empty(),
            "B must not see A's error"
        );
    });

    let b = space_b.clone();
    stores
        .participants
        .update(cx, move |s, cx| s.add(b, bad(), cx));
    wait_until(cx, "B op_error", |cx| {
        stores
            .participants
            .read_with(cx, |s, _| !s.op_errors_for(&space_b).is_empty())
    });
    stores.participants.read_with(cx, |s, _| {
        assert!(
            !s.op_errors_for(&space_a).is_empty(),
            "starting B's op must not clear A's error (per-space keying)"
        );
    });

    drain_runtime(&core);
}

/// P1: a "New Space from Template" failure is owned by the store (not detached)
/// and surfaced in the store's `op_error` (the Library banner), not silently
/// discarded.
#[gpui::test]
fn spaces_store_create_from_template_surfaces_error(cx: &mut TestAppContext) {
    let (stores, core, _dir, _space) = participants_scene(cx);
    let stores_clone = stores.clone();
    stores.spaces.update(cx, move |s, cx| {
        s.create_from_template("does-not-exist".into(), stores_clone, cx);
    });
    wait_until(cx, "create error surfaced", |cx| {
        stores.spaces.read_with(cx, |s, _| s.op_error().is_some())
    });
    stores.spaces.read_with(cx, |s, _| {
        assert!(
            s.op_error_for("anything").is_none(),
            "a create has no space to tag, so no space's inspector claims it"
        );
    });

    drain_runtime(&core);
}

#[gpui::test]
fn space_post_fans_out_one_turn_per_planned_responder(cx: &mut TestAppContext) {
    // The composer's Post drives app-core's notification plan: one concurrent
    // streaming turn per planned responder (wave 3b). Two agents with notify
    // "to people" → a human post starts two turns at once, each in its own
    // keyed runner. With no account configured every turn fails (typed) — and
    // each failure completes *independently*, leaving the saved post intact
    // and the space recovered (never bricked).
    let (stores, core, _dir, space_id) = participants_scene(cx);
    core.runtime()
        .block_on(core.add_space_participant(
            space_id.clone(),
            eidola_app_core::NewParticipant {
                label: "Second Agent".into(),
                model_ref: Some("kimi-k2-6".into()),
                system_prompt: None,
                notify_policy: "human".into(),
            },
        ))
        .expect("add second agent");

    let (window, view) = open_space(cx, &stores, Some(space_id.clone()));
    wait_until(cx, "transcript load", |cx| {
        view.read_with(cx, |v, cx| {
            matches!(
                v.space().read(cx).transcript(),
                eidola_gui::loadable::Loadable::Loaded { .. }
            )
        })
    });

    // Observe the shared Space entity: record the maximum number of concurrent
    // streams (both turns start inside the submit-completion update, so the
    // observer deterministically sees 2), and count turn failures — each turn's
    // independent completion emits one `Failed`.
    let space = view.read_with(cx, |v, _| v.space().clone());
    let max_streams = std::rc::Rc::new(std::cell::Cell::new(0usize));
    let failures = std::rc::Rc::new(std::cell::Cell::new(0usize));
    let (max2, fail2) = (max_streams.clone(), failures.clone());
    let _subs = cx.update(|cx| {
        [
            cx.observe(&space, move |space, cx| {
                let n = space.read(cx).streams().len();
                if n > max2.get() {
                    max2.set(n);
                }
            }),
            cx.subscribe(&space, move |_, ev: &eidola_gui::space::SpaceEvent, _| {
                if matches!(ev, eidola_gui::space::SpaceEvent::Failed(_)) {
                    fail2.set(fail2.get() + 1);
                }
            }),
        ]
    });

    open_space_draft(&view, window, cx, None);
    set_space_composer_text(&view, window, cx, "hello everyone");
    dispatch_space_action(&view, window, cx, Send);

    wait_until(cx, "both turns complete", |_| failures.get() >= 2);
    assert_eq!(
        max_streams.get(),
        2,
        "the plan fanned out two concurrent streaming turns"
    );
    view.read_with(cx, |v, cx| {
        let space = v.space().read(cx);
        assert!(space.streams().is_empty(), "both failed turns collapsed");
        assert_eq!(
            space.messages().len(),
            1,
            "the saved post survives its failed turns"
        );
        assert!(space.can_retry(), "a failed turn is recorded for Retry");
    });

    drain_runtime(&core);
}

// ---------------------------------------------------------------------------
// Quoted references (wave 2) — quote creation, the footnote rail, removals,
// and the source-post highlights.
// ---------------------------------------------------------------------------

/// A post fixture whose single block carries a real id, so a selection inside
/// it resolves to a quotable `(block, range)` pair.
fn fixture_post_with_block(action_id: &str, block_id: &str, text: &str) -> PostNode {
    let mut p = fixture_user_post(action_id, text);
    p.blocks[0].id = block_id.into();
    p
}

/// Seed a space with one user post carrying an identified block, force a
/// frame so `sync_bodies` mints its editor, and return the view.
fn seed_quotable_space(
    view: &Entity<SpaceView>,
    window: AnyWindowHandle,
    cx: &mut TestAppContext,
    posts: Vec<PostNode>,
) {
    let space = view.read_with(cx, |v, _| v.space().clone());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| s.set_post_tree_for_test(posts, cx));
    })
    .unwrap();
    cx.update_window(window, |_, window, _| window.refresh())
        .unwrap();
    cx.run_until_parked();
}

#[gpui::test]
fn space_quote_attaches_a_reference_and_injects_its_marker(cx: &mut TestAppContext) {
    // The whole creation path: a selection inside a post's read-only body
    // becomes a quotable `PostSelection` (which gates the Edit menu), `Quote`
    // attaches it to the active draft at ordinal 1, and the marker lands in
    // the draft body as its own paragraph so the editor renders it as a quote
    // block.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_post_with_block("a1", "b1", "the quick brown fox")],
    );

    view.read_with(cx, |v, _| {
        assert!(
            !v.has_post_selection_for_test(),
            "nothing selected: the Quote items are unregistered (greyed)"
        );
    });

    // Select "quick brown" (bytes 4..15).
    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| v.select_in_post_for_test("a1", 4..15, cx));
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert!(
            v.has_post_selection_for_test(),
            "a selection inside one block is quotable"
        );
        assert_eq!(v.post_selection_action_id(), Some("a1".to_string()));
    });

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.quote(&eidola_gui::actions::Quote, window, cx));
    })
    .unwrap();
    cx.run_until_parked();

    view.read_with(cx, |v, _| {
        assert_eq!(
            v.active_draft_references_for_test(),
            vec![(1u64, "quick brown".to_string())],
            "the first quote takes ordinal 1 and carries the selected text"
        );
        assert!(
            !v.has_post_selection_for_test(),
            "the selection is consumed, so a second Quote can't silently re-attach it"
        );
    });

    let composer = view
        .read_with(cx, |v, _| v.composer_state_for_test())
        .expect("the quote activated a draft");
    cx.update_window(window, |_, _, cx| {
        let body = composer.read(cx).value().to_string();
        assert!(
            body.contains("{{ embed 1 }}"),
            "the marker is injected into the draft body: {body:?}"
        );
        assert_eq!(
            composer.read(cx).embeds().get(1),
            Some("quick brown"),
            "the editor's embed map maps the ordinal to the quoted passage"
        );
    })
    .unwrap();
}

#[gpui::test]
fn space_cross_block_selection_is_not_quotable(cx: &mut TestAppContext) {
    // A reference edge names exactly one content block. A selection spanning
    // two blocks names none, so it must not be quotable — inventing a block
    // would make the stored range a lie.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let mut post = fixture_post_with_block("a1", "b1", "first block");
    post.blocks.push(PostBlock {
        id: "b2".into(),
        block_type: "text".into(),
        text: Some("second block".into()),
        tool_name: None,
        tool_call_id: None,
        data: None,
    });
    seed_quotable_space(&view, window, cx, vec![post]);

    // 5..17 straddles the "first block" / "second block" boundary at 11.
    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| v.select_in_post_for_test("a1", 5..17, cx));
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert!(
            !v.has_post_selection_for_test(),
            "a cross-block selection is not quotable"
        );
    });

    // A selection wholly inside the second block is.
    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| v.select_in_post_for_test("a1", 11..17, cx));
    })
    .unwrap();
    view.read_with(cx, |v, _| assert!(v.has_post_selection_for_test()));
}

#[gpui::test]
fn space_quote_in_reply_targets_the_quoted_post(cx: &mut TestAppContext) {
    // `Quote in Reply` answers *where the passage is*: the draft it opens
    // replies to the quoted post, not to the branch tail.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let mut a2 = fixture_assistant_post("a2", "the reply");
    a2.parent_action_id = Some("a1".into());
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![
            fixture_post_with_block("a1", "b1", "the quick brown fox"),
            a2,
        ],
    );

    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| v.select_in_post_for_test("a1", 4..9, cx));
    })
    .unwrap();
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.quote_in_reply(&eidola_gui::actions::QuoteInReply, window, cx)
        });
    })
    .unwrap();
    cx.run_until_parked();

    view.read_with(cx, |v, _| {
        assert_eq!(
            v.active_draft_parent_for_test(),
            Some("a1".to_string()),
            "the reply branches at the quoted post, not at the tail (a2)"
        );
        assert_eq!(
            v.active_draft_references_for_test(),
            vec![(1u64, "quick".to_string())]
        );
    });
}

#[gpui::test]
fn space_quoted_draft_posts_its_references_and_a_rejected_post_keeps_them(cx: &mut TestAppContext) {
    // The round trip: a quoted draft's references reach `Space::submit` as
    // specs in ordinal order — and a **rejected** post (the space busy)
    // leaves both the draft and its references untouched, the
    // accept-before-consume contract (PR #227) extended to quotes.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_post_with_block("a1", "b1", "the quick brown fox")],
    );
    let space = view.read_with(cx, |v, _| v.space().clone());

    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| v.select_in_post_for_test("a1", 4..15, cx));
    })
    .unwrap();
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.quote(&eidola_gui::actions::Quote, window, cx));
    })
    .unwrap();
    cx.run_until_parked();

    // Occupy the exclusive mutation slot so the post is refused.
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| s.arm_post_runner_for_test(cx));
    })
    .unwrap();
    dispatch_space_action(&view, window, cx, Send);
    view.read_with(cx, |v, _| {
        assert!(
            v.has_active_draft_for_test(),
            "a rejected post leaves the draft active"
        );
        assert_eq!(
            v.active_draft_references_for_test(),
            vec![(1u64, "quick brown".to_string())],
            "…and leaves its pending references intact"
        );
    });

    // Free the slot; the post now lands and carries the reference.
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| s.clear_post_runner_for_test(cx));
    })
    .unwrap();
    dispatch_space_action(&view, window, cx, Send);

    space.read_with(cx, |s, _| {
        let refs = s.last_submitted_references();
        assert_eq!(refs.len(), 1, "the post carried its quoted reference");
        assert_eq!(refs[0].antecedent_action_id, "a1");
        assert_eq!(refs[0].content_block_id.as_deref(), Some("b1"));
        assert_eq!(refs[0].range_start, Some(4));
        assert_eq!(refs[0].range_end, Some(15));
    });
    view.read_with(cx, |v, _| {
        assert!(
            !v.has_active_draft_for_test() || v.active_draft_references_for_test().is_empty(),
            "an accepted post consumes the draft and its references"
        );
    });
}

#[gpui::test]
fn space_removing_a_draft_quote_drops_its_marker_and_compacts_on_post(cx: &mut TestAppContext) {
    // Removing a pending quote takes its marker with it (a stranded marker
    // would render as literal `{{ embed N }}`), and does **not** renumber the
    // survivors — their markers already address them. The gap is reconciled
    // only at the durability boundary, where app-core assigns `1..=N`.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_post_with_block("a1", "b1", "alpha beta gamma")],
    );
    let space = view.read_with(cx, |v, _| v.space().clone());

    for range in [0..5usize, 6..10, 11..16] {
        cx.update_window(window, |_, _, cx| {
            view.update(cx, |v, cx| {
                v.select_in_post_for_test("a1", range.clone(), cx)
            });
        })
        .unwrap();
        cx.update_window(window, |_, window, cx| {
            view.update(cx, |v, cx| v.quote(&eidola_gui::actions::Quote, window, cx));
        })
        .unwrap();
    }
    cx.run_until_parked();
    view.read_with(cx, |v, _| {
        assert_eq!(
            v.active_draft_references_for_test()
                .iter()
                .map(|(o, _)| *o)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    });

    let composer = view
        .read_with(cx, |v, _| v.composer_state_for_test())
        .expect("draft");
    let draft_id = view.read_with(cx, |v, _| v.active_draft_id_for_test().unwrap());

    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| {
            v.remove_draft_reference(&draft_id.clone().into(), 2, cx)
        });
    })
    .unwrap();

    view.read_with(cx, |v, _| {
        assert_eq!(
            v.active_draft_references_for_test()
                .iter()
                .map(|(o, _)| *o)
                .collect::<Vec<_>>(),
            vec![1, 3],
            "the survivors keep their ordinals — the gap is correct"
        );
    });
    cx.update_window(window, |_, _, cx| {
        let body = composer.read(cx).value().to_string();
        assert!(
            !body.contains("{{ embed 2 }}"),
            "the removed quote's marker went with it: {body:?}"
        );
        assert!(body.contains("{{ embed 1 }}") && body.contains("{{ embed 3 }}"));
    })
    .unwrap();

    // Posting compacts to `1..=N` and rewrites the markers to match, because
    // that is the order app-core will assign the edges in.
    dispatch_space_action(&view, window, cx, Send);
    space.read_with(cx, |s, _| {
        let refs = s.last_submitted_references();
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].range_start, Some(0), "ordinal 1 = the first quote");
        assert_eq!(refs[1].range_start, Some(11), "ordinal 2 = the third quote");
    });
    // The optimistic post's body carries the compacted markers.
    space.read_with(cx, |s, _| {
        let last = s
            .messages()
            .last()
            .expect("the posted turn")
            .message
            .content
            .clone();
        assert!(last.contains("{{ embed 1 }}"), "{last:?}");
        assert!(last.contains("{{ embed 2 }}"), "{last:?}");
        assert!(!last.contains("{{ embed 3 }}"), "{last:?}");
    });
}

#[gpui::test]
fn space_draft_footnote_can_re_embed_its_quote(cx: &mut TestAppContext) {
    // A reference and its marker are separable — deleting the quote block in
    // the body leaves the footnote behind — so the rail carries the way back.
    // The affordance re-places the marker, and does **not** offer to place a
    // second one while the block is already there.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_post_with_block("a1", "b1", "alpha beta gamma")],
    );
    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| v.select_in_post_for_test("a1", 0..5, cx));
    })
    .unwrap();
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.quote(&eidola_gui::actions::Quote, window, cx));
    })
    .unwrap();
    cx.run_until_parked();

    let composer = view
        .read_with(cx, |v, _| v.composer_state_for_test())
        .expect("draft");
    let draft_id: gpui::SharedString = view
        .read_with(cx, |v, _| v.active_draft_id_for_test().unwrap())
        .into();

    cx.update_window(window, |_, _, cx| {
        assert!(composer.read(cx).value().contains("{{ embed 1 }}"));
        // Strip the marker the way a Backspace over the block would, leaving
        // the reference (and its footnote row) in place.
        composer.update(cx, |e, cx| e.set_value("just prose", cx));
    })
    .unwrap();

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.embed_draft_reference(&draft_id, 1, window, cx)
        });
    })
    .unwrap();
    cx.run_until_parked();

    cx.update_window(window, |_, _, cx| {
        let body = composer.read(cx).value().to_string();
        assert!(
            body.contains("{{ embed 1 }}"),
            "the rail put the quote back: {body:?}"
        );
        assert!(body.contains("just prose"), "prose is kept: {body:?}");
        assert_eq!(
            body.matches("{{ embed 1 }}").count(),
            1,
            "exactly one marker: {body:?}"
        );
    })
    .unwrap();

    // The reference itself is untouched — this places a marker, it does not
    // mint a quote.
    view.read_with(cx, |v, _| {
        assert_eq!(
            v.active_draft_references_for_test()
                .iter()
                .map(|(o, _)| *o)
                .collect::<Vec<_>>(),
            vec![1]
        );
    });

    // A reference the draft doesn't carry is a no-op (no stray marker).
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.embed_draft_reference(&draft_id, 9, window, cx)
        });
    })
    .unwrap();
    cx.update_window(window, |_, _, cx| {
        assert!(!composer.read(cx).value().contains("{{ embed 9 }}"));
    })
    .unwrap();
}

#[gpui::test]
fn space_post_footnote_removal_rides_the_edit_session(cx: &mut TestAppContext) {
    // A persisted post's references are removed through the existing Edit
    // session: the rail's chips mark ordinals, and committing hands them to
    // `edit_post_with_removals`. Ordinal 0 — the reply edge — is refused.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    // The body addresses both references by marker, as a real quoted post does.
    let mut post = fixture_post_with_block(
        "a1",
        "b1",
        "body text\n\n{{ embed 1 }}\n\nmore\n\n{{ embed 2 }}",
    );
    post.references = vec![
        eidola_app_core::PostReference {
            antecedent_action_id: "x1".into(),
            ordinal: 1,
            content_block_id: Some("bx".into()),
            range_start: Some(0),
            range_end: Some(4),
            annotation: None,
            delegation_end: None,
            snippet: Some("some passage".into()),
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
            delegation_end: None,
            // The stored range no longer maps — the rail says so rather than
            // guessing at a remap.
            snippet: None,
            antecedent_author_label: "Ada".into(),
            antecedent_author_kind: "agent".into(),
        },
    ];
    seed_quotable_space(&view, window, cx, vec![post]);
    let space = view.read_with(cx, |v, _| v.space().clone());

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.begin_edit("a1".into(), window, cx));
    })
    .unwrap();

    // Ordinal 0 can never be marked (the reply edge is the thread).
    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| v.toggle_reference_removal(0, cx));
        view.update(cx, |v, cx| v.toggle_reference_removal(2, cx));
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert_eq!(v.edit_removals_for_test(), vec![2]);
    });
    // Toggling is reversible.
    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| v.toggle_reference_removal(2, cx));
        view.update(cx, |v, cx| v.toggle_reference_removal(1, cx));
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert_eq!(v.edit_removals_for_test(), vec![1]);
    });

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.commit_edit(window, cx));
    })
    .unwrap();
    space.read_with(cx, |s, _| {
        assert_eq!(
            s.last_edit_removals(),
            &[1],
            "the marked ordinals reach edit_post_with_removals"
        );
        // The marker leaves with its edge. Left behind it would render as
        // literal wire syntax on reload — and go upstream literally, since
        // there is no edge left to expand it against.
        let text = s.last_edit_text();
        assert!(
            !text.contains("{{ embed 1 }}"),
            "the removed reference's marker is stripped from the submission: {text:?}"
        );
        assert!(
            text.contains("{{ embed 2 }}"),
            "a surviving reference keeps the marker that addresses it: {text:?}"
        );
        assert!(
            text.contains("body text") && text.contains("more"),
            "the prose around the removed marker survives: {text:?}"
        );
    });
}

#[gpui::test]
fn space_navigating_to_an_edited_generation_selects_its_tip_in_place(cx: &mut TestAppContext) {
    // A reference names a *concrete generation*, so once the quoted post is
    // edited that action is gone from the current-tip tree. Navigation must
    // still land on it via its item — resolving to the tip that superseded it
    // — rather than treating it as foreign and opening a duplicate window on
    // the space we are already looking at.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    // A branched space: a2 is the spine (selected by default), a3 a sibling —
    // so resolving onto a3's item is observable as a selection move.
    let mut a2 = fixture_assistant_post("a2", "the reply");
    a2.parent_action_id = Some("a1".into());
    let mut a3 = fixture_post_with_block("a3", "b3", "the quoted post");
    a3.parent_action_id = Some("a1".into());
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_post_with_block("a1", "b1", "root"), a2, a3],
    );

    // A reference to a *superseded* generation of a3's item resolves through
    // the item to the tip that replaced it (fixtures key items `item-<id>`).
    let found = cx
        .update_window(window, |_, window, cx| {
            view.update(cx, |v, cx| v.select_item_tip("item-a3", window, cx))
        })
        .unwrap();
    assert!(
        found,
        "the item's current generation is in this tree, so navigation stays in this window"
    );
    let selected = cx
        .update_window(window, |_, window, cx| {
            view.read_with(cx, |v, _| v.selected_leaf_for_test(window))
        })
        .unwrap();
    assert_eq!(
        selected.as_deref(),
        Some("a3"),
        "the tip that superseded the quoted generation is selected, not the spine"
    );

    // An item this space doesn't render is honestly not found — that is what
    // falls through to opening the reference's own space.
    let missing = cx
        .update_window(window, |_, window, cx| {
            view.update(cx, |v, cx| v.select_item_tip("item-elsewhere", window, cx))
        })
        .unwrap();
    assert!(!missing, "a foreign item does not resolve in this tree");
}

#[gpui::test]
fn space_incoming_references_paint_highlights_and_navigate(cx: &mut TestAppContext) {
    // Source-post highlights: an incoming reference's stored (block, range)
    // maps back onto the body editor's buffer offsets, a single referencer
    // navigates directly, and a range that no longer maps paints nothing.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let mut a2 = fixture_assistant_post("a2", "the reply");
    a2.parent_action_id = Some("a1".into());
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![
            fixture_post_with_block("a1", "b1", "the quick brown fox"),
            a2,
        ],
    );
    let space = view.read_with(cx, |v, _| v.space().clone());

    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, _| {
            s.seed_incoming_references_for_test(
                "a1",
                vec![
                    eidola_app_core::IncomingReference {
                        action_id: "a2".into(),
                        space_id: "s".into(),
                        ordinal: 1,
                        content_block_id: Some("b1".into()),
                        range_start: Some(4),
                        range_end: Some(9),
                        annotation: None,
                        created_at: 0,
                    },
                    // A range past the block's end no longer maps: dropped,
                    // never approximated.
                    eidola_app_core::IncomingReference {
                        action_id: "a2".into(),
                        space_id: "s".into(),
                        ordinal: 2,
                        content_block_id: Some("b1".into()),
                        range_start: Some(100),
                        range_end: Some(200),
                        annotation: None,
                        created_at: 0,
                    },
                ],
            );
        });
    })
    .unwrap();

    view.read_with(cx, |v, cx| {
        assert_eq!(
            v.highlight_ranges_for_test(0, cx),
            vec![(4usize..9usize, 0u64)],
            "only the range that still maps is painted"
        );
    });

    // One referencer → navigate straight there (the branch is selected).
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.click_highlight_for_test("a1", &[0], window, cx)
        });
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert!(
            v.highlight_picker_for_test().is_none(),
            "a single referencer needs no picker"
        );
    });
}

#[gpui::test]
fn space_multiple_referencers_open_a_picker(cx: &mut TestAppContext) {
    // Overlapping quotes of one passage can't be disambiguated by a click, so
    // the choice becomes the user's: a small picker naming each referencing
    // post, dismissed by choosing one.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let mut a2 = fixture_assistant_post("a2", "first responder");
    a2.parent_action_id = Some("a1".into());
    let mut a3 = fixture_user_post("a3", "second responder");
    a3.parent_action_id = Some("a1".into());
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![
            fixture_post_with_block("a1", "b1", "the quick brown fox"),
            a2,
            a3,
        ],
    );
    let space = view.read_with(cx, |v, _| v.space().clone());
    let incoming = |action: &str, lo: i64, hi: i64| eidola_app_core::IncomingReference {
        action_id: action.into(),
        space_id: "s".into(),
        ordinal: 1,
        content_block_id: Some("b1".into()),
        range_start: Some(lo),
        range_end: Some(hi),
        annotation: None,
        created_at: 0,
    };
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, _| {
            s.seed_incoming_references_for_test(
                "a1",
                vec![incoming("a2", 4, 15), incoming("a3", 10, 19)],
            );
        });
    })
    .unwrap();

    // Both ranges cover byte 12 — the editor reports both keys.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.click_highlight_for_test("a1", &[0, 1], window, cx)
        });
    })
    .unwrap();
    let choices = view
        .read_with(cx, |v, _| v.highlight_picker_for_test())
        .expect("two referencers open the picker");
    assert_eq!(choices.len(), 2);
    assert_eq!(choices[0].0, "a2");
    assert_eq!(choices[1].0, "a3");
    assert!(
        choices[0].1.contains("first responder"),
        "the row names the reply, not an opaque id: {:?}",
        choices[0].1
    );

    // Choosing one navigates and closes the picker.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.navigate_to_action("a3".into(), window, cx));
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert!(v.highlight_picker_for_test().is_none());
    });
}

#[gpui::test]
fn space_navigation_glides_to_its_destination_instead_of_jumping(cx: &mut TestAppContext) {
    // "See in context", a footnote row, a highlight's referencer: every
    // navigation that takes the reader somewhere in *this* space animates the
    // travel (task 46, bug 4). A jump gives no sense of where you were taken
    // from; the glide carries the intervening page past you. The landing is
    // still exact.
    let stores = stub_stores_with_agents(cx, "s");
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());
    let long = "a long paragraph of the conversation so far. ".repeat(40);
    let mut a2 = fixture_assistant_post("a2", &long);
    a2.parent_action_id = Some("a1".into());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("a1", &long), a2], cx)
        });
    })
    .unwrap();

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(520.)));
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| v.scroll_page_to_end_for_test());
    vcx.run_until_parked();
    let from = view.read_with(&vcx, |v, _| v.page_scroll_offset_y_for_test());
    assert!(
        from < -100.0,
        "the reader starts well down the page ({from})"
    );

    vcx.update(|window, cx| {
        view.update(cx, |v, cx| v.navigate_to_action("a1".into(), window, cx));
    });

    let (target, mid) = view.read_with(&vcx, |v, _| {
        (
            v.page_glide_target_for_test(),
            v.page_scroll_offset_y_for_test(),
        )
    });
    let target = target.expect("navigating arms a glide rather than jumping");
    assert!(
        (mid - target).abs() > 1.0,
        "the page has not teleported to the destination (offset {mid}, \
         destination {target})"
    );

    // It still lands exactly, and the glide retires itself. (Driven by hand:
    // no test dispatcher pumps `on_next_frame`, so the frame loop's body is
    // exercised through its own seam.)
    view.update(&mut vcx, |v, _| v.drive_page_glide_for_test(0.5));
    let half = view.read_with(&vcx, |v, _| v.page_scroll_offset_y_for_test());
    assert!(
        half > from + 1.0 && half < target - 1.0,
        "mid-glide the page is between where it was and where it is going \
         (from {from}, half {half}, destination {target})"
    );
    view.update(&mut vcx, |v, _| v.drive_page_glide_for_test(1.0));
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.page_glide_target_for_test(),
            None,
            "the glide retires when it arrives"
        );
        assert!(
            (v.page_scroll_offset_y_for_test() - target).abs() < 1.0,
            "and lands exactly on the destination (offset {}, destination \
             {target})",
            v.page_scroll_offset_y_for_test()
        );
    });
}

#[gpui::test]
fn space_keyboard_navigation_takes_the_page_from_a_glide_in_flight(cx: &mut TestAppContext) {
    // A glide owns `page_scroll` until something else takes it. The wheel and
    // the minimap say so (`note_scroll_activity` / `minimap_press`); the
    // *programmatic instant* scrolls did not, so a keyboard move (or a caret
    // reveal, or a post settling) wrote an offset the glide's next frame
    // overwrote from its own trajectory — the reader's own navigation undone,
    // repeatedly, until the glide landed.
    let stores = stub_stores_with_agents(cx, "s");
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());
    let long = "a long paragraph of the conversation so far. ".repeat(40);
    let mut a2 = fixture_assistant_post("a2", &long);
    a2.parent_action_id = Some("a1".into());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("a1", &long), a2], cx)
        });
    })
    .unwrap();

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(520.)));
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| v.scroll_page_to_end_for_test());
    vcx.run_until_parked();
    let from = view.read_with(&vcx, |v, _| v.page_scroll_offset_y_for_test());

    // Taken to the top of the conversation — a glide, in flight.
    vcx.update(|window, cx| {
        view.update(cx, |v, cx| v.navigate_to_action("a1".into(), window, cx));
    });
    vcx.run_until_parked();
    let target = view
        .read_with(&vcx, |v, _| v.page_glide_target_for_test())
        .expect("navigating arms a glide");

    // Mid-flight the reader navigates themselves: arrow into the conversation,
    // which reveals the focused post instantly.
    vcx.simulate_keystrokes("down");
    vcx.run_until_parked();
    let after = view.read_with(&vcx, |v, _| v.page_scroll_offset_y_for_test());
    assert!(
        (after - from).abs() > 1.0,
        "the keyboard reveal moved the page ({from} -> {after})"
    );
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.page_glide_target_for_test(),
            None,
            "an instant scroll takes the page from the glide, rather than \
             leaving it to be overwritten (destination {target})"
        );
    });

    // And the frame that would have continued the glide moves nothing.
    view.update(&mut vcx, |v, _| v.drive_page_glide_for_test(0.5));
    let settled = view.read_with(&vcx, |v, _| v.page_scroll_offset_y_for_test());
    assert!(
        (settled - after).abs() < 1.0,
        "the reader stays where their own navigation put them (offset \
         {settled}, expected {after}; the abandoned glide ran from {from} to \
         {target})"
    );
}

#[gpui::test]
fn space_branch_dot_takes_the_page_from_a_glide_in_flight(cx: &mut TestAppContext) {
    // A branch dot switches *horizontally*, so it writes no page offset of its
    // own — but it changes which document the page is scrolling through, which
    // makes an in-flight vertical glide's trajectory meaningless: it was aimed
    // at a `y` on the branch the reader just left, and kept dragging them
    // there. The other reader-driven takeovers already say so (the wheel via
    // `note_scroll_activity`, the minimap via `minimap_press`, the keyboard
    // through the instant-scroll door); the dot was the gap.
    let stores = stub_stores_with_agents(cx, "s");
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());
    let long = "a long paragraph of the conversation so far. ".repeat(40);
    // A short root keeps its band (and its branch dots) on screen at the top of
    // the page; the long replies below give the glide somewhere to travel.
    let mut a2 = fixture_assistant_post("a2", &long);
    a2.parent_action_id = Some("a1".into());
    let mut a3 = fixture_assistant_post("a3", &long);
    a3.parent_action_id = Some("a1".into());
    let mut a4 = fixture_user_post("a4", &long);
    a4.parent_action_id = Some("a2".into());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(
                vec![fixture_user_post("a1", "a short question"), a2, a4, a3],
                cx,
            )
        });
    })
    .unwrap();

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(520.)));
    vcx.run_until_parked();

    // Taken to a post far down the first branch — a glide, in flight.
    vcx.update(|window, cx| {
        view.update(cx, |v, cx| v.navigate_to_action("a4".into(), window, cx));
    });
    vcx.run_until_parked();
    let target = view
        .read_with(&vcx, |v, _| v.page_glide_target_for_test())
        .expect("navigating arms a glide");

    // Mid-flight, the reader switches branch by clicking its dot. Reduce-motion
    // is turned on *after* the glide was armed, so the horizontal switch lands
    // at once and the test can see it: no test dispatcher pumps the snap's
    // frame loop, and the switch is the vertical glide's whole problem.
    vcx.update(|_, cx| cx.set_reduce_motion(true));
    let dot = vcx
        .debug_bounds("space-dot-a1-1")
        .expect("a post with two replies paints a dot per branch");
    vcx.simulate_click(dot.center(), gpui::Modifiers::default());
    vcx.run_until_parked();
    let after = view.read_with(&vcx, |v, _| v.page_scroll_offset_y_for_test());
    let path = vcx.update(|window, cx| view.read(cx).selected_effective_path_for_test(window, cx));
    assert!(
        path.contains(&"a3".to_string()),
        "the dot click switched to the other branch ({path:?})"
    );
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.page_glide_target_for_test(),
            None,
            "the branch switch takes the page from the glide it interrupted \
             (destination {target})"
        );
    });

    // And the frame that would have continued it moves nothing.
    view.update(&mut vcx, |v, _| v.drive_page_glide_for_test(0.5));
    let settled = view.read_with(&vcx, |v, _| v.page_scroll_offset_y_for_test());
    assert!(
        (settled - after).abs() < 1.0,
        "the reader is left where the switch put them, not dragged toward a \
         `y` on the branch they left (offset {settled}, expected {after}, \
         abandoned destination {target})"
    );
}

#[gpui::test]
fn space_navigation_lands_at_once_under_reduce_motion(cx: &mut TestAppContext) {
    // The other half of the animated navigation: a reader who asked for less
    // motion gets the destination without the journey. `App::reduce_motion` is
    // gpui's own flag (it also stills every `Animation` element); nothing feeds
    // it from the platform at this pin, so honoring it is all we can do.
    cx.update(|cx| cx.set_reduce_motion(true));
    let stores = stub_stores_with_agents(cx, "s");
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());
    let long = "a long paragraph of the conversation so far. ".repeat(40);
    let mut a2 = fixture_assistant_post("a2", &long);
    a2.parent_action_id = Some("a1".into());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("a1", &long), a2], cx)
        });
    })
    .unwrap();

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(520.)));
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| v.scroll_page_to_end_for_test());
    vcx.run_until_parked();

    vcx.update(|window, cx| {
        view.update(cx, |v, cx| v.navigate_to_action("a1".into(), window, cx));
    });
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.page_glide_target_for_test(),
            None,
            "no glide is armed under reduce-motion"
        );
        assert!(
            v.page_scroll_offset_y_for_test().abs() < 1.0,
            "the page is at the destination already (offset {})",
            v.page_scroll_offset_y_for_test()
        );
    });
}

#[gpui::test]
fn space_quote_survives_the_round_trip_into_the_ask_path(cx: &mut TestAppContext) {
    // Ask-path integrity, end to end against a **real** core: selecting a
    // passage, quoting it into the tail draft, and posting must land a durable
    // `reference` edge at ordinal 1 whose snippet resolves — and the post body
    // must carry the matching `{{ embed 1 }}` marker.
    //
    // That pair is the whole contract the ask path rides on: wave 1's
    // `prepare_turn` expands exactly the structurally-recognized markers whose
    // ordinals the edges supply, so a marker+edge that agree here is a quote
    // the model will read (upstream expansion itself is pinned in app-core's
    // `upstream_context_expands_embed_markers_into_quotes`). `PostNode::
    // embed_map()` agreeing is the same guarantee on the render side.
    let (stores, core, _dir, space_id) = participants_scene(cx);

    // A persisted post to quote from.
    let seed = core
        .runtime()
        .block_on(core.post("the quick brown fox".into(), Some(space_id.clone())))
        .expect("seed post");

    let (window, view) = open_space(cx, &stores, Some(space_id.clone()));
    wait_until(cx, "transcript load", |cx| {
        view.read_with(cx, |v, _| v.post_count_for_test() > 0)
    });
    cx.update_window(window, |_, window, _| window.refresh())
        .unwrap();
    cx.run_until_parked();

    // Select "quick brown" in the seeded post and quote it.
    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| {
            v.select_in_post_for_test(&seed.action_id, 4..15, cx)
        });
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert!(
            v.has_post_selection_for_test(),
            "the seeded post's body is quotable"
        );
    });
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.quote(&eidola_gui::actions::Quote, window, cx));
    })
    .unwrap();
    cx.run_until_parked();

    // Type around the quote, then Post quietly (no model call needed — the
    // reference edge is what this test is about).
    let composer = view
        .read_with(cx, |v, _| v.composer_state_for_test())
        .expect("the quote activated a draft");
    let body = cx
        .update_window(window, |_, _, cx| composer.read(cx).value().to_string())
        .unwrap();
    assert!(
        body.contains("{{ embed 1 }}"),
        "marker in the draft: {body:?}"
    );
    cx.update_window(window, |_, _, cx| {
        composer.update(cx, |e, cx| {
            let v = format!("{body}\n\nwhat did you mean by this?");
            e.set_value(v, cx);
        });
    })
    .unwrap();
    dispatch_space_action(&view, window, cx, PostOnly);

    // The durable edge landed at ordinal 1, resolves its snippet, and the
    // rendered body still carries the marker addressing it.
    wait_until(cx, "the quoted post persists", |cx| {
        let _ = cx;
        core.runtime()
            .block_on(core.get_space_tree(space_id.clone()))
            .map(|nodes| nodes.iter().any(|n| !n.references.is_empty()))
            .unwrap_or(false)
    });
    let nodes = core
        .runtime()
        .block_on(core.get_space_tree(space_id.clone()))
        .expect("tree");
    let quoted = nodes
        .iter()
        .find(|n| !n.references.is_empty())
        .expect("a post carries the reference edge");

    assert_eq!(quoted.references.len(), 1);
    let r = &quoted.references[0];
    assert_eq!(
        r.ordinal, 1,
        "ordinal 0 is the reply edge; references start at 1"
    );
    assert_eq!(r.antecedent_action_id, seed.action_id);
    assert_eq!(
        r.snippet.as_deref(),
        Some("quick brown"),
        "the edge resolves the exact passage that was selected"
    );
    assert_eq!(
        quoted.embed_map().get(&1).map(String::as_str),
        Some("quick brown"),
        "the render-side embed map addresses it by the same ordinal"
    );
    let text: String = quoted
        .blocks
        .iter()
        .filter_map(|b| b.text.as_deref())
        .collect();
    assert!(
        text.contains("{{ embed 1 }}"),
        "the persisted body carries the marker the edge's ordinal addresses: {text:?}"
    );
    assert!(text.contains("what did you mean by this?"));

    drain_runtime(&core);
}

#[gpui::test]
fn space_composer_counts_the_bottom_breath_once(cx: &mut TestAppContext) {
    // The composer bar sizes itself to what its body draws — the editor's
    // laid-out text, the footnote rail's measured span, and the trailing
    // breath spacer — a straight sum of rendered elements, each counted
    // exactly once.
    //
    // The breath is one in-flow spacer, always the last thing in the body,
    // so a scrolled draft's last line stops a breath above the fold. The
    // rail's own bottom padding (`rail_pad`) stays *inside* the measured
    // flow-mark span, sized so pad-plus-breath totals a post's full bottom
    // pad — double-counting either term inflates the bar, and the whole
    // floating/docking runway with it: the editor's floor grows to swallow
    // the surplus, opening a gap between the last line of text and the
    // footnote rule.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_post_with_block("a1", "b1", "the quick brown fox")],
    );
    let breath = eidola_gui::space_view::composer::bottom_breath();

    // Phase 1 — no rail. The tail is the bare breath, drawn by the editor.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.seed_draft_quote_for_test(Some("a1"), "a plain reply", vec![], window, cx)
        });
    })
    .unwrap();
    cx.run_until_parked();
    cx.update_window(window, |_, window, _| window.refresh())
        .unwrap();
    cx.run_until_parked();

    let (reserved, rail, text) = view
        .read_with(cx, |v, cx| v.composer_geometry_for_test(cx))
        .expect("the seeded draft is the active composer");
    assert!(
        rail.abs() < 0.5,
        "no references, so the two flow marks coincide and the rail measures zero (was {rail})"
    );
    assert!(
        (reserved - (text + breath)).abs() < 0.5,
        "with no rail the bar reserves the editor's text plus one breath \
         (reserved {reserved}, text {text}, breath {breath})"
    );

    // Phase 2 — a populated rail. The reservation is the same sum with the
    // rail's measured span in the middle; the breath still follows outside
    // it, and the rail's own `rail_pad` tops the pair up to a full post pad.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.seed_draft_quote_for_test(
                Some("a1"),
                "a reply that quotes:\n\n{{ embed 1 }}\n\nand goes on",
                vec![(1, "kimi-k2", "quick brown")],
                window,
                cx,
            )
        });
    })
    .unwrap();
    cx.run_until_parked();
    cx.update_window(window, |_, window, _| window.refresh())
        .unwrap();
    cx.run_until_parked();

    let (reserved, rail, text) = view
        .read_with(cx, |v, cx| v.composer_geometry_for_test(cx))
        .expect("the quoting draft is the active composer");
    assert!(
        rail > 0.5,
        "the rail measures its rule, its row, and its own pad (rail {rail})"
    );
    assert!(
        (reserved - (text + rail + breath)).abs() < 0.5,
        "the bar reserves exactly what the body draws — text, the rail's \
         measured span, and one trailing breath (reserved {reserved}, \
         text {text}, rail {rail}, breath {breath})"
    );
}

#[gpui::test]
fn space_docked_composer_keeps_its_footnote_rail_on_screen(cx: &mut TestAppContext) {
    // The rail is the editor body's footer, so it must land on that body's
    // visible bottom edge in every configuration — floating, docked at the
    // end of the document, and docked mid-ramp. In compact layout the action
    // line follows it; side layout has no vertical action occupancy.
    //
    // Mid-ramp is where it used to fall off. `bar_h` is deliberately virtual
    // (the dock ramp grows it toward `full_h ≥ window` so the internal scroll
    // eases to zero), and the ramp carries the bar's *bottom* past the window
    // edge by up to `doc_reserve` on the way down. The painted quad is clipped
    // back to the window, but the rail was laid out at the end of the virtual
    // runway — so the quad's clip cut exactly the rail off, and it reappeared
    // only once the page reached the very end and the two bottoms coincided
    // again.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_post_with_block("a1", "b1", "the quick brown fox")],
    );

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(560.)));
    vcx.run_until_parked();

    vcx.update(|window, cx| {
        view.update(cx, |v, cx| {
            v.seed_draft_quote_for_test(
                Some("a1"),
                "a reply that quotes:\n\n{{ embed 1 }}\n\nand goes on",
                vec![(1, "kimi-k2", "quick brown")],
                window,
                cx,
            )
        });
    });
    vcx.run_until_parked();

    // Dock the composer at its "home" — slot top around 40% of the window,
    // which is the middle of the dock ramp (`composer_bar_h`'s `progress`
    // strictly between 0 and 1): the bar's virtual bottom is past the window
    // edge while its painted quad stops at it.
    vcx.update(|window, cx| view.update(cx, |v, cx| v.dock_active_draft_for_test(window, cx)));
    view.update(&mut vcx, |v, _| v.drive_page_glide_for_test(1.0));
    vcx.run_until_parked();
    vcx.update(|window, _| window.refresh());
    vcx.run_until_parked();

    // The test window has no CSD insets, so its content box is the size it
    // was resized to. This width is compact, so the editor's visible clip
    // ends above the bottom action bar, and the body's trailing breath
    // spacer separates the rail's lower mark from that clip edge.
    let win = 560.0;
    let rem_size = vcx.update(|window, _| window.rem_size().as_f32());
    let action_occupancy = view.read_with(&vcx, |v, _| {
        v.compact_action_occupancy_for_test(760.0, rem_size)
    });
    let breath = eidola_gui::space_view::composer::bottom_breath();
    view.read_with(&vcx, |v, cx| {
        assert!(
            !v.composer_overlayed_for_test(),
            "the composer must be docked for this to test the ramp"
        );
        let (_, rail, _) = v
            .composer_geometry_for_test(cx)
            .expect("the quoting draft is the active composer");
        assert!(rail > 0.5, "the seeded quote renders a rail (was {rail})");
        let bottom = v.composer_rail_bottom_for_test();
        let clip_bottom = win - action_occupancy;
        let expected = clip_bottom - breath;
        assert!(
            bottom >= expected - 0.5 && bottom <= clip_bottom + 0.5,
            "the docked rail stays on screen, its lower mark one breath above \
             the clip edge the action bar owns — not clipped below the window \
             (the bug), and not floated higher (an over-correction): rail \
             bottom {bottom}, expected {expected}..{clip_bottom}"
        );
    });
}

// ---------------------------------------------------------------------------
// Post context menus (task 28) — the pointer route to the verbs that already
// exist on the keyboard and in the Edit menu.
// ---------------------------------------------------------------------------

/// Right-click at `position` in `window`, the way a real pointer does — the
/// editor's own right-mouse-down handler is what opens the menu.
fn right_click(vcx: &mut VisualTestContext, position: Point<gpui::Pixels>) {
    vcx.simulate_event(gpui::MouseDownEvent {
        button: gpui::MouseButton::Right,
        position,
        modifiers: Modifiers::default(),
        click_count: 1,
        first_mouse: false,
    });
    vcx.run_until_parked();
}

/// A point inside the active composer's first painted line.
fn point_in_composer(view: &Entity<SpaceView>, vcx: &VisualTestContext) -> Point<gpui::Pixels> {
    let (x, y, h) = view
        .read_with(vcx, |v, cx| {
            v.composer_state_for_test().and_then(|e| {
                e.read(cx)
                    .debug_line_geometry()
                    .first()
                    .and_then(|(_, lines)| lines.first().copied())
            })
        })
        .expect("the composer's first painted line");
    gpui::point(px(x + 20.0), px(y + h.min(20.0) / 2.0))
}

/// A point inside post `node_id`'s first painted line.
fn point_in_post(
    view: &Entity<SpaceView>,
    vcx: &VisualTestContext,
    node_id: &str,
) -> Point<gpui::Pixels> {
    let (x, y, h) = view
        .read_with(vcx, |v, cx| {
            v.post_body_editor_for_test(node_id).and_then(|e| {
                e.read(cx)
                    .debug_line_geometry()
                    .first()
                    .and_then(|(_, lines)| lines.first().copied())
            })
        })
        .expect("the post's first painted line");
    gpui::point(px(x + 20.0), px(y + h.min(20.0) / 2.0))
}

#[gpui::test]
fn space_post_context_menu_offers_select_all_then_the_selection_verbs(cx: &mut TestAppContext) {
    // A read-only post affords Select All always; a live selection adds Copy,
    // and a *quotable* one adds the Edit menu's own quote pair. Nothing is
    // greyed — the menu builds only rows that do something.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_post_with_block("a1", "b1", "the quick brown fox")],
    );

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.run_until_parked();
    let at = point_in_post(&view, &vcx, "a1");

    right_click(&mut vcx, at);
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.context_menu_items_for_test(),
            Some(vec!["Select All".to_string()]),
            "an unselected read-only post offers only Select All"
        );
    });

    // Escape closes it (the composer's key handler consumes the first press).
    vcx.simulate_keystrokes("escape");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert!(
            v.context_menu_items_for_test().is_none(),
            "Escape dismisses the menu"
        );
    });

    // A press *outside* a selection collapses the caret to it — the platform
    // convention, and what makes a host's Paste land where the user pointed —
    // so the menu that opens is the unselected one.
    view.update(&mut vcx, |v, cx| {
        v.select_in_post_for_test("a1", 12..19, cx)
    });
    vcx.run_until_parked();
    right_click(&mut vcx, at);
    view.read_with(&vcx, |v, cx| {
        let sel = v
            .post_body_editor_for_test("a1")
            .expect("a1's editor")
            .read(cx)
            .selection();
        assert!(
            sel.is_collapsed(),
            "a press outside the selection places the caret there, got {sel:?}"
        );
        assert_eq!(
            v.context_menu_items_for_test(),
            Some(vec!["Select All".to_string()]),
            "and the menu that opens is the unselected one"
        );
    });

    // A press *inside* a quotable selection keeps it: Copy plus the quote
    // pair, then Select All. (Dismiss first — an open menu occludes what it
    // covers, so a second press on the same spot would land on the menu.)
    vcx.simulate_keystrokes("escape");
    vcx.run_until_parked();
    select_whole_post(&view, &mut vcx, "a1");
    right_click(&mut vcx, at);
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.context_menu_items_for_test(),
            Some(vec![
                "Copy".to_string(),
                "Quote".to_string(),
                "Quote in Reply".to_string(),
                "Quote Elsewhere…".to_string(),
                "Select All".to_string(),
            ]),
            "a quotable selection adds Copy and the Edit menu's quote pair"
        );
    });
}

/// Select a post's whole body, so any point in it is inside the selection.
fn select_whole_post(view: &Entity<SpaceView>, vcx: &mut VisualTestContext, node_id: &str) {
    let len = view
        .read_with(vcx, |v, cx| {
            v.post_body_editor_for_test(node_id)
                .map(|e| e.read(cx).value().len())
        })
        .expect("the post's editor");
    let id = node_id.to_string();
    view.update(vcx, |v, cx| v.select_in_post_for_test(&id, 0..len, cx));
    vcx.run_until_parked();
}

#[gpui::test]
fn space_post_context_menu_copies_and_selects_through_the_editor(cx: &mut TestAppContext) {
    // The clipboard verbs run the editor's own commands (the `perform` seam),
    // so the menu and ⌘C/⌘A cannot drift.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_post_with_block("a1", "b1", "the quick brown fox")],
    );

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.run_until_parked();
    let at = point_in_post(&view, &vcx, "a1");

    select_whole_post(&view, &mut vcx, "a1");
    right_click(&mut vcx, at);
    vcx.update(|window, cx| {
        view.update(cx, |v, cx| {
            assert!(v.activate_context_item_for_test("copy", window, cx));
        });
    });
    vcx.update(|_, cx| {
        let item = cx.read_from_clipboard().expect("Copy wrote the clipboard");
        assert_eq!(item.text().as_deref(), Some("the quick brown fox"));
    });
    view.read_with(&vcx, |v, _| {
        assert!(
            v.context_menu_items_for_test().is_none(),
            "choosing a row closes the menu"
        );
    });

    // Select All runs on the post's own editor, read-only and all.
    right_click(&mut vcx, at);
    vcx.update(|window, cx| {
        view.update(cx, |v, cx| {
            assert!(v.activate_context_item_for_test("select-all", window, cx));
        });
    });
    view.read_with(&vcx, |v, cx| {
        let editor = v.post_body_editor_for_test("a1").expect("a1's editor");
        let sel = editor.read(cx).selection();
        assert_eq!(
            (sel.lower_bound(), sel.upper_bound()),
            (0, "the quick brown fox".len()),
            "Select All covers the whole post"
        );
    });
}

#[gpui::test]
fn space_post_context_menu_quote_reuses_the_edit_menu_handler(cx: &mut TestAppContext) {
    // "Quote" is the Edit menu's own handler, not a parallel path: the same
    // pending reference lands on the active draft at ordinal 1.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_post_with_block("a1", "b1", "the quick brown fox")],
    );

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.run_until_parked();
    let at = point_in_post(&view, &vcx, "a1");
    select_whole_post(&view, &mut vcx, "a1");

    right_click(&mut vcx, at);
    vcx.update(|window, cx| {
        view.update(cx, |v, cx| {
            assert!(v.activate_context_item_for_test("quote", window, cx));
        });
    });
    vcx.run_until_parked();

    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.active_draft_references_for_test(),
            vec![(1u64, "the quick brown fox".to_string())],
            "the menu's Quote attaches exactly what Edit > Quote attaches"
        );
    });
}

#[gpui::test]
fn space_composer_context_menu_offers_the_editable_verbs(cx: &mut TestAppContext) {
    // An editable editor affords Paste and Select All always; Cut and Copy
    // join them once something is selected (an affordance appears when it is
    // actionable — the house rule the per-post verbs already follow).
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, None);
    cx.run_until_parked();
    set_space_composer_text(&view, window, cx, "a draft in progress");

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.run_until_parked();
    let composer = view
        .read_with(&vcx, |v, _| v.composer_state_for_test())
        .expect("a blank space opens with its composer");
    let at = point_in_composer(&view, &vcx);

    // Paste is clipboard-gated (see the test below), so seed one to keep this
    // test about the *selection* facts it is asserting.
    vcx.update(|_, cx| {
        cx.write_to_clipboard(gpui::ClipboardItem::new_string("pasteable".to_string()));
    });
    right_click(&mut vcx, at);
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.context_menu_items_for_test(),
            Some(vec!["Paste".to_string(), "Select All".to_string()]),
            "with a collapsed caret, Cut and Copy have nothing to act on"
        );
    });
    // Dismiss before re-opening: the menu occludes what it covers, so a
    // second right-click on the same spot would land on the menu itself.
    vcx.simulate_keystrokes("escape");
    vcx.run_until_parked();

    let len = composer.read_with(&vcx, |e, _| e.value().len());
    composer.update(&mut vcx, |e, cx| {
        e.apply_event_for_test(
            gpui_markdown_editor::EditorEvent::SetSelection(
                gpui_markdown_editor::Selection::range(0, len),
            ),
            cx,
        );
    });
    vcx.run_until_parked();
    right_click(&mut vcx, at);
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.context_menu_items_for_test(),
            Some(vec![
                "Cut".to_string(),
                "Copy".to_string(),
                "Paste".to_string(),
                "Select All".to_string(),
            ]),
            "a selection in an editable editor affords the full clipboard set"
        );
    });
}

#[gpui::test]
fn space_edit_session_escape_dismisses_the_menu_before_cancelling(cx: &mut TestAppContext) {
    // A right-click inside an inline Edit session opens the editable menu.
    // The first Escape must dismiss *only* the menu — the unsaved edit is the
    // thing at risk, and the post row's Escape handler is an inner element in
    // the dispatch path, so without a guard it cancels the session (restoring
    // the pre-edit buffer) on the very same press.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_post_with_block("a1", "b1", "the quick brown fox")],
    );

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.begin_edit("a1".into(), window, cx));
    })
    .unwrap();
    let editor = view
        .read_with(cx, |v, _| v.post_body_editor_for_test("a1"))
        .expect("the post's body editor");
    cx.update_window(window, |_, _, cx| {
        editor.update(cx, |e, cx| e.set_value("half-typed edit".to_string(), cx));
    })
    .unwrap();

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.run_until_parked();
    let at = point_in_post(&view, &vcx, "a1");

    right_click(&mut vcx, at);
    view.read_with(&vcx, |v, _| {
        assert!(
            v.context_menu_items_for_test().is_some(),
            "a right-click inside the session opens a menu over its editor"
        );
    });

    // First Escape: the menu goes, the session and its typing stay.
    vcx.simulate_keystrokes("escape");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert!(
            v.context_menu_items_for_test().is_none(),
            "the first Escape dismisses the menu"
        );
        assert_eq!(
            v.editing_action_id_for_test(),
            Some("a1".to_string()),
            "and leaves the edit session alive"
        );
    });
    vcx.update(|_, cx| {
        assert_eq!(
            editor.read(cx).value(),
            "half-typed edit",
            "the unsaved edit survives dismissing the menu"
        );
    });

    // Second Escape: now the session cancels and the buffer is restored.
    vcx.simulate_keystrokes("escape");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.editing_action_id_for_test(),
            None,
            "the second Escape cancels the edit"
        );
    });
    vcx.update(|_, cx| {
        assert_eq!(editor.read(cx).value(), "the quick brown fox");
    });
}

#[gpui::test]
fn space_composer_context_menu_offers_paste_only_when_there_is_text_to_paste(
    cx: &mut TestAppContext,
) {
    // Paste is resolved against the clipboard at open time, like the menu's
    // other two facts: `MarkdownEditorState::paste` returns without touching
    // the buffer when the clipboard holds no text, so an unconditional row
    // would be a visible affordance that does nothing — the one thing this
    // menu promises never to show.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, None);
    cx.run_until_parked();
    set_space_composer_text(&view, window, cx, "a draft in progress");

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.run_until_parked();
    let at = point_in_composer(&view, &vcx);

    // Nothing on the clipboard: no Paste row.
    right_click(&mut vcx, at);
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.context_menu_items_for_test(),
            Some(vec!["Select All".to_string()]),
            "an empty clipboard affords no Paste"
        );
    });
    vcx.simulate_keystrokes("escape");
    vcx.run_until_parked();

    // Text on the clipboard: the row appears.
    vcx.update(|_, cx| {
        cx.write_to_clipboard(gpui::ClipboardItem::new_string("pasteable".to_string()));
    });
    right_click(&mut vcx, at);
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.context_menu_items_for_test(),
            Some(vec!["Paste".to_string(), "Select All".to_string()]),
            "text on the clipboard affords Paste"
        );
    });
}

/// REGRESSION: the floating composer is an **opaque interactive surface** over
/// the page, so a press inside it must belong to it alone. It didn't: gpui
/// reports every hitbox under the cursor, so a drag-select in the composer also
/// landed in the post scrolled beneath it — selecting that post's text, and
/// (because a readonly post mid-drag-selection drives the page's
/// selection-autoscroll) scrolling the page up and down while you dragged.
#[gpui::test]
fn space_composer_drag_is_contained_and_does_not_scroll_the_page(cx: &mut TestAppContext) {
    let long = (1..=16)
        .map(|i| {
            format!(
                "Paragraph {i}. Sunlight is a fairly even mix across the visible spectrum, \
                 and as it crosses the atmosphere it meets molecules far smaller than its \
                 wavelength."
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("comp".into()));
    view.update(cx, |v, cx| {
        v.space().update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("p1", &long)], cx)
        });
    });
    cx.run_until_parked();
    open_space_draft(&view, window, cx, Some("p1"));

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(680.)));
    vcx.run_until_parked();
    let composer = view
        .read_with(&vcx, |v, _| v.composer_state_for_test())
        .expect("the branch tail opens a draft composer");
    composer.update(&mut vcx, |e, cx| {
        e.set_value("a draft being edited".to_string(), cx)
    });
    // Park the page at the top so the composer's slot sits far below the fold
    // and the bar renders *floating* over the post rather than docked under it.
    view.read_with(&vcx, |v, _| v.scroll_page_to_top_for_test());
    view.update(&mut vcx, |_, cx| cx.notify());
    vcx.run_until_parked();
    assert!(
        view.read_with(&vcx, |v, _| v.composer_overlayed_for_test()),
        "fixture drift: the composer must be floating over the page for this repro"
    );

    // A point that is inside the composer's own painted text **and** over the
    // post's painted text — the bug only exists where the two overlap, so the
    // point is derived from real geometry rather than assumed (and the search
    // failing is a loud fixture-drift panic, which is what gives this test its
    // teeth).
    let lines = |e: &Entity<gpui_markdown_editor::MarkdownEditorState>,
                 vcx: &VisualTestContext|
     -> Vec<(f32, f32, f32)> {
        e.read_with(vcx, |e, _| {
            e.debug_line_geometry()
                .iter()
                .flat_map(|(_, l)| l.iter().copied())
                .collect()
        })
    };
    let composer_lines = lines(&composer, &vcx);
    let post = view
        .read_with(&vcx, |v, _| v.post_body_editor_for_test("p1"))
        .expect("p1's editor");
    let post_lines = lines(&post, &vcx);
    let at = composer_lines
        .iter()
        .find_map(|&(x, cy, ch)| {
            post_lines
                .iter()
                .find(|&&(_, py, ph)| py < cy + ch && cy < py + ph)
                .map(|&(_, py, ph)| {
                    let top = cy.max(py);
                    let bottom = (cy + ch).min(py + ph);
                    gpui::point(px(x + 30.0), px((top + bottom) / 2.0))
                })
        })
        .unwrap_or_else(|| {
            panic!("fixture drift: composer {composer_lines:?} never overlaps post {post_lines:?}")
        });

    let before_scroll = view.read_with(&vcx, |v, _| v.page_scroll_offset_y_for_test());

    let end = gpui::point(at.x + px(220.), at.y + px(4.));
    vcx.simulate_event(gpui::MouseDownEvent {
        button: gpui::MouseButton::Left,
        position: at,
        modifiers: Modifiers::default(),
        click_count: 1,
        first_mouse: false,
    });
    vcx.simulate_event(gpui::MouseMoveEvent {
        position: end,
        pressed_button: Some(gpui::MouseButton::Left),
        modifiers: Modifiers::default(),
    });
    vcx.run_until_parked();
    vcx.simulate_event(gpui::MouseUpEvent {
        button: gpui::MouseButton::Left,
        position: end,
        modifiers: Modifiers::default(),
        click_count: 1,
    });
    vcx.run_until_parked();

    view.read_with(&vcx, |v, cx| {
        let sel = v
            .post_body_editor_for_test("p1")
            .expect("p1's editor")
            .read(cx)
            .selection();
        assert_eq!(
            sel.lower_bound(),
            sel.upper_bound(),
            "a drag inside the composer must leave the post beneath unselected, got {sel:?}"
        );
        assert_eq!(
            v.page_scroll_offset_y_for_test(),
            before_scroll,
            "and must not drive the page's selection-autoscroll"
        );
    });
}

// --- Trace visibility (task 34) -----------------------------------------

/// One turn's tool round, as `AppCore::space_traces` reports it.
fn trace_tool(name: &str, args: &str, result: Option<&str>, request: Option<&str>) -> TraceEntry {
    TraceEntry::Tool {
        action_id: format!("tc-{name}"),
        request_id: request.map(str::to_string),
        call_id: format!("call-{name}"),
        name: name.into(),
        arguments: args.into(),
        result: result.map(str::to_string),
    }
}

#[gpui::test]
fn space_trace_disclosure_is_collapsed_until_asked(cx: &mut TestAppContext) {
    // Quiet by default: a post that anchors activity carries one subordinate
    // line and nothing else until it is opened. Expansion lives on the shared
    // `Space` (like the thinking disclosure), so both windows on a space agree.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let mut a2 = fixture_assistant_post("a2", "the reply");
    a2.parent_action_id = Some("a1".into());
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_user_post("a1", "the ask"), a2],
    );
    let space = view.read_with(cx, |v, _| v.space().clone());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, _| {
            s.seed_traces_for_test(vec![PostTrace {
                id: "t1".into(),
                anchor_action_id: "a2".into(),
                participant_label: "kimi-k2".into(),
                unanswered: false,
                entries: vec![
                    trace_tool("list_branches", "{}", Some("2 branches"), Some("req-1")),
                    trace_tool(
                        "read_thread",
                        "{\"handle\":\"h1\"}",
                        Some("8 posts"),
                        Some("req-2"),
                    ),
                ],
            }]);
        });
    })
    .unwrap();

    // Collapsed: the disclosure exists but reveals nothing.
    space.read_with(cx, |s, _| {
        assert_eq!(
            s.traces_for("a2").len(),
            1,
            "the reply anchors its turn's trace"
        );
        assert!(s.traces_for("a1").is_empty(), "the ask anchors nothing");
        assert!(!s.trace_expanded("t1"));
    });

    // Expansion is keyed on the *turn*, not the post it hangs under — several
    // turns can land on one post, and opening one must not open its siblings.
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| s.toggle_trace("t1", cx));
    })
    .unwrap();
    space.read_with(cx, |s, _| {
        assert!(s.trace_expanded("t1"));
        assert!(
            !s.trace_expanded("a2"),
            "the anchor is not a disclosure key"
        );
    });
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| s.toggle_trace("t1", cx));
    })
    .unwrap();
    space.read_with(cx, |s, _| assert!(!s.trace_expanded("t1")));
}

#[gpui::test]
fn space_decline_renders_in_the_gap_under_the_post_it_answered(cx: &mut TestAppContext) {
    // The audit value of a decline is that a non-event is visible: the turn
    // wrote no post, so its disclosure hangs under the post it *answered*.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_quotable_space(&view, window, cx, vec![fixture_user_post("a1", "the ask")]);
    let space = view.read_with(cx, |v, _| v.space().clone());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, _| {
            s.seed_traces_for_test(vec![PostTrace {
                id: "t2".into(),
                anchor_action_id: "a1".into(),
                participant_label: "Mara".into(),
                unanswered: true,
                entries: vec![TraceEntry::Declined {
                    action_id: "d1".into(),
                    reason: Some("not my area".into()),
                }],
            }]);
        });
    })
    .unwrap();

    space.read_with(cx, |s, _| {
        let traces = s.traces_for("a1");
        assert_eq!(traces.len(), 1, "the decline hangs under the ask");
        assert!(traces[0].unanswered, "and is marked as leaving no post");
        assert_eq!(traces[0].participant_label, "Mara");
    });
}

#[gpui::test]
fn space_trace_row_deep_links_into_the_record(cx: &mut TestAppContext) {
    // Disclosure, not duplication: a round's line links to its own raw
    // exchange rather than re-rendering the payload in the reading column.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let mut a2 = fixture_assistant_post("a2", "the reply");
    a2.parent_action_id = Some("a1".into());
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_user_post("a1", "the ask"), a2],
    );
    let space = view.read_with(cx, |v, _| v.space().clone());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.seed_traces_for_test(vec![PostTrace {
                id: "t3".into(),
                anchor_action_id: "a2".into(),
                participant_label: "kimi-k2".into(),
                unanswered: false,
                entries: vec![trace_tool("read_post", "{}", Some("ok"), Some("req-7"))],
            }]);
            s.toggle_trace("t3", cx);
        });
    })
    .unwrap();

    view.read_with(cx, |v, _| {
        assert_eq!(v.last_record_request_for_test(), None)
    });
    // The row's own request id — read off the trace the view renders from —
    // is what the link follows.
    let request_id = space.read_with(cx, |s, _| match &s.traces_for("a2")[0].entries[0] {
        TraceEntry::Tool { request_id, .. } => request_id.clone().expect("a recorded exchange"),
        other => panic!("expected a tool round, got {other:?}"),
    });
    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| v.open_in_record(request_id, cx));
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert_eq!(
            v.last_record_request_for_test(),
            Some("req-7"),
            "the round's line opens that round's request in the Record"
        );
    });
}

#[gpui::test]
fn space_change_invalidates_the_trace_index(cx: &mut TestAppContext) {
    // A turn's rounds land with the turn, so a `Change::Space` must drop the
    // cached index; the next frame re-requests it.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_quotable_space(&view, window, cx, vec![fixture_user_post("a1", "the ask")]);
    let space = view.read_with(cx, |v, _| v.space().clone());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, _| {
            s.seed_traces_for_test(vec![PostTrace {
                id: "t4".into(),
                anchor_action_id: "a1".into(),
                participant_label: "Mara".into(),
                unanswered: true,
                entries: vec![TraceEntry::Declined {
                    action_id: "d1".into(),
                    reason: None,
                }],
            }]);
        });
    })
    .unwrap();
    space.read_with(cx, |s, _| assert_eq!(s.traces_for("a1").len(), 1));

    cx.update(|cx| {
        stores.spaces.update(cx, |st, cx| {
            st.notify_space_changed("s", ChangeOrigin::Caller, u64::MAX, cx)
        });
    });
    space.read_with(cx, |s, _| {
        assert!(
            s.traces_for("a1").is_empty(),
            "the cached index is dropped on the space changing"
        );
    });
}

// ---------------------------------------------------------------------------
// Wave B — focus and keyboard (task 12)
// ---------------------------------------------------------------------------

/// A fork at `a1`: two branches (`a2`, the spine, and `a3`), the second of
/// which continues to `a4`.
fn seed_branched_space(view: &Entity<SpaceView>, window: AnyWindowHandle, cx: &mut TestAppContext) {
    let mut a2 = fixture_assistant_post("a2", "the first branch");
    a2.parent_action_id = Some("a1".into());
    let mut a3 = fixture_assistant_post("a3", "the second branch");
    a3.parent_action_id = Some("a1".into());
    let mut a4 = fixture_user_post("a4", "deeper in the second branch");
    a4.parent_action_id = Some("a3".into());
    seed_quotable_space(
        view,
        window,
        cx,
        vec![fixture_user_post("a1", "the root post"), a2, a3, a4],
    );
}

#[gpui::test]
fn space_arrow_keys_walk_the_visible_rows(cx: &mut TestAppContext) {
    // Down/Up move through the selected path — what the eye does — and
    // Home/End reach its ends. Nothing is focused until an arrow enters the
    // conversation, which is what makes the model reachable without a pointer.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_user_post("a1", "the root post")],
    );
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(560.)));
    vcx.run_until_parked();

    view.read_with(&vcx, |v, _| {
        assert_eq!(v.tree_focus_for_test(), None, "nothing focused at rest");
    });

    vcx.simulate_keystrokes("down");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.tree_focus_for_test(),
            Some(("a1".to_string(), None)),
            "the first arrow enters the conversation at the top of the path"
        );
    });

    // A lone root: Down stops rather than wrapping or falling into the draft.
    vcx.simulate_keystrokes("down");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(v.tree_focus_for_test(), Some(("a1".to_string(), None)));
    });
}

#[gpui::test]
fn space_branch_moves_enter_and_leave_a_fork(cx: &mut TestAppContext) {
    // Right at the fork's anchor enters the next sibling branch; Left from the
    // first branch returns to the anchor; and at a post with no siblings both
    // are deliberate no-ops (Mike's decision — predictability over cleverness).
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_branched_space(&view, window, cx);
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(560.)));
    vcx.run_until_parked();

    vcx.simulate_keystrokes("down");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(v.tree_focus_for_test(), Some(("a1".to_string(), None)));
    });

    // The anchor stands alone on its own level — Left is a no-op there.
    vcx.simulate_keystrokes("left");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.tree_focus_for_test(),
            Some(("a1".to_string(), None)),
            "Left on a spine post does nothing"
        );
    });

    // Right enters the branch after the one the fork rests on.
    vcx.simulate_keystrokes("right");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.tree_focus_for_test(),
            Some(("a3".to_string(), None)),
            "Right at the fork enters the next sibling branch"
        );
    });

    // Down follows the newly selected branch, not the old one.
    vcx.simulate_keystrokes("down");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(v.tree_focus_for_test(), Some(("a4".to_string(), None)));
    });

    // Back up into the strip, then Left across it and out to the anchor.
    vcx.simulate_keystrokes("up left");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.tree_focus_for_test(),
            Some(("a2".to_string(), None)),
            "Left crosses to the previous branch"
        );
    });
    vcx.simulate_keystrokes("left");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.tree_focus_for_test(),
            Some(("a1".to_string(), None)),
            "Left from the first branch returns to the fork's anchor"
        );
    });
}

#[gpui::test]
fn space_enter_descends_into_the_posts_affordances_and_escape_climbs_out(cx: &mut TestAppContext) {
    // The two-level model, and the last two rungs of the Escape chain:
    // affordance → post → nothing. Escape never closes the window.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_user_post("a1", "the root post")],
    );
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(560.)));
    vcx.run_until_parked();

    vcx.simulate_keystrokes("down");
    vcx.run_until_parked();
    // A hover-gated verb is revealed by focus alone (audit S7): gpui suppresses
    // hover entirely under keyboard modality, so this is the only way a
    // keyboard user ever sees Edit / Regenerate.
    view.read_with(&vcx, |v, _| {
        assert!(
            v.post_affordances_revealed("a1"),
            "focus reveals the hover-gated affordance row"
        );
    });

    vcx.simulate_keystrokes("enter");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.tree_focus_for_test(),
            Some(("a1".to_string(), Some(0))),
            "Enter enters the post's affordance row"
        );
    });

    vcx.simulate_keystrokes("escape");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.tree_focus_for_test(),
            Some(("a1".to_string(), None)),
            "Escape steps back to the post"
        );
    });

    vcx.simulate_keystrokes("escape");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.tree_focus_for_test(),
            None,
            "Escape releases the conversation"
        );
    });
    assert_eq!(
        cx.windows().len(),
        1,
        "and Escape never closes the window, at any rung"
    );
}

#[gpui::test]
fn space_escape_yields_to_the_context_menu_before_the_focus_levels(cx: &mut TestAppContext) {
    // The full Escape priority chain: an open context menu speaks first (PR
    // #259's root-owner rule), and the same press must not also unwind a focus
    // level. The next press does.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_post_with_block("a1", "b1", "the quick brown fox")],
    );
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(560.)));
    vcx.run_until_parked();

    vcx.simulate_keystrokes("down enter");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(v.tree_focus_for_test(), Some(("a1".to_string(), Some(0))));
    });

    let at = point_in_post(&view, &vcx, "a1");
    right_click(&mut vcx, at);
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert!(
            v.context_menu_items_for_test().is_some(),
            "the menu is open"
        );
    });

    vcx.simulate_keystrokes("escape");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert!(
            v.context_menu_items_for_test().is_none(),
            "the first Escape closes the menu"
        );
        assert_eq!(
            v.tree_focus_for_test(),
            Some(("a1".to_string(), Some(0))),
            "…and does not also unwind the focus level"
        );
    });

    vcx.simulate_keystrokes("escape");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.tree_focus_for_test(),
            Some(("a1".to_string(), None)),
            "the next Escape takes the affordance rung"
        );
    });
}

#[gpui::test]
fn space_typing_jumps_to_the_trailing_draft_without_moving_the_page(cx: &mut TestAppContext) {
    // Task 38. A printable character with nothing composing starts the tail
    // draft, applies the character at the end of whatever it already held, and
    // leaves the space's scroll position exactly where the reader put it.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    // A document several windows tall, so the tail composer sits far below the
    // fold: without the suppression the caret reveal would drag the page down
    // to it, which is exactly what this shortcut promises not to do.
    let long = "a long paragraph that wraps several times over. ".repeat(24);
    let mut posts = vec![fixture_user_post("a1", &long)];
    for i in 2..8u32 {
        let id = format!("a{i}");
        let mut p = if i % 2 == 0 {
            fixture_assistant_post(&id, &long)
        } else {
            fixture_user_post(&id, &long)
        };
        p.parent_action_id = Some(format!("a{}", i - 1));
        posts.push(p);
    }
    seed_quotable_space(&view, window, cx, posts);
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(560.)));
    vcx.run_until_parked();

    // Seed the tail draft with text, then leave it (so nothing is composing).
    let draft = view
        .read_with(&vcx, |v, _| v.tail_draft_state_for_test())
        .expect("a docked tail draft");
    draft.update(&mut vcx, |e, cx| e.set_value("already here", cx));
    view.update(&mut vcx, |v, cx| v.retire_draft_for_test(cx));
    vcx.run_until_parked();

    let before = view.read_with(&vcx, |v, _| v.page_scroll_offset_y_for_test());

    vcx.simulate_keystrokes("x");
    vcx.run_until_parked();

    view.read_with(&vcx, |v, cx| {
        assert!(v.has_active_draft_for_test(), "the draft is now composing");
        let editor = v.composer_state_for_test().expect("the active composer");
        let text = editor.read(cx).value().to_string();
        assert_eq!(
            text, "already herex",
            "the character lands at the end of the existing text"
        );
        assert_eq!(
            editor.read(cx).cursor_offset(),
            text.len(),
            "…with the caret after it"
        );
        assert_eq!(
            v.page_scroll_offset_y_for_test(),
            before,
            "and the page did not move"
        );
    });
}

#[gpui::test]
fn space_typing_into_an_open_composer_is_not_intercepted(cx: &mut TestAppContext) {
    // The jump only exists for the "nothing is composing" state; with a draft
    // active the editor owns every keystroke.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, None);
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert!(
            v.has_active_draft_for_test(),
            "a blank space opens composing"
        );
    });
    view.update(&mut vcx, |v, cx| {
        let ed = v.composer_state_for_test().expect("the composer");
        ed.update(cx, |e, cx| e.set_value("typed by hand", cx));
    });
    vcx.simulate_keystrokes("x");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, cx| {
        // The editor's own input handler is the only writer here; our jump
        // must not have appended a second copy of the character.
        let text = v
            .composer_state_for_test()
            .expect("the composer")
            .read(cx)
            .value()
            .to_string();
        assert!(
            !text.ends_with("xx"),
            "the jump did not double the keystroke, got {text:?}"
        );
    });
}

#[gpui::test]
fn focus_visible_tracks_the_input_modality(cx: &mut TestAppContext) {
    // The ring's whole condition. gpui owns it (`Window::last_input_was_keyboard`,
    // read by `focus_visible`), so what this pins is that we depend on the
    // right signal: a keystroke arms it, a pointer press disarms it, and a
    // mouse user therefore never sees a ring.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_user_post("a1", "the root post")],
    );
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(560.)));
    vcx.run_until_parked();

    vcx.simulate_keystrokes("down");
    vcx.run_until_parked();
    vcx.update(|window, _| {
        assert!(
            window.last_input_was_keyboard(),
            "a keystroke puts the window in keyboard modality — the ring shows"
        );
    });

    let at = point_in_post(&view, &vcx, "a1");
    vcx.simulate_mouse_move(at, None, gpui::Modifiers::default());
    vcx.simulate_click(at, gpui::Modifiers::default());
    vcx.run_until_parked();
    vcx.update(|window, _| {
        assert!(
            !window.last_input_was_keyboard(),
            "a pointer press leaves pointer modality — the ring stays hidden"
        );
    });
}

/// The same post, one generation on: a new action id, the item id unchanged —
/// exactly what an edit or a regeneration lands in the reloaded transcript.
fn next_generation(mut post: PostNode, new_action_id: &str) -> PostNode {
    post.action_id = new_action_id.into();
    post.generation += 1;
    post.generation_count += 1;
    post
}

#[gpui::test]
fn space_tree_focus_follows_an_edited_post_to_its_new_generation(cx: &mut TestAppContext) {
    // Tree focus names a post by its **action** id, which an edit supersedes.
    // Without forwarding, the reloaded transcript no longer carries the id
    // focus sits on: `tree_target` can't find it in the path and returns
    // `None`, and the deliberate-no-op arm still reports the press as handled
    // — so every arrow reads as inert (and Enter finds no post, so it is dead
    // too) until the reader escapes out of the conversation entirely.
    // Threading already follows item identity; so does focus.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let a1 = fixture_user_post("a1", "the root post");
    let mut a2 = fixture_assistant_post("a2", "the reply");
    a2.parent_action_id = Some("a1".into());
    seed_quotable_space(&view, window, cx, vec![a1.clone(), a2.clone()]);
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(560.)));
    vcx.run_until_parked();

    vcx.simulate_keystrokes("down");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(v.tree_focus_for_test(), Some(("a1".to_string(), None)));
    });

    // The edit lands: a1's item is now action `a1b`, and a2 replies to it.
    let mut a2_edited = a2.clone();
    a2_edited.parent_action_id = Some("a1b".into());
    let space = view.read_with(&vcx, |v, _| v.space().clone());
    vcx.update(|_, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![next_generation(a1, "a1b"), a2_edited], cx)
        });
    });
    vcx.run_until_parked();

    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.tree_focus_for_test(),
            Some(("a1b".to_string(), None)),
            "focus followed the item to its current tip"
        );
    });

    // …and the arrows still walk from there.
    vcx.simulate_keystrokes("down");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.tree_focus_for_test(),
            Some(("a2".to_string(), None)),
            "Down moves on from the forwarded position"
        );
    });
}

#[gpui::test]
fn space_tree_focus_follows_a_regenerated_post_and_clears_when_it_vanishes(
    cx: &mut TestAppContext,
) {
    // The agent-side half of the same rule (a regeneration is a new generation
    // of the inference's item), plus the one case where clearing is the honest
    // answer: the item genuinely left the snapshot.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let a1 = fixture_user_post("a1", "the root post");
    let mut a2 = fixture_assistant_post("a2", "the reply");
    a2.parent_action_id = Some("a1".into());
    seed_quotable_space(&view, window, cx, vec![a1.clone(), a2.clone()]);
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(560.)));
    vcx.run_until_parked();

    vcx.simulate_keystrokes("down down");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(v.tree_focus_for_test(), Some(("a2".to_string(), None)));
    });

    let space = view.read_with(&vcx, |v, _| v.space().clone());
    vcx.update(|_, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![a1.clone(), next_generation(a2.clone(), "a2b")], cx)
        });
    });
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.tree_focus_for_test(),
            Some(("a2b".to_string(), None)),
            "focus followed the regenerated answer"
        );
    });
    vcx.simulate_keystrokes("up");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.tree_focus_for_test(),
            Some(("a1".to_string(), None)),
            "Up walks from the forwarded position"
        );
    });

    // The focused post's item leaves the snapshot entirely — the one case
    // where releasing focus is the honest outcome.
    vcx.simulate_keystrokes("down");
    vcx.run_until_parked();
    vcx.update(|_, cx| {
        space.update(cx, |s, cx| s.set_post_tree_for_test(vec![a1], cx));
    });
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.tree_focus_for_test(),
            None,
            "a post that no longer exists releases focus"
        );
    });
}

#[gpui::test]
fn space_tree_focus_releases_when_the_conversation_loses_focus(cx: &mut TestAppContext) {
    // `tree_focus` is bookkeeping; the window's focus is the truth. Tab away
    // (or click the composer) and the post's manually-drawn ring kept
    // painting beside the control that actually held focus — two apparent
    // focus targets — with the post's hover-gated verbs still revealed on a
    // post nobody was on. Observed from the real focus state at the head of
    // render, so there is no exit to forget.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_user_post("a1", "the root post")],
    );
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(560.)));
    vcx.run_until_parked();

    vcx.simulate_keystrokes("down");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(v.tree_focus_for_test(), Some(("a1".to_string(), None)));
        assert!(
            v.post_affordances_revealed("a1"),
            "focus reveals the post's verbs"
        );
    });

    // Tab on: the conversation no longer holds the window's focus.
    vcx.update(|window, cx| window.focus_next(cx));
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.tree_focus_for_test(),
            None,
            "the ring is released with the focus that justified it"
        );
        assert!(
            !v.post_affordances_revealed("a1"),
            "…and so are the verbs it revealed"
        );
    });
}

#[gpui::test]
fn space_typing_consumes_the_press_so_the_platform_cannot_retype_it(cx: &mut TestAppContext) {
    // The handler's `true` used to be discarded, so the press stayed
    // propagating. `gpui_macos::handle_key_event` reports a key as handled to
    // AppKit only when the callback comes back with `propagate == false`;
    // otherwise it falls through to `[[self inputContext] handleEvent:]`,
    // which hands the same native event to the installed input handler — the
    // editor this press just focused. `Window::dispatch_keystroke` has the
    // same shape (`if !result.propagate { return true }`, else
    // `input_handler.dispatch_input(...)`), and its return value is what makes
    // "consumed" assertable from a test at all.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_user_post("a1", "the root post")],
    );
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(560.)));
    vcx.run_until_parked();
    view.update(&mut vcx, |v, cx| v.retire_draft_for_test(cx));
    vcx.run_until_parked();

    let consumed = vcx
        .update(|window, cx| window.dispatch_keystroke(gpui::Keystroke::parse("x").unwrap(), cx));
    vcx.run_until_parked();
    assert!(
        consumed,
        "type-to-compose must consume the press; an unconsumed one is re-delivered \
         to the platform input context and typed a second time"
    );

    // The arrow keys are handled too, and must be consumed for the same reason.
    view.update(&mut vcx, |v, cx| v.retire_draft_for_test(cx));
    vcx.run_until_parked();
    let consumed = vcx.update(|window, cx| {
        window.dispatch_keystroke(gpui::Keystroke::parse("down").unwrap(), cx)
    });
    assert!(consumed, "a handled arrow is consumed too");
}

#[gpui::test]
fn space_typing_reseeds_the_composers_accessible_value(cx: &mut TestAppContext) {
    // `activate_draft` seeds the accessible value from the draft *as it stood*
    // — before the character that started the session. The §4 freeze rule
    // means nothing refreshes a focused composer's value, so a pre-keystroke
    // seed sticks until focus leaves.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_user_post("a1", "the root post")],
    );
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(560.)));
    vcx.run_until_parked();

    let draft = view
        .read_with(&vcx, |v, _| v.tail_draft_state_for_test())
        .expect("a docked tail draft");
    draft.update(&mut vcx, |e, cx| e.set_value("already here", cx));
    view.update(&mut vcx, |v, cx| v.retire_draft_for_test(cx));
    vcx.run_until_parked();

    vcx.simulate_keystrokes("x");
    vcx.run_until_parked();

    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.composer_aria_value_for_test().as_ref(),
            "already herex",
            "the accessible value includes the character that started the session"
        );
    });
}

#[gpui::test]
fn space_end_reaches_the_last_post_past_the_trailing_draft(cx: &mut TestAppContext) {
    // A space's selected path ends in the tail draft, so `End` resolved to the
    // composer and was then thrown away by a post-hoc filter — a no-op on the
    // one key whose whole job is "take me to the bottom". The filter belongs on
    // the path, not on the answer.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let mut a2 = fixture_assistant_post("a2", "the reply");
    a2.parent_action_id = Some("a1".into());
    let mut a3 = fixture_user_post("a3", "and on");
    a3.parent_action_id = Some("a2".into());
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_user_post("a1", "the root post"), a2, a3],
    );
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(560.)));
    vcx.run_until_parked();
    assert!(
        view.read_with(&vcx, |v, _| v.tail_draft_state_for_test().is_some()),
        "the fixture really does end in a draft"
    );

    vcx.simulate_keystrokes("down");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(v.tree_focus_for_test(), Some(("a1".to_string(), None)));
    });

    vcx.simulate_keystrokes("end");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.tree_focus_for_test(),
            Some(("a3".to_string(), None)),
            "End reaches the last *post*, not the draft below it"
        );
    });

    vcx.simulate_keystrokes("home");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(v.tree_focus_for_test(), Some(("a1".to_string(), None)));
    });
}

#[gpui::test]
fn space_tree_focus_releases_from_the_affordance_level_too(cx: &mut TestAppContext) {
    // The affordance level's verbs ride gpui's *implicit* focus handles, so
    // "is one of them still focused" is only answerable from a container:
    // `FocusHandle::contains_focused` resolves through the dispatch tree, where
    // an implicit descendant is an ordinary node.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_user_post("a1", "the root post")],
    );
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(560.)));
    vcx.run_until_parked();

    vcx.simulate_keystrokes("down enter");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(v.tree_focus_for_test(), Some(("a1".to_string(), Some(0))));
    });

    vcx.update(|window, cx| window.focus_next(cx));
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.tree_focus_for_test(),
            None,
            "Tab out of the affordance row releases the level"
        );
    });
}

#[gpui::test]
fn space_composer_resize_handle_adjusts_with_the_arrows(cx: &mut TestAppContext) {
    // A `Role::Slider` that only answers the mouse is a tab stop where the
    // arrows do nothing — the dead-stop shape the focus model exists to
    // prevent. The handle adjusts, in both directions, and clamps.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(560.)));
    vcx.run_until_parked();

    let start = view.read_with(&vcx, |v, _| v.composer_fraction_for_test());
    view.update(&mut vcx, |v, cx| v.nudge_composer_fraction(true, 560.0, cx));
    let taller = view.read_with(&vcx, |v, _| v.composer_fraction_for_test());
    assert!(taller > start, "Up grows the bar: {start} -> {taller}");

    view.update(&mut vcx, |v, cx| {
        v.nudge_composer_fraction(false, 560.0, cx)
    });
    let back = view.read_with(&vcx, |v, _| v.composer_fraction_for_test());
    assert!(
        (back - start).abs() < 1e-4,
        "Down undoes it: {back} vs {start}"
    );

    // Clamped, not unbounded.
    for _ in 0..100 {
        view.update(&mut vcx, |v, cx| v.nudge_composer_fraction(true, 560.0, cx));
    }
    let maxed = view.read_with(&vcx, |v, _| v.composer_fraction_for_test());
    assert!(
        maxed <= 0.85 + 1e-4,
        "clamped at the drag's own ceiling: {maxed}"
    );
    for _ in 0..100 {
        view.update(&mut vcx, |v, cx| {
            v.nudge_composer_fraction(false, 560.0, cx)
        });
    }
    let floored = view.read_with(&vcx, |v, _| v.composer_fraction_for_test());
    assert!(floored >= 0.1 - 1e-4, "and at its floor: {floored}");
}

#[gpui::test]
fn compact_resize_floor_rises_with_the_bars_fixed_surfaces(cx: &mut TestAppContext) {
    // The Exact clamp keeps the rendered bar above its fixed chrome — but a
    // *stored* fraction below that height would be a dead zone: arrow steps
    // that change the reported value while the bar doesn't move. The floor is
    // therefore applied to the stored fraction itself, so stored and rendered
    // agree at every position the slider can reach.
    let (_window, view, mut vcx) = open_floating_composer_scene(cx, "resize-floor");
    const WIN: f32 = 300.0;
    vcx.simulate_resize(gpui::size(px(760.), px(WIN)));
    vcx.run_until_parked();
    let editor = view
        .read_with(&vcx, |v, _| v.composer_state_for_test())
        .expect("the scene's draft is active");
    editor.update(&mut vcx, |e, cx| e.set_value("a line of draft", cx));
    vcx.run_until_parked();
    for _ in 0..2 {
        vcx.update(|window, _| window.refresh());
        vcx.run_until_parked();
    }

    let (chrome, gutters) = view.read_with(&vcx, |v, _| {
        let (top, total) = v.composer_gutter_contract_for_test();
        (v.composer_chrome_for_test(), total - top)
    });
    let floor = view.read_with(&vcx, |v, _| v.composer_min_fraction_for_test(WIN));
    assert!(
        floor > 0.1 + 1e-4,
        "precondition: the fixed surfaces exceed the nominal minimum \
         (floor {floor} in a {WIN}px window)"
    );
    assert!(
        floor > (chrome + gutters) / WIN + 1e-4,
        "the floor reserves an editor viewport beyond the fixed surfaces \
         (floor {floor}, chrome {chrome} + action bar {gutters})"
    );

    for _ in 0..100 {
        view.update(&mut vcx, |v, cx| v.nudge_composer_fraction(false, WIN, cx));
    }
    view.read_with(&vcx, |v, _| {
        let fraction = v.composer_fraction_for_test();
        assert!(
            (fraction - floor).abs() < 1e-4,
            "the stored fraction floors at the fixed surfaces' share \
             ({fraction} vs {floor})"
        );
        let bar = v.composer_float_bar_h_for_test(WIN);
        assert!(
            (bar - fraction * WIN).abs() < 0.5,
            "stored and rendered agree — no dead zone (bar {bar}, \
             fraction*win {})",
            fraction * WIN
        );
    });
}

#[gpui::test]
fn record_closing_a_detail_returns_focus_to_the_listing(cx: &mut TestAppContext) {
    // Opening a detail replaces the listing, so the element tracking the
    // section's `list_focus` unmounts while the window's focus still names that
    // handle — a dead handle: the dispatch tree has no node for it, so the
    // roving keys reach nothing and `focus_next` restarts the walk from the top
    // of the window. Backing out hands focus back to the listing.
    let stores = stub_stores(cx, |_| {});
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| RecordView::new(stores, window, cx))
    });
    view.update(cx, |v, cx| {
        v.select_section(RecordSection::Requests, cx);
        v.set_requests_for_test(vec![stub_request("req-1", 1_000)], false);
    });
    draw_frame(cx, window);
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.focus_listing_for_test(window, cx));
    })
    .unwrap();
    assert!(
        cx.update_window(window, |_, window, cx| {
            view.read_with(cx, |v, _| v.listing_is_focused_for_test(window))
        })
        .unwrap()
    );

    // Open a detail — the listing (and its handle's element) unmounts — and
    // then Tab, which is what a keyboard reader does to reach Back. Focus is
    // now on a detail affordance that the very next press unmounts.
    view.update(cx, |v, cx| v.open_request("req-1".into(), cx));
    draw_frame(cx, window);
    cx.update_window(window, |_, window, cx| window.focus_next(cx))
        .unwrap();
    assert!(
        !cx.update_window(window, |_, window, cx| {
            view.read_with(cx, |v, _| v.listing_is_focused_for_test(window))
        })
        .unwrap(),
        "the listing does not hold focus while its detail is open"
    );

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.close_detail(window, cx));
    })
    .unwrap();
    draw_frame(cx, window);
    assert!(
        cx.update_window(window, |_, window, cx| {
            view.read_with(cx, |v, _| v.listing_is_focused_for_test(window))
        })
        .unwrap(),
        "backing out puts the keyboard back on the listing it returns to"
    );
}

#[gpui::test]
fn space_an_open_picker_keeps_printables_out_of_the_conversation(cx: &mut TestAppContext) {
    // The key handler yielded to the context and band menus but not the
    // highlight picker, so with a picker open every arrow, Escape and printable
    // character fell through to the conversation behind it — and a printable
    // character *starts a draft*, which is a keystroke landing somewhere the
    // reader cannot see. One predicate now answers "is an overlay open" for
    // both the handler and the focus observation.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let a1 = fixture_post_with_block("a1", "b1", "the quick brown fox jumps");
    let mut a2 = fixture_assistant_post("a2", "first responder");
    a2.parent_action_id = Some("a1".into());
    let mut a3 = fixture_assistant_post("a3", "second responder");
    a3.parent_action_id = Some("a1".into());
    seed_quotable_space(&view, window, cx, vec![a1, a2, a3]);
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(560.)));
    vcx.run_until_parked();

    // Leave nothing composing, so a stray printable would be visible as a jump.
    view.update(&mut vcx, |v, cx| v.retire_draft_for_test(cx));
    vcx.run_until_parked();

    let space = view.read_with(&vcx, |v, _| v.space().clone());
    let incoming = |action: &str, lo: i64, hi: i64| eidola_app_core::IncomingReference {
        action_id: action.into(),
        space_id: "s".into(),
        ordinal: 1,
        content_block_id: Some("b1".into()),
        range_start: Some(lo),
        range_end: Some(hi),
        annotation: None,
        created_at: 0,
    };
    vcx.update(|_, cx| {
        space.update(cx, |s, _| {
            s.seed_incoming_references_for_test(
                "a1",
                vec![incoming("a2", 4, 15), incoming("a3", 10, 19)],
            );
        });
    });
    vcx.update(|window, cx| {
        view.update(cx, |v, cx| {
            v.click_highlight_for_test("a1", &[0, 1], window, cx)
        });
    });
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert!(
            v.highlight_picker_for_test().is_some(),
            "the picker is open"
        );
    });

    vcx.simulate_keystrokes("x");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert!(
            !v.has_active_draft_for_test(),
            "a printable character behind an open picker must not start a draft"
        );
        assert!(
            v.highlight_picker_for_test().is_some(),
            "…and the picker is still the thing that owns the keyboard"
        );
        assert_eq!(
            v.tree_focus_for_test(),
            None,
            "nor does an overlay move tree focus"
        );
    });
}

#[gpui::test]
fn space_affordance_index_follows_the_verb_tab_moved_to(cx: &mut TestAppContext) {
    // The level's index is bookkeeping; which verb has focus is the truth. Tab
    // walks Save → Cancel, and without a resync the index stayed on the verb
    // Enter entered, so the next Right cycled from a stale position. Every verb
    // of the post holding the level tracks its own slot handle, so "which one"
    // is a lookup.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_user_post("a1", "the root post")],
    );
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(560.)));
    vcx.run_until_parked();

    // An inline edit session is the only two-verb row (Save, Cancel), and it
    // makes the conversation's key handler yield outright — so the level is
    // established through the seam rather than through `Enter`, which cannot
    // reach it. Everything after that is the real Tab path.
    vcx.update(|window, cx| {
        view.update(cx, |v, cx| v.begin_edit("a1".into(), window, cx));
    });
    vcx.run_until_parked();
    vcx.update(|window, cx| {
        view.update(cx, |v, cx| v.focus_affordance_for_test("a1", 0, window, cx));
    });
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(v.tree_focus_for_test(), Some(("a1".to_string(), Some(0))));
    });

    // Tab on to the second verb: the level follows the focus.
    vcx.update(|window, cx| window.focus_next(cx));
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.tree_focus_for_test(),
            Some(("a1".to_string(), Some(1))),
            "the index resyncs to the verb Tab actually moved to"
        );
    });

    // …which is what `Left`/`Right` cycle from. (The cycle itself cannot be
    // exercised here: the same edit session that supplies the second verb makes
    // the key handler yield, so the arrows never reach the tree.)
    vcx.update(|window, cx| window.focus_next(cx));
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.tree_focus_for_test(),
            None,
            "and a Tab out of the row still releases the level"
        );
    });
}

#[gpui::test]
fn space_typing_accepts_a_multi_character_commit(cx: &mut TestAppContext) {
    // macOS builds `key_char` with `UCKeyTranslate` into a four-unit buffer, so
    // a dead-key or ligature layout can hand one keystroke several characters.
    // Refusing those dropped the text on the floor: with the conversation
    // focused there is no input handler to fall through to.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_user_post("a1", "the root post")],
    );
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(560.)));
    vcx.run_until_parked();
    view.update(&mut vcx, |v, cx| v.retire_draft_for_test(cx));
    vcx.run_until_parked();

    // `Keystroke::parse`'s `key->key_char` form is gpui's own way of spelling a
    // keystroke whose committed text differs from its key.
    let consumed = vcx.update(|window, cx| {
        window.dispatch_keystroke(gpui::Keystroke::parse("a->ábc").unwrap(), cx)
    });
    vcx.run_until_parked();
    assert!(consumed, "a multi-character commit is handled, not dropped");
    view.read_with(&vcx, |v, cx| {
        let editor = v.composer_state_for_test().expect("the composer opened");
        assert_eq!(
            editor.read(cx).value().to_string(),
            "ábc",
            "the whole commit lands, not its first character"
        );
    });
}

#[gpui::test]
fn backends_revealing_the_base_url_editor_focuses_it(cx: &mut TestAppContext) {
    // A keyboard-activated reveal must focus what it revealed: the affordance
    // that opened the editor unmounts as the row becomes a form, so without it
    // the window keeps a handle whose element is gone — the dispatch tree has
    // no node for it, and Tab restarts from the top of the window. Its
    // siblings here already did this; this row did not.
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.eidola_trust = Some(eidola_trust());
        s.backends = backends_fixture(true);
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SettingsView::new(stores.clone(), window, cx))
    });
    let pane = view.read_with(cx, |v, _| v.backends_pane());

    assert!(
        !cx.update_window(window, |_, window, cx| {
            pane.read_with(cx, |p, cx| p.base_url_input_is_focused(window, cx))
        })
        .unwrap()
    );

    cx.update_window(window, |_, window, cx| {
        pane.update(cx, |p, cx| p.begin_edit_base_url(window, cx));
    })
    .unwrap();
    assert!(
        cx.update_window(window, |_, window, cx| {
            pane.read_with(cx, |p, cx| p.base_url_input_is_focused(window, cx))
        })
        .unwrap(),
        "the revealed input has the keyboard"
    );
}

#[gpui::test]
fn library_ending_a_rename_hands_focus_back_to_the_listing(cx: &mut TestAppContext) {
    // `begin_rename` focuses the row's input; committing or cancelling removes
    // it. Without handing focus back the window keeps a dead handle — the same
    // class as the Record's detail close, and the same cure. The guard matters
    // too: a `Blur` ends a session *because* focus went elsewhere, so it must
    // not be dragged back.
    let stores = stub_stores(cx, |s| {
        s.spaces = vec![
            stub_space("s1", Some("Tides"), None, 1_000),
            stub_space("s2", Some("Borrow checker"), None, 900),
        ];
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| LibraryView::new(stores, window, cx))
    });
    draw_frame(cx, window);

    for commit in [false, true] {
        cx.update_window(window, |_, window, cx| {
            view.update(cx, |v, cx| {
                v.begin_rename("s1".into(), Some("Tides".into()), window, cx)
            });
        })
        .unwrap();
        assert!(
            !cx.update_window(window, |_, window, cx| {
                view.read_with(cx, |v, _| v.list_is_focused_for_test(window))
            })
            .unwrap(),
            "the rename input holds the keyboard while the session is open"
        );

        cx.update_window(window, |_, window, cx| {
            view.update(cx, |v, cx| {
                if commit {
                    v.commit_rename(window, cx)
                } else {
                    v.cancel_rename(window, cx)
                }
            });
        })
        .unwrap();
        assert!(
            cx.update_window(window, |_, window, cx| {
                view.read_with(cx, |v, _| v.list_is_focused_for_test(window))
            })
            .unwrap(),
            "ending the session (commit={commit}) returns focus to the listing"
        );
    }
}

#[gpui::test]
fn space_revealing_a_post_upward_clears_the_title_band(cx: &mut TestAppContext) {
    // The reveal used to treat the raw content height as visible, so a post
    // brought in from above landed flush with the window's top — underneath the
    // title band, which paints over it. The band's height is the document's own
    // top reserve, and the usable band starts below it.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let long = "a long paragraph that wraps several times over. ".repeat(24);
    let mut posts = vec![fixture_user_post("a1", &long)];
    for i in 2..8u32 {
        let id = format!("a{i}");
        let mut p = if i % 2 == 0 {
            fixture_assistant_post(&id, &long)
        } else {
            fixture_user_post(&id, &long)
        };
        p.parent_action_id = Some(format!("a{}", i - 1));
        posts.push(p);
    }
    seed_quotable_space(&view, window, cx, posts);
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(560.)));
    vcx.run_until_parked();

    // Walk to the end, then back up: the upward moves are the ones that pull a
    // post down from above the fold.
    vcx.simulate_keystrokes("end up up");
    vcx.run_until_parked();

    let (top, reserve) = vcx.update(|window, cx| {
        let v = view.read(cx);
        (
            v.focused_post_window_top_for_test(window, cx),
            v.doc_reserve_for_test(),
        )
    });
    let top = top.expect("a post is focused");
    assert!(
        top >= reserve,
        "the revealed post's top ({top}) must clear the title band ({reserve})"
    );
}

#[gpui::test]
fn space_navigating_beside_an_escaped_fork_draft_keeps_the_branch(cx: &mut TestAppContext) {
    // Escape a Reply draft beside an existing reply and the fork's selected
    // branch *is* that draft. The level's active index used to be remapped to
    // the nearest persisted sibling, which made that sibling a member of the
    // navigable path — so the next arrow resolved a target on the *other*
    // branch and `select_path_to` moved the reader's selection for them.
    // Navigation observes the tree; it does not steer it.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let mut a2 = fixture_assistant_post("a2", "the existing reply");
    a2.parent_action_id = Some("a1".into());
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_user_post("a1", "the root post"), a2],
    );
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(560.)));
    vcx.run_until_parked();

    // Reply at a1: a second branch, which is a draft. Then escape it — it stays
    // as an inactive inline draft, and it stays the selected branch.
    vcx.update(|window, cx| {
        view.update(cx, |v, cx| {
            v.create_draft_for_test(Some("a1".into()), window, cx)
        });
    });
    vcx.run_until_parked();
    // Type into it: an *empty* fork draft is pruned on escape, and the case
    // this pins is the one that survives — a reply you started and set aside.
    view.update(&mut vcx, |v, cx| {
        let ed = v.composer_state_for_test().expect("the fork draft");
        ed.update(cx, |e, cx| e.set_value("a reply in progress", cx));
    });
    view.update(&mut vcx, |v, cx| v.deactivate_for_test(cx));
    vcx.run_until_parked();

    // Enter the conversation and walk down from the fork's anchor.
    vcx.simulate_keystrokes("down");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(v.tree_focus_for_test(), Some(("a1".to_string(), None)));
    });

    vcx.simulate_keystrokes("down");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.tree_focus_for_test(),
            Some(("a1".to_string(), None)),
            "the selected branch below a1 is the draft, so there is no post to \
             move to — and certainly not one on a branch the reader did not choose"
        );
    });
}

#[gpui::test]
fn space_an_overlay_that_hands_off_focus_does_not_get_it_back(cx: &mut TestAppContext) {
    // A transient overlay borrows the keyboard and, at this pin, does not hand
    // it back — so the falling edge restores the conversation's focus level.
    // But a menu *item* can hand it somewhere on purpose: Reply creates a draft
    // and focuses its editor. Restoring then yanks focus out of the very thing
    // the reader asked the menu for. The borrow is returned only to a lender
    // who still has nothing.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_user_post("a1", "the root post")],
    );
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(760.), px(560.)));
    vcx.run_until_parked();
    view.update(&mut vcx, |v, cx| v.retire_draft_for_test(cx));
    vcx.run_until_parked();

    vcx.simulate_keystrokes("down");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(v.tree_focus_for_test(), Some(("a1".to_string(), None)));
    });

    // Open the band's Reply-or-Ask menu over that post, and let a frame record
    // the borrow.
    view.update(&mut vcx, |v, cx| v.set_band_menu_for_test(Some("a1"), cx));
    vcx.run_until_parked();

    // Pick Reply: a draft opens and its editor takes the keyboard, and the menu
    // closes on the same gesture.
    vcx.update(|window, cx| {
        view.update(cx, |v, cx| {
            v.create_draft_for_test(Some("a1".into()), window, cx);
            v.set_band_menu_for_test(None, cx);
        });
    });
    vcx.run_until_parked();

    let editor_focused = vcx.update(|window, cx| {
        let v = view.read(cx);
        let editor = v.composer_state_for_test().expect("the reply draft opened");
        let handle = editor.read(cx).focus_handle(cx);
        handle.is_focused(window)
    });
    assert!(
        editor_focused,
        "the draft the reader asked for keeps the keyboard"
    );
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.tree_focus_for_test(),
            None,
            "and the conversation's level is released, not restored over it"
        );
    });
}

// App lifecycle — the app outlives its windows (task 17, wave 2).
//
// The quit *policy* is the pure `lifecycle::quit_mode`, unit-tested in the
// module. What is machine-verifiable here is the window-registry half: with
// the mode production uses on macOS (and in windowless mode), every window
// can close and a fresh one opens cleanly afterwards — and what a
// reactivation does in each case. That the *process* survives is
// hand-verified on the real `.app`; `TestPlatform::quit` is a no-op, so no
// test can observe the difference.
// ---------------------------------------------------------------------------

#[gpui::test]
fn every_window_can_close_and_a_new_one_opens_cleanly(cx: &mut TestAppContext) {
    cx.update(|cx| cx.set_quit_mode(gpui::QuitMode::Explicit));
    let stores = stub_stores_with_config(cx);

    let (first, _) = open_space(cx, &stores, Some("s".into()));
    let (second, _) = open_space(cx, &stores, Some("s2".into()));
    cx.update(|cx| assert_eq!(cx.windows().len(), 2));

    for window in [first, second] {
        cx.update_window(window, |_, window, _| window.remove_window())
            .unwrap();
    }
    cx.run_until_parked();
    cx.update(|cx| {
        assert!(
            cx.windows().is_empty(),
            "both windows are gone; the app is still here"
        );
    });

    // The whole point: a window opens again over the same live stores.
    let (third, view) = open_space(cx, &stores, Some("s".into()));
    draw_window(cx, third);
    cx.update(|cx| assert_eq!(cx.windows().len(), 1));
    let space = view.read_with(cx, |v, _| v.space().clone());
    space.read_with(cx, |s, _| assert_eq!(s.id(), "s"));
}

#[gpui::test]
fn reactivating_with_no_windows_opens_the_door(cx: &mut TestAppContext) {
    use std::cell::Cell;
    use std::rc::Rc;

    let opened = Rc::new(Cell::new(0));
    let counter = opened.clone();
    cx.update(|cx| {
        eidola_gui::lifecycle::reactivate(cx, move |_| counter.set(counter.get() + 1));
    });
    assert_eq!(opened.get(), 1, "no window: reactivation opens one");
}

// Status item + retire-to-the-background (task 17, waves 3 and 3b).
//
// The quit decision itself is the pure `status_item::quit_intent`, unit-tested
// in the module. What only a real gpui app can answer is what retiring does to
// the window registry — and the AppKit half (NSStatusItem,
// setActivationPolicy:, SMAppService) is a live system surface no test
// platform reaches.
// ---------------------------------------------------------------------------

#[gpui::test]
fn retiring_closes_every_window_and_leaves_the_app_standing(cx: &mut TestAppContext) {
    // The window half of ⌘Q's retire-to-the-background. gpui drops a window
    // as its own `update_window` unwinds, so the registry is genuinely empty
    // when `close_all_windows` returns — and under `QuitMode::Explicit` (what
    // macOS uses) emptying it quits nothing, which is the whole trick: the
    // process, the stores and the loaded engines outlive the windows because
    // nothing on the quit path ever ran.
    cx.update(|cx| cx.set_quit_mode(gpui::QuitMode::Explicit));
    let stores = stub_stores_with_config(cx);

    open_space(cx, &stores, Some("s".into()));
    open_space(cx, &stores, Some("s2".into()));
    assert_eq!(cx.update(|cx| cx.windows().len()), 2);

    cx.update(eidola_gui::lifecycle::close_all_windows);
    cx.run_until_parked();
    assert_eq!(
        cx.update(|cx| cx.windows().len()),
        0,
        "every window closes, synchronously"
    );

    // …and the app is still an app: the way back in opens a window again.
    open_space(cx, &stores, Some("s3".into()));
    assert_eq!(
        cx.update(|cx| cx.windows().len()),
        1,
        "a retired app still opens windows"
    );
}

#[gpui::test]
fn retiring_from_a_windows_own_update_still_closes_that_window(cx: &mut TestAppContext) {
    // ⌘Q arrives with a window key, and `App::dispatch_action` routes an
    // action *through* the active window — so the Quit handler runs inside
    // that window's `update_window`, which has taken it out of the registry.
    // A sweep from there skips the one window the user is looking at, which
    // on the real app meant going `Accessory` with a window still on screen
    // and no menu bar (measured, before the defer). Both halves are pinned
    // here: the gpui fact, and the cure.
    cx.update(|cx| cx.set_quit_mode(gpui::QuitMode::Explicit));
    let stores = stub_stores_with_config(cx);
    let (first, _) = open_space(cx, &stores, Some("s".into()));
    open_space(cx, &stores, Some("s2".into()));

    cx.update_window(first, |_, _, cx| {
        assert_eq!(
            cx.windows().len(),
            2,
            "the updating window is still listed — the trap is that it is not *reachable*"
        );
        eidola_gui::lifecycle::close_all_windows(cx);
    })
    .unwrap();
    cx.run_until_parked();
    assert_eq!(
        cx.update(|cx| cx.windows().len()),
        1,
        "its own `update` refuses (the slot is taken), so an undeferred sweep leaves it standing"
    );

    let (third, _) = open_space(cx, &stores, Some("s3".into()));
    cx.update_window(third, |_, _, cx| {
        cx.defer(eidola_gui::lifecycle::close_all_windows);
    })
    .unwrap();
    cx.run_until_parked();
    assert_eq!(
        cx.update(|cx| cx.windows().len()),
        0,
        "deferred until the update unwinds, the sweep reaches every window"
    );
}

#[gpui::test]
fn a_window_open_set_in_motion_before_the_retire_is_abandoned(cx: &mut TestAppContext) {
    // The real shape of the bug: a click starts a window open that has to
    // `await` (a launch-time backend read, a create-from-template write, a
    // cross-space action lookup), ⌘Q retires the app inside that gap, and the
    // awaited work then opens its window anyway — the app comes back to the
    // front having just been told to go away.
    use eidola_gui::lifecycle::{abandon_pending_opens, intend_to_open};
    use std::cell::Cell;
    use std::rc::Rc;

    cx.update(|cx| cx.set_quit_mode(gpui::QuitMode::Explicit));
    let stores = stub_stores_with_config(cx);

    let opened = Rc::new(Cell::new(0usize));
    let sink = opened.clone();
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let task = cx.update(|cx| {
        let intent = intend_to_open(cx);
        cx.spawn(async move |cx: &mut gpui::AsyncApp| {
            let _ = rx.await;
            cx.update(|cx| {
                if intent.still_wanted(cx) {
                    sink.set(sink.get() + 1);
                }
            });
        })
    });

    // ⌘Q lands while the open is still in flight. `quit_or_retire` bumps the
    // generation synchronously, which is what the ticket compares against.
    cx.update(abandon_pending_opens);
    let _ = tx.send(());
    cx.run_until_parked();
    drop(task);
    assert_eq!(
        opened.get(),
        0,
        "the user's last instruction was ⌘Q, so the stale open is abandoned"
    );

    // A ticket taken *after* the retire is a fresh instruction and stands —
    // this is what keeps the status menu's Open / New Space working.
    let fresh = cx.update(|cx| intend_to_open(cx));
    assert!(
        cx.update(|cx| fresh.still_wanted(cx)),
        "an open asked for after the retire is not stale"
    );
    // And the app is still an app: opening still works.
    open_space(cx, &stores, Some("s".into()));
    assert_eq!(cx.update(|cx| cx.windows().len()), 1);
}

#[gpui::test]
fn a_ticket_survives_everything_except_a_retire(cx: &mut TestAppContext) {
    // The generation must move on retires and nothing else, or every async
    // open would be abandoned by unrelated activity.
    use eidola_gui::lifecycle::{abandon_pending_opens, close_all_windows, intend_to_open};

    cx.update(|cx| cx.set_quit_mode(gpui::QuitMode::Explicit));
    let stores = stub_stores_with_config(cx);
    let intent = cx.update(|cx| intend_to_open(cx));

    open_space(cx, &stores, Some("s".into()));
    cx.update(close_all_windows);
    cx.run_until_parked();
    assert!(
        cx.update(|cx| intent.still_wanted(cx)),
        "closing windows is not retiring — ⌘W must not abandon a pending open"
    );

    cx.update(abandon_pending_opens);
    assert!(!cx.update(|cx| intent.still_wanted(cx)));
}

#[gpui::test]
fn the_login_item_row_starts_from_the_system_and_invents_nothing(cx: &mut TestAppContext) {
    use eidola_gui::general::GeneralView;

    // "Open at login" has no store behind it — the system owns that state,
    // so the pane reads it at construction and holds no opinion of its own.
    // An error is only ever the system's words on a refused *write*, and
    // this test performs none deliberately: registering a login item is a
    // real change to the machine running the suite.
    let stores = stub_stores_with_config(cx);
    let (_window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| GeneralView::new(stores.config.clone(), window, cx))
    });

    view.read_with(cx, |v, _| {
        assert_eq!(v.login_item(), eidola_gui::login_item::state());
        assert!(
            v.login_item_error().is_none(),
            "nothing was written, so there is nothing to report"
        );
        // Whatever the system said, the row has honest copy for it — and an
        // unmanageable login item never reads as a plain "off".
        assert!(!v.login_item().description().is_empty());
    });
}

#[gpui::test]
fn reactivating_with_a_window_focuses_it_instead(cx: &mut TestAppContext) {
    use std::cell::Cell;
    use std::rc::Rc;

    let stores = stub_stores_with_config(cx);
    let (_window, _) = open_space(cx, &stores, Some("s".into()));

    let opened = Rc::new(Cell::new(0));
    let counter = opened.clone();
    cx.update(|cx| {
        eidola_gui::lifecycle::reactivate(cx, move |_| counter.set(counter.get() + 1));
    });
    assert_eq!(
        opened.get(),
        0,
        "an existing window is focused, not duplicated"
    );
    cx.update(|cx| assert_eq!(cx.windows().len(), 1));
}

// ---------------------------------------------------------------------------
// The space inspector (task 26.2) — the per-space settings panel that splits
// the space window. Its doors are the Space menu item and ⌥⌘I only; the space
// itself carries no visual toggle.
// ---------------------------------------------------------------------------

/// A space whose settings the stub store already holds, so the panel renders
/// its rows rather than a spinner.
fn stub_stores_with_space_settings(
    cx: &mut TestAppContext,
    space_id: &str,
    settings: eidola_app_core::SpaceSettings,
) -> Stores {
    let space_id = space_id.to_string();
    stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.space_settings = Some((space_id, settings));
    })
}

#[gpui::test]
fn space_inspector_is_closed_until_asked_and_is_per_window(cx: &mut TestAppContext) {
    let stores =
        stub_stores_with_space_settings(cx, "s1", eidola_app_core::SpaceSettings::default());
    let (window_a, view_a) = open_space(cx, &stores, Some("s1".into()));
    let (_window_b, view_b) = open_space(cx, &stores, Some("s1".into()));

    assert!(
        !view_a.read_with(cx, |v, _| v.inspector_open_for_test()),
        "a space opens with no inspector"
    );

    dispatch_space_action(&view_a, window_a, cx, eidola_gui::actions::ToggleInspector);
    assert!(view_a.read_with(cx, |v, _| v.inspector_open_for_test()));
    assert!(
        !view_b.read_with(cx, |v, _| v.inspector_open_for_test()),
        "open state is per window — two windows on one space are two vantage points"
    );

    // The same action closes it again (one item, both directions).
    dispatch_space_action(&view_a, window_a, cx, eidola_gui::actions::ToggleInspector);
    assert!(!view_a.read_with(cx, |v, _| v.inspector_open_for_test()));
}

#[gpui::test]
fn space_inspector_split_narrows_the_conversation_page(cx: &mut TestAppContext) {
    let stores =
        stub_stores_with_space_settings(cx, "s1", eidola_app_core::SpaceSettings::default());
    let (window, view) = open_space(cx, &stores, Some("s1".into()));
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(1000.), px(700.)));
    vcx.run_until_parked();

    let width_of = |vcx: &mut VisualTestContext| {
        vcx.update_window(window, |_, window, cx| {
            view.read(cx).page_width_for_test(window)
        })
        .unwrap()
    };
    let closed = width_of(&mut vcx);
    assert!(
        (closed - 1000.0).abs() < 1.0,
        "closed: the page is the window"
    );

    vcx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.set_inspector_open_for_test(true, window, cx))
    })
    .unwrap();
    vcx.run_until_parked();
    let open = width_of(&mut vcx);
    assert!(
        (open - (1000.0 - 320.0)).abs() < 1.0,
        "a real split: the conversation compresses by the panel's column, got {open}"
    );
}

#[gpui::test]
fn space_inspector_overlays_a_window_too_narrow_to_split(cx: &mut TestAppContext) {
    let stores =
        stub_stores_with_space_settings(cx, "s1", eidola_app_core::SpaceSettings::default());
    let (window, view) = open_space(cx, &stores, Some("s1".into()));
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_resize(gpui::size(px(700.), px(700.)));
    vcx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.set_inspector_open_for_test(true, window, cx))
    })
    .unwrap();
    vcx.run_until_parked();

    let width = vcx
        .update_window(window, |_, window, cx| {
            view.read(cx).page_width_for_test(window)
        })
        .unwrap();
    assert!(
        (width - 700.0).abs() < 1.0,
        "below the floor the panel covers the page instead of squeezing it further, got {width}"
    );
}

#[gpui::test]
fn space_inspector_shows_the_settings_the_store_holds(cx: &mut TestAppContext) {
    let stores = stub_stores_with_space_settings(
        cx,
        "s1",
        eidola_app_core::SpaceSettings {
            cascade_limit: 6,
            router_model: Some("gemma4-31b@eidola".into()),
            ..Default::default()
        },
    );
    let (window, view) = open_space(cx, &stores, Some("s1".into()));
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.set_inspector_open_for_test(true, window, cx))
    })
    .unwrap();
    cx.run_until_parked();

    view.read_with(cx, |v, cx| {
        assert_eq!(v.inspector_cascade_for_test(cx), Some(6));
        assert_eq!(
            v.inspector_router_for_test(cx),
            Some(Some("gemma4-31b@eidola".into()))
        );
    });
}

#[gpui::test]
fn space_inspector_title_field_renames_the_space(cx: &mut TestAppContext) {
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.space_settings = Some(("s1".into(), eidola_app_core::SpaceSettings::default()));
        s.spaces = vec![stub_space("s1", Some("Tides"), None, 0)];
    });
    let (window, view) = open_space(cx, &stores, Some("s1".into()));
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.set_inspector_open_for_test(true, window, cx))
    })
    .unwrap();
    draw_window(cx, window);

    // The field arrives seeded with the space's real title (the Library index's).
    let state = view
        .read_with(cx, |v, _| v.inspector_title_state_for_test())
        .expect("the title field exists once the inspector renders");
    assert_eq!(state.read_with(cx, |s, _| s.value().to_string()), "Tides");

    cx.update_window(window, |_, window, cx| {
        state.update(cx, |s, cx| {
            s.set_value("Tides and the moon".to_string(), window, cx)
        });
        view.update(cx, |v, cx| v.inspector_commit_title(cx));
    })
    .unwrap();
    cx.run_until_parked();

    stores.spaces.read_with(cx, |s, _| {
        assert_eq!(
            s.list()
                .iter()
                .find(|r| r.id == "s1")
                .and_then(|r| r.title.clone()),
            Some("Tides and the moon".into()),
            "the title writes through the Library index, like the Library's own rename"
        );
    });
}

#[gpui::test]
fn space_inspector_typing_in_a_field_does_not_jump_to_the_composer(cx: &mut TestAppContext) {
    // Type-to-compose (task 38) treats any printable in the window as "start
    // the trailing draft" — and it *consumes* the press, so without a guard a
    // character typed into the inspector's title field would land in the
    // composer instead. The scene is one where the jump really would fire: a
    // transcript with a docked tail draft, and nothing composing.
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.space_settings = Some(("s1".into(), eidola_app_core::SpaceSettings::default()));
        s.spaces = vec![stub_space("s1", Some("Tides"), None, 0)];
    });
    let (window, view) = open_space(cx, &stores, Some("s1".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_user_post("a1", "the question")],
    );
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.set_inspector_open_for_test(true, window, cx))
    })
    .unwrap();
    draw_window(cx, window);
    assert!(
        view.read_with(cx, |v, _| v.draft_count_for_test() > 0),
        "precondition: a tail draft exists for the jump to land in"
    );

    let state = view
        .read_with(cx, |v, _| v.inspector_title_state_for_test())
        .expect("title field");
    cx.update_window(window, |_, window, cx| {
        state.update(cx, |s, cx| s.focus(window, cx));
    })
    .unwrap();
    cx.run_until_parked();

    // The real keystroke path: the press reaches the focused field, and the
    // conversation must not treat it as the type-to-compose jump.
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_keystrokes("x");
    vcx.run_until_parked();

    assert!(
        !view.read_with(&vcx, |v, _| v.has_active_draft_for_test()),
        "no draft was started behind the panel"
    );
    assert!(
        state
            .read_with(&vcx, |s, _| s.value().to_string())
            .contains('x'),
        "the character went into the field the reader is typing in"
    );
}

#[gpui::test]
fn space_inspector_escape_closes_its_router_picker(cx: &mut TestAppContext) {
    let stores =
        stub_stores_with_space_settings(cx, "s1", eidola_app_core::SpaceSettings::default());
    let (window, view) = open_space(cx, &stores, Some("s1".into()));
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.set_inspector_open_for_test(true, window, cx);
            v.inspector_toggle_router_picker_for_test(cx);
        })
    })
    .unwrap();
    draw_window(cx, window);
    assert!(view.read_with(cx, |v, _| v.inspector_picker_open_for_test()));

    let focus = view.read_with(cx, |v, _| v.focus_handle());
    cx.update_window(window, |_, window, cx| {
        window.focus(&focus, cx);
    })
    .unwrap();
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_keystrokes("escape");
    vcx.run_until_parked();
    assert!(
        !view.read_with(&vcx, |v, _| v.inspector_picker_open_for_test()),
        "the view root closes the picker, the same rung the context menu owns"
    );
}

/// The same Escape rung, for the Participants section's model dropdown: it is a
/// transient overlay over the same panel, so the conversation's key handler
/// yields to it and the view root has to be what closes it.
#[gpui::test]
fn space_inspector_escape_closes_a_participants_model_picker(cx: &mut TestAppContext) {
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.space_settings = Some(("s1".into(), eidola_app_core::SpaceSettings::default()));
    });
    let (window, view) = open_space(cx, &stores, Some("s1".into()));
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.set_inspector_open_for_test(true, window, cx);
            v.inspector_begin_add_participant(window, cx);
            v.inspector_open_add_picker_for_test(cx);
        })
    })
    .unwrap();
    draw_window(cx, window);
    assert!(view.read_with(cx, |v, _| v.inspector_participant_picker_open_for_test()));

    let focus = view.read_with(cx, |v, _| v.focus_handle());
    cx.update_window(window, |_, window, cx| {
        window.focus(&focus, cx);
    })
    .unwrap();
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_keystrokes("escape");
    vcx.run_until_parked();
    assert!(
        !view.read_with(&vcx, |v, _| v.inspector_participant_picker_open_for_test()),
        "the view root closes it, beside the router picker's rung"
    );
}

/// A scene with one agent participant, a post to hang a tail draft off, and the
/// inspector open — the surface the dropdown-ownership tests below drive.
fn inspector_participants_stub_scene(
    cx: &mut TestAppContext,
) -> (AnyWindowHandle, Entity<SpaceView>) {
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.space_settings = Some(("s1".into(), eidola_app_core::SpaceSettings::default()));
        s.participants = Some((
            "s1".into(),
            vec![eidola_app_core::ParticipantInfo {
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
            }],
        ));
    });
    let (window, view) = open_space(cx, &stores, Some("s1".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_user_post("a1", "the question")],
    );
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.set_inspector_open_for_test(true, window, cx));
    })
    .unwrap();
    draw_window(cx, window);
    assert!(
        view.read_with(cx, |v, _| v.draft_count_for_test() > 0),
        "precondition: a tail draft exists for the jump to land in"
    );
    (window, view)
}

/// Type a printable at the view root and report whether it reached
/// type-to-compose — the observable consequence of `transient_overlay_open`,
/// which yields the conversation's keyboard to whatever it believes is on top.
fn printable_reaches_the_conversation(
    cx: &mut TestAppContext,
    window: AnyWindowHandle,
    view: &Entity<SpaceView>,
) -> bool {
    let focus = view.read_with(cx, |v, _| v.focus_handle());
    cx.update_window(window, |_, window, cx| {
        window.focus(&focus, cx);
    })
    .unwrap();
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_keystrokes("x");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| v.has_active_draft_for_test())
}

/// **A form that leaves takes its dropdown with it** (Codex review, PR #278).
/// The model dropdown is painted *inside* a form, so any transition that
/// unmounts that form takes the dropdown off the screen — and the flag
/// recording the click must stop claiming an overlay owns the keyboard, or the
/// window goes on yielding every arrow, Escape and printable to something
/// nobody can see. "Save these participants as a template…" is such a
/// transition: it drops the add form (and the editor) to make room for itself.
#[gpui::test]
fn space_inspector_a_form_that_leaves_takes_its_dropdown_with_it(cx: &mut TestAppContext) {
    let (window, view) = inspector_participants_stub_scene(cx);
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.inspector_begin_add_participant(window, cx);
            v.inspector_open_add_picker_for_test(cx);
        })
    })
    .unwrap();
    draw_window(cx, window);
    assert!(view.read_with(cx, |v, _| v.inspector_participant_picker_open_for_test()));

    // The reader reaches past the open dropdown for "Save these participants as
    // a template…", then thinks better of it.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.inspector_begin_template(window, cx);
            v.inspector_cancel_template(window, cx);
        })
    })
    .unwrap();
    draw_window(cx, window);

    assert!(
        !view.read_with(cx, |v, _| v.inspector_participant_picker_open_for_test()),
        "the dropdown went with the form that painted it"
    );
    assert!(
        printable_reaches_the_conversation(cx, window, &view),
        "the conversation answers the keyboard again, with no Escape needed"
    );
}

/// The same rule at the sibling transition: **Remove** unmounts the very editor
/// its dropdown hangs in.
#[gpui::test]
fn space_inspector_removing_a_participant_takes_its_dropdown_with_it(cx: &mut TestAppContext) {
    let (window, view) = inspector_participants_stub_scene(cx);
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.inspector_toggle_participant("agent-1", window, cx);
            v.inspector_open_editor_picker_for_test(cx);
        })
    })
    .unwrap();
    draw_window(cx, window);
    assert!(view.read_with(cx, |v, _| v.inspector_participant_picker_open_for_test()));

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.inspector_remove_participant("agent-1", window, cx)
        });
    })
    .unwrap();
    draw_window(cx, window);

    assert!(
        !view.read_with(cx, |v, _| v.inspector_participant_picker_open_for_test()),
        "the dropdown went with the editor Remove closed"
    );
    assert!(
        printable_reaches_the_conversation(cx, window, &view),
        "the conversation answers the keyboard again"
    );
}

/// And when the row itself leaves under the open editor — another window
/// removed the participant, so the store's re-list drops it — the editor is
/// unmounted by the roster rather than by a verb. It owes the same two things:
/// its dropdown, and the keyboard its field was holding.
#[gpui::test]
fn space_inspector_a_participant_leaving_the_roster_retires_its_editor(cx: &mut TestAppContext) {
    let (stores, core, _dir, space) = participants_scene(cx);
    let (window, view) = open_participants_inspector(cx, &stores, &space);
    wait_until(cx, "participants load", |cx| {
        !participant_labels(&stores, &space, cx).is_empty()
    });
    let agent = stores.participants.read_with(cx, |s, _| {
        s.list(&space)
            .iter()
            .find(|p| p.kind == "agent")
            .expect("a seeded agent")
            .id
            .clone()
    });

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.inspector_toggle_participant(&agent, window, cx);
            v.inspector_open_editor_picker_for_test(cx);
        })
    })
    .unwrap();
    draw_window(cx, window);
    assert!(view.read_with(cx, |v, _| v.inspector_participant_picker_open_for_test()));
    // The editor's own field is holding the keyboard — the borrow the unmount
    // has to give back.
    let name = view
        .read_with(cx, |v, _| v.inspector_editing_label_state())
        .expect("the disclosure is the editor");
    cx.update_window(window, |_, window, cx| {
        name.update(cx, |s, cx| s.focus(window, cx));
    })
    .unwrap();
    cx.run_until_parked();

    // The other window's removal, straight through the store — this view never
    // hears a verb, only the re-list.
    let (sid, pid) = (space.clone(), agent.clone());
    stores
        .participants
        .update(cx, |s, cx| s.remove(sid, pid, cx));
    wait_until(cx, "participant removed", |cx| {
        !participant_labels(&stores, &space, cx).is_empty()
            && stores
                .participants
                .read_with(cx, |s, _| s.list(&space).iter().all(|p| p.id != agent))
    });
    draw_window(cx, window);

    assert_eq!(
        view.read_with(cx, |v, _| v
            .inspector_editing_participant()
            .map(str::to_string)),
        None,
        "an editor whose row is gone is not an editor"
    );
    assert!(
        !view.read_with(cx, |v, _| v.inspector_participant_picker_open_for_test()),
        "and its dropdown went with it"
    );
    // The keyboard is back with the conversation — no click on the page first,
    // which is the whole point of the handoff: the field it was in is gone.
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_keystrokes("x");
    vcx.run_until_parked();
    assert!(
        view.read_with(&vcx, |v, _| v.has_active_draft_for_test()),
        "the press reached the conversation instead of a dead input"
    );
    drain_runtime(&core);
}

/// Type-to-compose must yield to **every** field the panel paints, not just the
/// title: a character typed into a participant's system prompt is text, and the
/// jump would consume the press and apply it to the composer instead.
#[gpui::test]
fn space_inspector_typing_in_a_system_prompt_does_not_jump_to_the_composer(
    cx: &mut TestAppContext,
) {
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.space_settings = Some(("s1".into(), eidola_app_core::SpaceSettings::default()));
        s.participants = Some((
            "s1".into(),
            vec![eidola_app_core::ParticipantInfo {
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
            }],
        ));
    });
    let (window, view) = open_space(cx, &stores, Some("s1".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_user_post("a1", "the question")],
    );
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.set_inspector_open_for_test(true, window, cx);
            v.inspector_toggle_participant("agent-1", window, cx);
        })
    })
    .unwrap();
    draw_window(cx, window);
    assert!(
        view.read_with(cx, |v, _| v.draft_count_for_test() > 0),
        "precondition: a tail draft exists for the jump to land in"
    );

    let prompt = view
        .read_with(cx, |v, _| v.inspector_editing_prompt_state())
        .expect("the disclosure is the editor");
    cx.update_window(window, |_, window, cx| {
        prompt.update(cx, |s, cx| s.focus(window, cx));
    })
    .unwrap();
    cx.run_until_parked();

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_keystrokes("x");
    vcx.run_until_parked();

    assert!(
        !view.read_with(&vcx, |v, _| v.has_active_draft_for_test()),
        "no draft was started behind the panel"
    );
    assert!(
        prompt
            .read_with(&vcx, |s, _| s.value().to_string())
            .contains('x'),
        "the character went into the charter the reader is writing"
    );
}

/// **A view must observe the stores it renders** (STATE.md). The inspector's
/// rows come from `SpaceSettingsStore`, whose every announcement is
/// asynchronous — the panel's opening `ensure` load completing, each write's
/// re-read, a bus-driven refresh — and none of them repaints this window by any
/// other route. Without the subscription the panel sat on "Loading…" until some
/// unrelated event happened to redraw it.
///
/// Driven against a real core so the load genuinely lands, with the window
/// quiesced first so the only thing that can notify the view afterwards is the
/// settings load itself.
#[gpui::test]
fn space_inspector_repaints_when_its_settings_land(cx: &mut TestAppContext) {
    let (stores, core, _dir, space) = participants_scene(cx);
    let (window, view) = open_space(cx, &stores, Some(space.clone()));
    // Let the transcript load and every launch-time refresh settle, so nothing
    // else is in flight to repaint this window.
    wait_until(cx, "the window quiesces", |cx| {
        view.read_with(cx, |v, cx| {
            !matches!(
                v.space().read(cx).transcript(),
                eidola_gui::loadable::Loadable::NotLoaded | eidola_gui::loadable::Loadable::Loading
            )
        })
    });

    let repaints = std::rc::Rc::new(std::cell::Cell::new(0usize));
    let counter = repaints.clone();
    let _sub = cx.update(|cx| cx.observe(&view, move |_, _| counter.set(counter.get() + 1)));

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.set_inspector_open_for_test(true, window, cx))
    })
    .unwrap();
    // The toggle's own notify is not what this test is about.
    repaints.set(0);
    assert!(
        stores
            .space_settings
            .read_with(cx, |s, _| s.settings(&space).is_loading()),
        "precondition: opening the panel started the settings load"
    );

    wait_until(cx, "the settings load lands", |cx| {
        stores
            .space_settings
            .read_with(cx, |s, _| s.settings(&space).has_value())
    });
    assert!(
        repaints.get() > 0,
        "the settings landing must schedule a repaint — otherwise the panel \
         keeps rendering the spinner it drew before the load completed"
    );

    drain_runtime(&core);
}

/// The empty-title rejection must leave the *field* honest too. Clearing the
/// title and committing is read as a mistake rather than an intent (a space's
/// title is generated), so nothing is written — but the field is then showing a
/// blank where the space still has a name. The repair is
/// `sync_inspector_title`, and it only fires when the seed disagrees with the
/// stored title, so the rejection has to invalidate the seed; leaving it equal
/// made the sync read "already synchronized" and the field stayed blank.
#[gpui::test]
fn space_inspector_rejecting_a_blank_title_restores_the_stored_one(cx: &mut TestAppContext) {
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.space_settings = Some(("s1".into(), eidola_app_core::SpaceSettings::default()));
        s.spaces = vec![stub_space("s1", Some("Tides"), None, 0)];
    });
    let (window, view) = open_space(cx, &stores, Some("s1".into()));
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.set_inspector_open_for_test(true, window, cx))
    })
    .unwrap();
    draw_window(cx, window);

    let state = view
        .read_with(cx, |v, _| v.inspector_title_state_for_test())
        .expect("the title field exists once the inspector renders");
    assert_eq!(state.read_with(cx, |s, _| s.value().to_string()), "Tides");

    // Clear the field and commit — the seam both `PressEnter` and `Blur` route
    // through, with the field left unfocused as it is after a blur.
    cx.update_window(window, |_, window, cx| {
        state.update(cx, |s, cx| s.set_value(String::new(), window, cx));
        view.update(cx, |v, cx| v.inspector_commit_title(cx));
    })
    .unwrap();
    draw_window(cx, window);

    stores.spaces.read_with(cx, |s, _| {
        assert_eq!(
            s.list()
                .iter()
                .find(|r| r.id == "s1")
                .and_then(|r| r.title.clone()),
            Some("Tides".into()),
            "a blanked field writes nothing — the space keeps its name"
        );
    });
    assert_eq!(
        state.read_with(cx, |s, _| s.value().to_string()),
        "Tides",
        "and the field is repaired from the stored title rather than left blank"
    );
}

/// **A refused rename takes the field back with it.** The title row writes
/// through `SpacesStore::rename`, which edits the cached row optimistically —
/// so a write the database refuses used to leave the field (and the Library, and
/// the window title) reading a name nothing persisted. With the store
/// reconciling its index from the re-list on every exit, the panel is honest for
/// free: the stored title moves back under the field, the seed disagrees with
/// it, and `sync_inspector_title` re-seeds — and the refusal itself surfaces in
/// the panel's one op-error banner.
///
/// Driven through the store's own settle (the production
/// `stores::settle_mutation`), which is what a refused write's completion
/// applies; that the real path reaches it is
/// `a_refused_rename_reconciles_the_index_and_surfaces_the_refusal`
/// (`tests/stores.rs`).
#[gpui::test]
fn space_inspector_a_refused_rename_restores_the_stored_title(cx: &mut TestAppContext) {
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.space_settings = Some(("s1".into(), eidola_app_core::SpaceSettings::default()));
        s.spaces = vec![stub_space("s1", Some("Tides"), None, 0)];
    });
    let (window, view) = open_space(cx, &stores, Some("s1".into()));
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.set_inspector_open_for_test(true, window, cx))
    })
    .unwrap();
    draw_window(cx, window);

    let state = view
        .read_with(cx, |v, _| v.inspector_title_state_for_test())
        .expect("the title field exists once the inspector renders");

    // Type a new name and commit it: the optimistic edit lands everywhere at
    // once, which is the whole point of it.
    cx.update_window(window, |_, window, cx| {
        state.update(cx, |s, cx| s.set_value("Nile".to_string(), window, cx));
        view.update(cx, |v, cx| v.inspector_commit_title(cx));
    })
    .unwrap();
    draw_window(cx, window);
    assert_eq!(state.read_with(cx, |s, _| s.value().to_string()), "Nile");

    // …and the write is refused. The store settles: the re-list puts the stored
    // title back, the refusal lands in the op-error slot tagged with this space.
    stores.spaces.update(cx, |s, cx| {
        s.settle_for_test(
            Some("s1".into()),
            Ok(vec![stub_space("s1", Some("Tides"), None, 0)]),
            Some("Couldn't rename this space: space not found: s1"),
            cx,
        )
    });
    draw_window(cx, window);

    assert_eq!(
        state.read_with(cx, |s, _| s.value().to_string()),
        "Tides",
        "the field re-seeds from the stored title rather than keeping a name \
         nothing persisted"
    );
    stores.spaces.read_with(cx, |s, _| {
        assert!(
            s.op_error_for("s1").is_some(),
            "and the refusal is the panel's to show"
        );
    });
}

/// Two quick presses of the cascade stepper are two increments. Each press
/// derives `next` from the store's cached value, and a write's round trip is
/// slower than a second click — so the store has to advance its own snapshot as
/// the write leaves (see `SpaceSettingsStore::write_then_reread`). Without that
/// both presses stepped from the same cached 4, the second write superseded the
/// first, and 4 + 1 + 1 persisted as 5.
#[gpui::test]
fn space_inspector_two_quick_cascade_steps_both_count(cx: &mut TestAppContext) {
    let (stores, core, _dir, space) = participants_scene(cx);
    let (window, view) = open_space(cx, &stores, Some(space.clone()));
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.set_inspector_open_for_test(true, window, cx))
    })
    .unwrap();
    wait_until(cx, "the settings load", |cx| {
        stores
            .space_settings
            .read_with(cx, |s, _| s.settings(&space).has_value())
    });
    assert_eq!(
        view.read_with(cx, |v, cx| v.inspector_cascade_for_test(cx)),
        Some(eidola_app_core::DEFAULT_CASCADE_LIMIT),
        "precondition: the panel shows the space's stored limit"
    );

    // Two presses with nothing settling in between — the first write has not
    // round-tripped when the second derives its value.
    view.update(cx, |v, cx| v.inspector_step_cascade_for_test(1, cx));
    view.update(cx, |v, cx| v.inspector_step_cascade_for_test(1, cx));

    let target = eidola_app_core::DEFAULT_CASCADE_LIMIT + 2;
    // Wait on the **durable row**, not the store's cell — the optimistic value
    // is there the moment the press is handled, and what this test is about is
    // that both increments survive the round trip.
    let durable = |core: &std::sync::Arc<AppCore>, space: &str| {
        core.runtime()
            .block_on(core.space_settings(space.to_string()))
            .expect("read the space's settings back")
            .cascade_limit
    };
    wait_until(cx, "both steps persist", |_| {
        durable(&core, &space) == target
    });
    assert_eq!(
        view.read_with(cx, |v, cx| v.inspector_cascade_for_test(cx)),
        Some(target),
        "and the panel agrees with the row it wrote"
    );

    drain_runtime(&core);
}

/// **Focus comes back from the panel** — the `RecordView::close_detail` rule
/// (see the a11y section of the crate's AGENTS.md), here for a surface that
/// unmounts while its retained `InputState` still holds the window's keyboard.
/// The handle survives the close (the field is a view field, minted once), so
/// the window goes on naming a focus node the dispatch tree no longer has:
/// keystrokes reach nothing, `focus_next` restarts from the top of the window,
/// and type-to-compose is dead until a click revives it.
#[gpui::test]
fn space_inspector_closing_returns_focus_from_its_field(cx: &mut TestAppContext) {
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.space_settings = Some(("s1".into(), eidola_app_core::SpaceSettings::default()));
        s.spaces = vec![stub_space("s1", Some("Tides"), None, 0)];
    });
    let (window, view) = open_space(cx, &stores, Some("s1".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_user_post("a1", "the question")],
    );
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.set_inspector_open_for_test(true, window, cx))
    })
    .unwrap();
    draw_window(cx, window);

    let state = view
        .read_with(cx, |v, _| v.inspector_title_state_for_test())
        .expect("title field");
    cx.update_window(window, |_, window, cx| {
        state.update(cx, |s, cx| s.focus(window, cx));
    })
    .unwrap();
    cx.run_until_parked();

    let field = state.read_with(cx, |s, cx| s.focus_handle(cx));
    let root = view.read_with(cx, |v, _| v.focus_handle());
    cx.update_window(window, |_, window, _| {
        assert!(
            field.is_focused(window),
            "precondition: the reader is typing in the panel"
        );
    })
    .unwrap();

    // ⌥⌘I from inside the field: the action's dispatch path runs from the
    // focused field up through the space root, which owns the handler. (The
    // scrim's click listener funnels through the same `set_inspector_open`.)
    dispatch_space_action(&view, window, cx, eidola_gui::actions::ToggleInspector);

    cx.update_window(window, |_, window, _| {
        assert!(
            !field.is_focused(window),
            "the unmounted field must not keep the window's keyboard"
        );
        assert!(
            root.is_focused(window),
            "the conversation the panel annotated takes it back"
        );
    })
    .unwrap();

    // The consequence, end to end: the keyboard works again.
    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_keystrokes("x");
    vcx.run_until_parked();
    assert!(
        view.read_with(&vcx, |v, _| v.has_active_draft_for_test()),
        "type-to-compose reaches the conversation again"
    );
}

/// The other direction of the same rule: **restore only from a panel that
/// actually holds the keyboard.** A reader composing with the inspector merely
/// open (its field never focused) must keep their caret when the panel closes —
/// `overlay_borrowed_focus`'s "a borrow is only returned by a lender who still
/// has nothing", applied to a surface that never borrowed at all.
#[gpui::test]
fn space_inspector_closing_leaves_a_composing_reader_alone(cx: &mut TestAppContext) {
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.space_settings = Some(("s1".into(), eidola_app_core::SpaceSettings::default()));
        s.spaces = vec![stub_space("s1", Some("Tides"), None, 0)];
    });
    let (window, view) = open_space(cx, &stores, Some("s1".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_user_post("a1", "the question")],
    );
    open_space_draft(&view, window, cx, None);
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.set_inspector_open_for_test(true, window, cx))
    })
    .unwrap();
    draw_window(cx, window);

    let editor = view
        .read_with(cx, |v, _| v.composer_state_for_test())
        .expect("an active draft (composer) is open");
    let caret = editor.read_with(cx, |e, _| e.focus_handle.clone());
    cx.update_window(window, |_, window, _| {
        assert!(
            caret.is_focused(window),
            "precondition: the reader is composing, not editing a setting"
        );
    })
    .unwrap();

    dispatch_space_action(&view, window, cx, eidola_gui::actions::ToggleInspector);

    cx.update_window(window, |_, window, _| {
        assert!(
            caret.is_focused(window),
            "closing a panel that never held the keyboard must not take it"
        );
    })
    .unwrap();
    assert!(
        view.read_with(cx, |v, _| v.has_active_draft_for_test()),
        "and the draft is still the composing session"
    );
}

/// The same class as `space_inspector_repaints_when_its_settings_land`, closed
/// for the panel's **whole** read set: the router picker's options come from
/// `ModelsStore` (via `router_field` → `model_groups`), whose catalog fetches
/// land well after the window has drawn — a remote catalog arriving with an
/// open picker must repaint it, or the remote refs simply are not in the list
/// until something unrelated redraws the window. The Space Templates pane (the
/// other `router_field` consumer) already observed this store; the space view
/// was the one that did not.
#[gpui::test]
fn space_inspector_repaints_when_a_model_catalog_lands(cx: &mut TestAppContext) {
    let (stores, core, _dir, space) = participants_scene(cx);
    let (window, view) = open_space(cx, &stores, Some(space.clone()));
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.set_inspector_open_for_test(true, window, cx);
            v.inspector_toggle_router_picker_for_test(cx);
        })
    })
    .unwrap();
    // Quiesce: the settings load and every launch-time refresh settle, so the
    // only thing left to notify this window is the fetch started below.
    wait_until(cx, "the window quiesces", |cx| {
        stores
            .space_settings
            .read_with(cx, |s, _| s.settings(&space).has_value())
            && stores.models.read_with(cx, |s, _| !s.models().is_loading())
    });
    assert!(
        view.read_with(cx, |v, _| v.inspector_picker_open_for_test()),
        "precondition: the reader is looking at the open router picker"
    );

    let repaints = std::rc::Rc::new(std::cell::Cell::new(0usize));
    let counter = repaints.clone();
    let _sub = cx.update(|cx| cx.observe(&view, move |_, _| counter.set(counter.get() + 1)));

    // A catalog refresh: the registry read, then the per-backend fetch. Both
    // land asynchronously; the synchronous `to_loading` notify that starts it
    // is not what this test is about.
    stores.models.update(cx, |s, cx| s.refresh(cx));
    repaints.set(0);

    wait_until(cx, "the catalog fetch completes", |cx| {
        stores
            .models
            .read_with(cx, |s, _| s.models().error().is_some())
    });
    assert!(
        repaints.get() > 0,
        "a catalog landing must schedule a repaint — otherwise the open picker \
         keeps showing the options it drew before the fetch completed"
    );

    drain_runtime(&core);
}

// ---------------------------------------------------------------------------
// Cross-space references (task 37) — creation, handoff, denied follow
// ---------------------------------------------------------------------------

#[gpui::test]
fn space_quoting_elsewhere_names_who_will_see_the_passage(cx: &mut TestAppContext) {
    // The creation UI's whole point: choosing a destination makes the app say
    // what choosing it *means* — the passage becomes visible to everyone in
    // that conversation, as a copy. The reader is the flow-control point, so
    // the sentence stands between the choice and the write.
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.spaces = vec![
            stub_space("s", Some("Here"), None, 2),
            stub_space("other", Some("Tides"), None, 1),
        ];
    });
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_post_with_block("a1", "b1", "the quick brown fox")],
    );

    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| v.select_in_post_for_test("a1", 4..15, cx));
    })
    .unwrap();
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.quote_elsewhere(&eidola_gui::actions::QuoteElsewhere, window, cx)
        });
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert_eq!(
            v.quote_destination_for_test(),
            Some(None),
            "the picker opens on the list of conversations, with nothing said yet"
        );
    });

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.arm_quote_destination_for_test("other", "Tides", window, cx)
        });
    })
    .unwrap();
    let statement = view
        .read_with(cx, |v, _| v.quote_destination_for_test())
        .expect("still open")
        .expect("a destination is armed");
    assert!(
        statement.contains("visible to everyone in Tides"),
        "the statement names the destination: {statement}"
    );
    assert!(
        statement.contains("copies"),
        "…and says the passage is copied, not linked: {statement}"
    );

    // Confirming hands the quote to the destination's own entity — nothing
    // durable, and nothing written into this conversation.
    let destination = stores
        .spaces
        .update(cx, |spaces, cx| spaces.open("other".into(), cx));
    // Read the mailbox *inside* the same update: the confirm's deferred window
    // open runs when this closure's effects flush, and that window drains the
    // offer at once — which is the feature working, not a race to observe.
    let handed_over = cx
        .update_window(window, |_, window, cx| {
            view.update(cx, |v, cx| v.confirm_quote_destination_for_test(window, cx));
            destination.read(cx).has_offered_quote()
        })
        .unwrap();
    assert!(
        handed_over,
        "the passage is handed to the destination's entity — the only thing two windows share"
    );
    // …and the deferred window open takes it from there — once that window has
    // actually read the conversation. Until then the offer waits rather than
    // guessing a tail (see `space_a_quote_into_an_unloaded_conversation_waits_
    // for_its_tail`), which in stub mode is where it would stay.
    cx.run_until_parked();
    destination.read_with(cx, |space, _| {
        assert!(
            space.has_offered_quote(),
            "the window opened, but it has not read the conversation yet"
        );
    });
    cx.update(|cx| {
        destination.update(cx, |s, cx| {
            s.set_post_tree_for_test(
                vec![fixture_post_with_block(
                    "d1",
                    "db1",
                    "a conversation over here",
                )],
                cx,
            );
        });
    });
    cx.run_until_parked();
    destination.read_with(cx, |space, _| {
        assert!(
            !space.has_offered_quote(),
            "the transcript landed, so the window that opened on it took the offer"
        );
    });
    view.read_with(cx, |v, _| {
        assert!(
            v.quote_destination_for_test().is_none(),
            "the picker closed"
        );
        assert!(
            !v.has_post_selection_for_test(),
            "the passage has left this post; a second press can't re-send it"
        );
        assert!(
            v.active_draft_references_for_test().is_empty(),
            "…and nothing was attached to a draft *here*"
        );
    });
}

#[gpui::test]
fn space_a_quote_into_an_open_conversation_lands_in_the_window_it_raises(cx: &mut TestAppContext) {
    // The destination is **already open**. The passage must land in that
    // window, and no second window may open onto the same conversation: the
    // one-shot mailbox is drained by whichever view draws first, so opening a
    // duplicate is a coin flip between showing the reader their quote and
    // showing them a fresh, empty composer with the quote in the window behind
    // it (Codex review, PR #280). The invariant: the window presented is the
    // window holding the quote.
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.spaces = vec![
            stub_space("s", Some("Here"), None, 2),
            stub_space("other", Some("Tides"), None, 1),
        ];
    });
    // Two windows on the destination, because that is where "whichever view
    // renders first" stops being a shape and starts being a coin flip: the
    // offer is addressed, so only the raised one — the newest — may take it.
    let (_older_window, older_view) = open_space(cx, &stores, Some("other".into()));
    let (dest_window, dest_view) = open_space(cx, &stores, Some("other".into()));
    // Both windows share the one entity, so seeding through either gives the
    // destination the loaded tail an offer waits for.
    seed_quotable_space(
        &dest_view,
        dest_window,
        cx,
        vec![fixture_post_with_block(
            "d1",
            "db1",
            "a conversation over here",
        )],
    );
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let windows_before = cx.update(|cx| cx.windows().len());
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_post_with_block("a1", "b1", "the quick brown fox")],
    );

    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| v.select_in_post_for_test("a1", 4..15, cx));
    })
    .unwrap();
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.quote_elsewhere(&eidola_gui::actions::QuoteElsewhere, window, cx)
        });
        view.update(cx, |v, cx| {
            v.arm_quote_destination_for_test("other", "Tides", window, cx)
        });
        view.update(cx, |v, cx| v.confirm_quote_destination_for_test(window, cx));
    })
    .unwrap();
    cx.run_until_parked();

    assert_eq!(
        cx.update(|cx| cx.windows().len()),
        windows_before,
        "the conversation was already open — raising it, not opening a duplicate"
    );
    dest_view.read_with(cx, |v, _| {
        assert_eq!(
            v.active_draft_references_for_test(),
            vec![(1u64, "quick brown".to_string())],
            "the window the reader is shown is the one holding the passage"
        );
    });
    older_view.read_with(cx, |v, _| {
        assert!(
            v.active_draft_references_for_test().is_empty(),
            "the window sharing the conversation drew straight past an offer              addressed to its sibling"
        );
    });
    view.read_with(cx, |v, _| {
        assert!(
            v.active_draft_references_for_test().is_empty(),
            "and nothing was attached here"
        );
    });
}

#[gpui::test]
fn space_a_quote_into_an_unloaded_conversation_waits_for_its_tail(cx: &mut TestAppContext) {
    // Quoting into a conversation that was **not** already open lands in a
    // window whose first frames run while the transcript is still loading.
    // `sync_tail_drafts` deliberately does nothing in that state, so a take on
    // that frame minted a draft against a tree with no posts in it: a *root*
    // draft, visually attached to nothing, that submitted with no `reply_to`
    // and was persisted under whatever the tail turned out to be — a guess
    // that looked like an answer (Codex review, PR #280). The mailbox already
    // survives frames, so the offer simply waits for the tail it belongs to.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("dest".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());
    assert!(
        space.read_with(cx, |s, _| !matches!(
            s.transcript(),
            eidola_gui::loadable::Loadable::Loaded { .. }
        )),
        "the premise: this window opened on a conversation it has not read yet"
    );

    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.offer_quote(
                eidola_gui::space::OfferedQuote {
                    spec: eidola_app_core::ReferenceSpec {
                        antecedent_action_id: "a1".into(),
                        content_block_id: Some("b1".into()),
                        range_start: Some(4),
                        range_end: Some(15),
                        annotation: None,
                    },
                    byline: "You".into(),
                    snippet: "quick brown".into(),
                },
                None,
                cx,
            );
        });
    })
    .unwrap();
    draw_window(cx, window);

    assert!(
        space.read_with(cx, |s, _| s.has_offered_quote()),
        "the offer waits in the mailbox rather than minting a draft against a tree with no tail"
    );
    view.read_with(cx, |v, _| {
        assert!(
            v.active_draft_references_for_test().is_empty(),
            "nothing is attached yet"
        );
    });

    // The transcript lands, `sync_tail_drafts` mints the real tail composer,
    // and the offer is taken on that same frame.
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_post_with_block(
            "d1",
            "db1",
            "a conversation over here",
        )],
    );
    view.read_with(cx, |v, _| {
        assert_eq!(
            v.active_draft_references_for_test(),
            vec![(1u64, "quick brown".to_string())],
            "the passage lands once there is somewhere for it to land"
        );
        assert_eq!(
            v.active_draft_parent_for_test().as_deref(),
            Some("d1"),
            "and in the branch's real tail composer — what a submit will carry \
             as its reply antecedent, rather than a root draft's nothing"
        );
    });
    assert!(
        space.read_with(cx, |s, _| !s.has_offered_quote()),
        "taken exactly once"
    );
}

#[gpui::test]
fn space_quote_destination_frame_work_is_constant_in_library_size(cx: &mut TestAppContext) {
    // Virtualizing the *elements* and then materializing the whole display
    // model to slice it is half the move: the indexer cloned an id and a label
    // for every conversation the reader has ever had, and the render built the
    // same vector again to ask its length (Codex review, PR #280). The range
    // now goes on before anything is cloned, and the count is a count — so a
    // frame's allocations are O(visible) and only a pointer scan is O(loaded).
    let small: Vec<_> = (0..40)
        .map(|i| {
            stub_space(
                &format!("s{i}"),
                Some(&format!("Conversation {i}")),
                None,
                i as i64,
            )
        })
        .collect();
    let large: Vec<_> = (0..2000)
        .map(|i| {
            stub_space(
                &format!("s{i}"),
                Some(&format!("Conversation {i}")),
                None,
                i as i64,
            )
        })
        .collect();
    let visible = 0..10usize;

    let run = |rows: Vec<eidola_app_core::SpaceInfo>, cx: &mut TestAppContext| {
        let stores = stub_stores(cx, |s| {
            s.config_state = Some(config_state(true));
            s.spaces = rows;
        });
        let (window, view) = open_space(cx, &stores, Some("s0".into()));
        cx.update_window(window, |_, window, cx| {
            view.update(cx, |v, cx| {
                v.quote_destination_frame_work_for_test(visible.clone(), window, cx)
            })
        })
        .unwrap()
    };

    let small_built = run(small, cx);
    let large_built = run(large, cx);

    assert_eq!(small_built, 10, "the visible window is what gets built");
    assert_eq!(
        large_built, 10,
        "…and it does not grow with the Library — 2000 conversations, ten rows"
    );
    // **What this test does *not* claim.** The remaining half of the finding —
    // that the ids and labels are cloned only for the visible range — is not
    // gated here, deliberately. Both shapes walk the index (the total that
    // `aria_size_of_set` needs is a count over it), so only the allocations
    // differ, and wall-clock separates them by ~3.5× at any fixture size: a
    // ratio gate at that margin passed a materialize-then-slice regression on
    // a second run, which is worse than no gate at all. The locality is held
    // by construction instead — the range goes on the iterator *before* the
    // `map` that clones, and the count returns `usize` so there is no vector
    // to materialize — and a gate for it would want allocation counting the
    // crate does not have. What is pinned here is the half that paints.
}

#[gpui::test]
fn space_two_quotes_offered_before_a_draw_both_land(cx: &mut TestAppContext) {
    // Two source windows quote into the same conversation before it redraws —
    // it may be minimised, or simply behind. Each confirm is a deliberate act
    // over a passage the reader chose, and quotes **compose**: the draft takes
    // as many pending references as it is given. A mailbox that held one
    // dropped the first silently (Codex review, PR #280) — the second offer
    // replaced it, and nothing said so.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("dest".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_post_with_block(
            "d1",
            "db1",
            "a conversation over here",
        )],
    );
    let space = view.read_with(cx, |v, _| v.space().clone());
    let here = cx
        .update_window(window, |_, window, _| window.window_handle().window_id())
        .unwrap();

    let offer = |snippet: &str| eidola_gui::space::OfferedQuote {
        spec: eidola_app_core::ReferenceSpec {
            antecedent_action_id: "a1".into(),
            content_block_id: Some("b1".into()),
            range_start: Some(4),
            range_end: Some(15),
            annotation: None,
        },
        byline: "You".into(),
        snippet: snippet.into(),
    };

    // Both confirms land before this window draws a single frame.
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.offer_quote(offer("quick brown"), Some(here), cx);
            s.offer_quote(offer("lazy dog"), Some(here), cx);
        });
    })
    .unwrap();
    draw_window(cx, window);

    view.read_with(cx, |v, _| {
        assert_eq!(
            v.active_draft_references_for_test(),
            vec![
                (1u64, "quick brown".to_string()),
                (2u64, "lazy dog".to_string())
            ],
            "both passages are pending references, in the order they were confirmed"
        );
    });
    assert!(
        space.read_with(cx, |s, _| !s.has_offered_quote()),
        "and the queue is empty — nothing stranded behind the one that landed"
    );
    // (That the drain takes the *batch* rather than one entry per frame is the
    // implementation's own choice — no flicker of ordinals arriving one at a
    // time — but it is not separately gated here: this harness redraws until
    // quiescent, so a one-per-frame drain converges to the same assertions.)
}

#[gpui::test]
fn space_a_quote_addressed_to_a_closed_window_still_lands(cx: &mut TestAppContext) {
    // An offer whose addressed window is gone has no claimant: the sender's
    // raise-failure path re-addresses what it knows about, but a window that
    // dies after a successful raise and before it draws leaves the passage
    // behind. It is a confirmed act over a passage the reader chose, so it goes
    // to a live window on the same conversation rather than waiting in an
    // entity nobody drains. *Which* live window is a race, deliberately: the
    // addressee is gone, and any window the reader can see beats none.
    //
    // (That an offer addressed to a **live** sibling is not this window's to
    // take is pinned by `space_a_quote_into_an_open_conversation_lands_in_the_
    // window_it_raises`, where the older window draws straight past one.)
    let stores = stub_stores_with_config(cx);
    let (gone_window, _gone_view) = open_space(cx, &stores, Some("dest".into()));
    let gone_id = cx
        .update_window(gone_window, |_, window, _| {
            window.window_handle().window_id()
        })
        .unwrap();
    cx.update_window(gone_window, |_, window, _| window.remove_window())
        .unwrap();
    cx.run_until_parked();

    let (window, view) = open_space(cx, &stores, Some("dest".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_post_with_block(
            "d1",
            "db1",
            "a conversation over here",
        )],
    );
    let space = view.read_with(cx, |v, _| v.space().clone());

    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.offer_quote(
                eidola_gui::space::OfferedQuote {
                    spec: eidola_app_core::ReferenceSpec {
                        antecedent_action_id: "a1".into(),
                        content_block_id: Some("b1".into()),
                        range_start: Some(4),
                        range_end: Some(15),
                        annotation: None,
                    },
                    byline: "You".into(),
                    snippet: "quick brown".into(),
                },
                // Addressed to a window that no longer exists.
                Some(gone_id),
                cx,
            );
        });
    })
    .unwrap();
    draw_window(cx, window);

    view.read_with(cx, |v, _| {
        assert_eq!(
            v.active_draft_references_for_test(),
            vec![(1u64, "quick brown".to_string())],
            "orphaned, so the live window on this conversation takes it"
        );
    });
    assert!(
        space.read_with(cx, |s, _| !s.has_offered_quote()),
        "and nothing is left stranded"
    );
}

#[gpui::test]
fn space_a_quote_lands_in_a_conversation_whose_refresh_failed(cx: &mut TestAppContext) {
    // The other side of the wait: `Failed { prior: Some(..) }` is a *refresh*
    // that failed over posts we still hold, and those posts are what the reader
    // is looking at — `messages()` reads through `Loadable::value` — with the
    // tail composer already hanging off them. A gate that accepted only
    // `Loaded` left a quote sent to such a window waiting in the mailbox
    // indefinitely, in plain sight of somewhere to land (Codex review, PR
    // #280). A retained value is an answer.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("dest".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_post_with_block(
            "d1",
            "db1",
            "a conversation over here",
        )],
    );
    let space = view.read_with(cx, |v, _| v.space().clone());

    // The next reload fails; the conversation stays on screen.
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| s.fail_transcript_refresh_for_test(cx));
    })
    .unwrap();
    space.read_with(cx, |s, _| {
        assert!(
            s.transcript().error().is_some(),
            "the refresh failed, as the premise requires"
        );
        assert_eq!(
            s.messages().len(),
            1,
            "and the posts it failed over are still the ones on screen"
        );
    });

    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.offer_quote(
                eidola_gui::space::OfferedQuote {
                    spec: eidola_app_core::ReferenceSpec {
                        antecedent_action_id: "a1".into(),
                        content_block_id: Some("b1".into()),
                        range_start: Some(4),
                        range_end: Some(15),
                        annotation: None,
                    },
                    byline: "You".into(),
                    snippet: "quick brown".into(),
                },
                None,
                cx,
            );
        });
    })
    .unwrap();
    draw_window(cx, window);

    view.read_with(cx, |v, _| {
        assert_eq!(
            v.active_draft_references_for_test(),
            vec![(1u64, "quick brown".to_string())],
            "the passage lands in the conversation the reader can see"
        );
        assert_eq!(
            v.active_draft_parent_for_test().as_deref(),
            Some("d1"),
            "in its real tail composer, not a root draft"
        );
    });
    assert!(
        space.read_with(cx, |s, _| !s.has_offered_quote()),
        "and the mailbox is empty rather than holding it forever"
    );
}

#[gpui::test]
fn space_a_quote_from_another_window_lands_in_this_ones_draft(cx: &mut TestAppContext) {
    // The receiving half. A draft is window-local, so the shared `Space`
    // entity is the courier: the window the offer names takes it and attaches
    // it exactly as `Edit > Quote` would — ordinal 1, a footnote row, a marker
    // in the body, and nothing durable until it posts.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("dest".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_post_with_block(
            "d1",
            "db1",
            "a conversation over here",
        )],
    );
    let space = view.read_with(cx, |v, _| v.space().clone());

    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.offer_quote(
                eidola_gui::space::OfferedQuote {
                    spec: eidola_app_core::ReferenceSpec {
                        antecedent_action_id: "a1".into(),
                        content_block_id: Some("b1".into()),
                        range_start: Some(4),
                        range_end: Some(15),
                        annotation: None,
                    },
                    byline: "You".into(),
                    snippet: "quick brown".into(),
                },
                // Unaddressed: the sender found no window on this space and
                // opened one, so the next view to draw it takes the offer.
                None,
                cx,
            );
        });
    })
    .unwrap();
    cx.update_window(window, |_, window, _| window.refresh())
        .unwrap();
    cx.run_until_parked();

    view.read_with(cx, |v, _| {
        assert_eq!(
            v.active_draft_references_for_test(),
            vec![(1u64, "quick brown".to_string())],
            "the offered passage is an ordinary pending reference now"
        );
    });
    space.read_with(cx, |s, _| {
        assert!(
            !s.has_offered_quote(),
            "the mailbox is a one-shot: a second window must not paste it again"
        );
    });
}

#[gpui::test]
fn space_dismissing_a_bottom_band_hands_the_keyboard_back(cx: &mut TestAppContext) {
    // Every bottom band's Dismiss is a real tab stop (`probe(Role::Button)`
    // derives `focusable()` + `tab_index(0)`), and pressing it unmounts the
    // band around it — the dead-handle class, on the denied-follow notice this
    // PR added and on the two bands beside it (Codex review, PR #280). Each
    // asks containment before it clears, so a reader composing beside a notice
    // keeps their caret.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_post_with_block("a1", "b1", "the quick brown fox")],
    );
    let root = view.read_with(cx, |v, _| v.focus_handle());

    // The denied-follow notice.
    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| {
            v.report_navigation_failure_for_test(
                eidola_app_core::error::AppError::NotAParticipant {
                    participant_id: "p1".into(),
                    action_id: "a-private".into(),
                },
                cx,
            )
        });
    })
    .unwrap();
    draw_window(cx, window);
    let band = view
        .read_with(cx, |v, _| v.band_focus_for_test(1))
        .expect("the notice paints");
    cx.update_window(window, |_, window, cx| window.focus(&band, cx))
        .unwrap();
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.dismiss_reference_notice(window, cx));
    })
    .unwrap();
    assert!(
        cx.update_window(window, |_, window, _| root.is_focused(window))
            .unwrap(),
        "the notice handed the keyboard back rather than leaving it on a band nobody paints"
    );

    // The cascade notice — same family, same rule (pre-existing gap).
    let space = view.read_with(cx, |v, _| v.space().clone());
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.emit_cascade_paused_for_test(2, 2, "a1".into(), cx)
        });
    })
    .unwrap();
    draw_window(cx, window);
    let band = view
        .read_with(cx, |v, _| v.band_focus_for_test(2))
        .expect("the cascade notice paints");
    cx.update_window(window, |_, window, cx| window.focus(&band, cx))
        .unwrap();
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.dismiss_cascade(window, cx));
    })
    .unwrap();
    assert!(
        cx.update_window(window, |_, window, _| root.is_focused(window))
            .unwrap(),
        "and so does the cascade band"
    );

    // …and a band that was not holding the keyboard takes nothing.
    open_space_draft(&view, window, cx, Some("a1"));
    let composer = view
        .read_with(cx, |v, _| v.composer_state_for_test())
        .expect("a draft is open");
    let caret = composer.read_with(cx, |e, cx| e.focus_handle(cx));
    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| {
            v.report_navigation_failure_for_test(
                eidola_app_core::error::AppError::NotAParticipant {
                    participant_id: "p1".into(),
                    action_id: "a-private".into(),
                },
                cx,
            )
        });
    })
    .unwrap();
    draw_window(cx, window);
    cx.update_window(window, |_, window, cx| window.focus(&caret, cx))
        .unwrap();
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.dismiss_reference_notice(window, cx));
    })
    .unwrap();
    assert!(
        cx.update_window(window, |_, window, _| caret.is_focused(window))
            .unwrap(),
        "a reader composing beside a notice keeps their caret"
    );
}

#[gpui::test]
fn space_a_denied_follow_says_so_without_naming_the_conversation(cx: &mut TestAppContext) {
    // Rule 4's human arm. The refusal is app-core's (the resolve behind a
    // footnote click is membership-gated, and its tests pin both the gate and
    // the non-leaking payload); what this pins is the **voice**: a quiet
    // notice, not a failure — nothing broke and there is nothing to retry —
    // and one that names nothing about the conversation it refused.
    let stores = stub_stores_with_config(cx);
    let (window, view) = open_space(cx, &stores, Some("s".into()));

    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| {
            v.report_navigation_failure_for_test(
                eidola_app_core::error::AppError::NotAParticipant {
                    participant_id: "p-cartographer".into(),
                    action_id: "a-private-post".into(),
                },
                cx,
            )
        });
    })
    .unwrap();

    let notice = view
        .read_with(cx, |v, _| v.reference_notice_for_test())
        .expect("the reader is told the follow went nowhere");
    assert!(
        notice.contains("don't take part in"),
        "said in the app's voice: {notice}"
    );
    for leak in ["a-private-post", "p-cartographer"] {
        assert!(
            !notice.contains(leak),
            "the notice leaked {leak:?}: {notice}"
        );
    }
    view.read_with(cx, |v, cx| {
        assert!(
            v.error_for_test(cx).is_none(),
            "a refusal is not a failure — no danger band, no Retry"
        );
    });

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.dismiss_reference_notice(window, cx));
    })
    .unwrap();
    view.read_with(cx, |v, _| {
        assert!(v.reference_notice_for_test().is_none(), "and it dismisses");
    });
}

#[gpui::test]
fn inspector_inviting_an_agent_from_another_space_shares_it_and_adds_it_here(
    cx: &mut TestAppContext,
) {
    // Task 37's grant, from the surface a reader would use it from: the space
    // whose privacy the decision is about. The candidate here is a **space-
    // owned** agent from another conversation, so the grant is a share *and* a
    // membership — one core call, so the irreversible half can never land
    // alone — and the sentence says both, including the part that can't be
    // undone.
    let (stores, core, _dir, space) = participants_scene(cx);
    let elsewhere = core
        .runtime()
        .block_on(core.create_space(Some("Tides".into())))
        .expect("a second conversation")
        .id;
    let visitor = core
        .runtime()
        .block_on(core.list_space_participants(elsewhere.clone()))
        .expect("participants")
        .into_iter()
        .find(|p| p.kind == "agent")
        .expect("the template seeds an agent there too");

    let (window, view) = open_participants_inspector(cx, &stores, &space);
    wait_until(cx, "participants load", |cx| {
        participant_labels(&stores, &space, cx).len() == 2
    });

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_begin_invite(window, cx));
    })
    .unwrap();
    wait_until(cx, "the candidates land", |cx| {
        view.read_with(cx, |v, _| {
            v.inspector_invite_for_test()
                .is_some_and(|(labels, _)| !labels.is_empty())
        })
    });

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_arm_invite(&visitor.id, window, cx));
    })
    .unwrap();
    let statement = view
        .read_with(cx, |v, _| v.inspector_invite_for_test())
        .expect("the form is open")
        .1
        .expect("a candidate is armed");
    assert!(
        statement.contains("shares it across spaces") && statement.contains("can't be undone"),
        "an unshared agent's grant says what it costs: {statement}"
    );
    assert!(
        statement.contains("read this conversation"),
        "…and what it gives: {statement}"
    );

    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_confirm_invite(window, cx));
    })
    .unwrap();
    wait_until(cx, "the grant lands", |cx| {
        stores
            .participants
            .read_with(cx, |s, _| s.list(&space).iter().any(|p| p.id == visitor.id))
    });

    let member = stores.participants.read_with(cx, |s, _| {
        s.list(&space)
            .iter()
            .find(|p| p.id == visitor.id)
            .expect("granted")
            .clone()
    });
    assert_eq!(member.scope, "global", "it had to be shared to join at all");
    assert_eq!(member.role, "observer", "read-only is what was granted");
    assert_eq!(
        member.label, visitor.label,
        "and it arrives as itself — promotion moves ownership, not configuration"
    );
    assert!(
        view.read_with(cx, |v, _| v.inspector_invite_for_test().is_none()),
        "the form closes on the grant"
    );

    // It is no longer a candidate here (an affordance that could only be a
    // no-op is not offered), and its own space still has it.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_begin_invite(window, cx));
    })
    .unwrap();
    wait_until(cx, "the second read lands", |cx| {
        view.read_with(cx, |v, _| {
            v.inspector_invite_for_test().is_some_and(|(l, _)| {
                !l.iter().any(|label| label == &visitor.label) || l.is_empty()
            })
        })
    });
    drain_runtime(&core);
}

#[gpui::test]
fn inspector_the_invite_form_takes_the_keyboard_its_door_had(cx: &mut TestAppContext) {
    // The mount side of the handback rule. Opening the form **replaces** the
    // "Invite an agent…" door, and arming a candidate replaces the rows inside
    // the form — both are real tab stops (`probe(Role::Button)` derives
    // `focusable()` + `tab_index(0)`), so a transition that mounts a surface
    // and focuses nothing leaves the window on a dead handle: keystrokes reach
    // nothing and Tab restarts from the window root (Codex review, PR #280).
    let (stores, core, _dir, space) = participants_scene(cx);
    let elsewhere = core
        .runtime()
        .block_on(core.create_space(Some("Tides".into())))
        .expect("a second conversation")
        .id;
    let visitor = core
        .runtime()
        .block_on(core.list_space_participants(elsewhere.clone()))
        .expect("participants")
        .into_iter()
        .find(|p| p.kind == "agent")
        .expect("the template seeds an agent there too");
    let (window, view) = open_participants_inspector(cx, &stores, &space);
    wait_until(cx, "participants load", |cx| {
        participant_labels(&stores, &space, cx).len() == 2
    });

    // Opening it: a reveal focuses what it revealed — the rule the disclosure,
    // Add and the template form already follow. This form has no text field,
    // so the keyboard goes to the form itself.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_begin_invite(window, cx));
    })
    .unwrap();
    let form = view
        .read_with(cx, |v, _| v.inspector_invite_focus_handle())
        .expect("the form is open");
    assert!(
        cx.update_window(window, |_, window, _| form.is_focused(window))
            .unwrap(),
        "the form the door became holds the keyboard"
    );

    wait_until(cx, "the candidates land", |cx| {
        view.read_with(cx, |v, _| {
            v.inspector_invite_for_test()
                .is_some_and(|(labels, _)| !labels.is_empty())
        })
    });
    draw_window(cx, window);

    // Tab onto a candidate row, then arm it: the row unmounts with the list it
    // was in, and the form survives, so the keyboard stays on the form — the
    // rule the share confirmation already applies to its own verbs.
    cx.update_window(window, |_, window, cx| window.focus_next(cx))
        .unwrap();
    assert!(
        cx.update_window(window, |_, window, cx| {
            form.contains_focused(window, cx) && !form.is_focused(window)
        })
        .unwrap(),
        "Tab from the form lands on a control inside it"
    );
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_arm_invite(&visitor.id, window, cx));
    })
    .unwrap();
    draw_window(cx, window);
    assert!(
        cx.update_window(window, |_, window, cx| form.contains_focused(window, cx))
            .unwrap(),
        "arming the grant leaves the keyboard on the form it replaced a row in"
    );
    drain_runtime(&core);
}

#[gpui::test]
fn space_the_quote_picker_takes_the_keyboard_and_gives_it_back(cx: &mut TestAppContext) {
    // Both sides of the picker's own focus contract (Codex review, PR #280).
    // **Mount:** a reveal focuses what it revealed — and the context-menu door
    // makes it a defect rather than an inconvenience, since `run_context_item`
    // unmounts the focused menu row *before* dispatching, leaving a keyboard
    // reader on a dead handle while the surface they asked for stands
    // unfocused. **Unmount:** a surface that took the keyboard owes it back;
    // its rows and verbs are real tab stops, so dropping it while it holds the
    // focus is the same dead handle one step later.
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.spaces = vec![
            stub_space("s", Some("Here"), None, 2),
            stub_space("other", Some("Tides"), None, 1),
        ];
    });
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_post_with_block("a1", "b1", "the quick brown fox")],
    );
    let root = view.read_with(cx, |v, _| v.focus_handle());
    let open_the_picker = |cx: &mut TestAppContext| {
        cx.update_window(window, |_, window, cx| {
            view.update(cx, |v, cx| v.select_in_post_for_test("a1", 4..15, cx));
            view.update(cx, |v, cx| {
                v.quote_elsewhere(&eidola_gui::actions::QuoteElsewhere, window, cx)
            });
        })
        .unwrap();
        view.read_with(cx, |v, _| v.quote_destination_focus_handle())
            .expect("the picker is open")
    };

    // Opening it — from either door; both funnel through this action, which is
    // why one focus call covers the context menu's row and the Edit menu's item.
    // The keyboard lands on the **list**, the surface's single tab stop, so ↑/↓
    // work the moment it opens; the popover subtree contains it either way,
    // which is what the handback below asks about.
    let picker = open_the_picker(cx);
    let list = view
        .read_with(cx, |v, _| v.quote_destination_list_focus_handle())
        .expect("the picker is open");
    assert!(
        cx.update_window(window, |_, window, _| list.is_focused(window))
            .unwrap(),
        "the list holds the keyboard the picker revealed"
    );
    assert!(
        cx.update_window(window, |_, window, cx| picker.contains_focused(window, cx))
            .unwrap(),
        "…inside the picker, so a close still knows it is holding it"
    );

    // Escape: the picker goes, and the keyboard comes back to the view root —
    // live, so the conversation's own key model answers the next press.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| assert!(v.close_quote_destination(window, cx)));
    })
    .unwrap();
    assert!(
        cx.update_window(window, |_, window, _| root.is_focused(window))
            .unwrap(),
        "Escape hands the keyboard back rather than leaving it on a picker nobody paints"
    );

    // "Quote there" unmounts the verb that was pressed. The passage leaves for
    // another window; what is owed here is that this one's keyboard stays live.
    let _picker = open_the_picker(cx);
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.arm_quote_destination_for_test("other", "Tides", window, cx)
        });
        view.update(cx, |v, cx| v.confirm_quote_destination_for_test(window, cx));
    })
    .unwrap();
    assert!(
        cx.update_window(window, |_, window, _| root.is_focused(window))
            .unwrap(),
        "the quote left for another window; the keyboard stayed in this one"
    );
    cx.run_until_parked();

    // **A reader navigating the conversation gets their place back, not the
    // root.** `keyboard_home` gives the same answer `sync_tree_focus`'s falling
    // edge would, so the explicit handback and the observation a frame later
    // agree — and the level survives instead of being cleared as "the
    // conversation lost focus".
    {
        let mut vcx = VisualTestContext::from_window(window, cx);
        vcx.simulate_keystrokes("down");
        vcx.run_until_parked();
    }
    view.read_with(cx, |v, _| {
        assert_eq!(
            v.tree_focus_for_test(),
            Some(("a1".to_string(), None)),
            "the reader is on the post"
        );
    });
    let _picker = open_the_picker(cx);
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.close_quote_destination(window, cx));
    })
    .unwrap();
    assert!(
        !cx.update_window(window, |_, window, _| root.is_focused(window))
            .unwrap(),
        "not the view root — the reader had a place in the conversation"
    );
    view.read_with(cx, |v, _| {
        assert_eq!(
            v.tree_focus_for_test(),
            Some(("a1".to_string(), None)),
            "and their level stands"
        );
    });

    // And the borrow rule: a picker that never held the keyboard has none to
    // give back — a reader composing beside it keeps their caret.
    open_space_draft(&view, window, cx, Some("a1"));
    let composer = view
        .read_with(cx, |v, _| v.composer_state_for_test())
        .expect("a draft is open");
    let picker = open_the_picker(cx);
    let caret = composer.read_with(cx, |e, cx| e.focus_handle(cx));
    cx.update_window(window, |_, window, cx| {
        window.focus(&caret, cx);
    })
    .unwrap();
    assert!(
        !cx.update_window(window, |_, window, cx| picker.contains_focused(window, cx))
            .unwrap(),
        "the reader moved the keyboard out of the picker"
    );
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.close_quote_destination(window, cx));
    })
    .unwrap();
    assert!(
        cx.update_window(window, |_, window, _| caret.is_focused(window))
            .unwrap(),
        "closing a picker that was not holding the keyboard takes nothing"
    );
}

#[gpui::test]
fn space_the_quote_picker_roves_a_cursor_over_its_whole_index(cx: &mut TestAppContext) {
    // A virtualized list is **one** tab stop with a roving cursor. Per-row tab
    // stops describe a tab order that does not contain the rows nobody has
    // scrolled to: Tab walked off the end of the materialized slice and out of
    // the picker, and every conversation past the first dozen was unreachable
    // by keyboard (Codex review, PR #280). ↑/↓/Home/End move the cursor,
    // `scroll_to_item` materializes what it lands on, Enter arms it.
    let many: Vec<_> = (0..60)
        .map(|i| {
            stub_space(
                &format!("s{i}"),
                Some(&format!("Conversation {i}")),
                None,
                i as i64,
            )
        })
        .collect();
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.spaces = many;
    });
    let (window, view) = open_space(cx, &stores, Some("s0".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_post_with_block("a1", "b1", "the quick brown fox")],
    );
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.select_in_post_for_test("a1", 4..15, cx));
        view.update(cx, |v, cx| {
            v.quote_elsewhere(&eidola_gui::actions::QuoteElsewhere, window, cx)
        });
    })
    .unwrap();
    draw_window(cx, window);
    view.read_with(cx, |v, cx| {
        assert_eq!(
            v.quote_destination_cursor_for_test(cx),
            Some(0),
            "the cursor starts at the top"
        );
    });

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_keystrokes("down down");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, cx| {
        assert_eq!(
            v.quote_destination_cursor_for_test(cx),
            Some(2),
            "↓ moves it"
        );
    });
    vcx.simulate_keystrokes("up");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, cx| {
        assert_eq!(
            v.quote_destination_cursor_for_test(cx),
            Some(1),
            "↑ moves it back"
        );
    });

    // **End reaches the last destination — one the visible slice never held.**
    // This is the arc the single tab stop exists for: the cursor moves, the
    // list scrolls it into being, and it is readable and activatable there.
    vcx.simulate_keystrokes("end");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, cx| {
        assert_eq!(
            v.quote_destination_cursor_for_test(cx),
            Some(58),
            "End lands on the last of 59 destinations, far past the ten that painted"
        );
    });

    // **The same focus gate the invite list has.** The stored cursor is where
    // ↑/↓ left it; whether a row *claims* it is a question about where the
    // keyboard is — this surface reaches the ungated state by Tab (its retry
    // line is a stop beside the list) and by a reader clicking back into the
    // page with the picker still open.
    let popover = view
        .read_with(&vcx, |v, _| v.quote_destination_focus_handle())
        .expect("the picker is open");
    vcx.update(|window, cx| window.focus(&popover, cx));
    let (stored, shown) = vcx.update(|window, cx| {
        view.read_with(cx, |v, app| {
            (
                v.quote_destination_cursor_for_test(app),
                v.quote_destination_cursor_row_for_test(window, app),
            )
        })
    });
    assert_eq!(stored, Some(58), "the cursor stands");
    assert_eq!(
        shown, None,
        "…and no row claims it from a focus it doesn't have"
    );
    let list = view
        .read_with(&vcx, |v, _| v.quote_destination_list_focus_handle())
        .expect("the picker is open");
    vcx.update(|window, cx| window.focus(&list, cx));
    let shown = vcx.update(|window, cx| {
        view.read_with(cx, |v, app| {
            v.quote_destination_cursor_row_for_test(window, app)
        })
    });
    assert_eq!(shown, Some(58), "and returns with the keyboard");

    // Enter arms *that* one — the statement names it, so what the cursor
    // reached is what the reader is about to quote into.
    vcx.simulate_keystrokes("enter");
    vcx.run_until_parked();
    let statement = view
        .read_with(&vcx, |v, _| v.quote_destination_for_test())
        .expect("still open")
        .expect("a destination is armed");
    assert!(
        statement.contains("Conversation 59"),
        "Enter armed the destination the cursor was on: {statement}"
    );
}

#[gpui::test]
fn space_the_quote_pickers_cursor_leaves_escape_alone(cx: &mut TestAppContext) {
    // The roving idiom answers five keys and nothing else. Escape over this
    // surface means *dismiss* — a rung of the space root's own chain — so a
    // cursor that consumed it would shadow the picker's only way out.
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.spaces = vec![
            stub_space("s", Some("Here"), None, 2),
            stub_space("other", Some("Tides"), None, 1),
        ];
    });
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_post_with_block("a1", "b1", "the quick brown fox")],
    );
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.select_in_post_for_test("a1", 4..15, cx));
        view.update(cx, |v, cx| {
            v.quote_elsewhere(&eidola_gui::actions::QuoteElsewhere, window, cx)
        });
    })
    .unwrap();
    draw_window(cx, window);
    view.read_with(cx, |v, _| {
        assert!(
            v.quote_destination_for_test().is_some(),
            "the picker is open"
        );
    });

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_keystrokes("escape");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert!(
            v.quote_destination_for_test().is_none(),
            "Escape still closes it — the cursor took no rung of the chain"
        );
    });
}

#[gpui::test]
fn space_arming_a_quote_destination_keeps_the_keyboard_on_the_picker(cx: &mut TestAppContext) {
    // The same rule, one surface over: choosing a destination replaces the list
    // of conversations with the visibility statement and its two verbs, so the
    // row that was pressed unmounts. The picker survives it and keeps the
    // keyboard (Codex review, PR #280) — and only if the picker was the one
    // holding it, so a pointer press from the page takes nothing.
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.spaces = vec![
            stub_space("s", Some("Here"), None, 2),
            stub_space("other", Some("Tides"), None, 1),
        ];
    });
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_post_with_block("a1", "b1", "the quick brown fox")],
    );
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.select_in_post_for_test("a1", 4..15, cx));
        view.update(cx, |v, cx| {
            v.quote_elsewhere(&eidola_gui::actions::QuoteElsewhere, window, cx)
        });
    })
    .unwrap();
    draw_window(cx, window);
    let picker = view
        .read_with(cx, |v, _| v.quote_destination_focus_handle())
        .expect("the picker is open");

    // A reader who tabbed into the picker is on one of its rows.
    cx.update_window(window, |_, window, cx| {
        window.focus(&picker, cx);
        window.focus_next(cx);
    })
    .unwrap();
    assert!(
        cx.update_window(window, |_, window, cx| {
            picker.contains_focused(window, cx) && !picker.is_focused(window)
        })
        .unwrap(),
        "Tab from the picker lands on a destination row"
    );
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.arm_quote_destination_for_test("other", "Tides", window, cx)
        });
    })
    .unwrap();
    draw_window(cx, window);
    assert!(
        cx.update_window(window, |_, window, cx| picker.contains_focused(window, cx))
            .unwrap(),
        "the statement's verbs are where the keyboard is, not a row nobody paints"
    );
}

#[gpui::test]
fn inspector_a_notebook_offers_no_grant_door(cx: &mut TestAppContext) {
    // A notebook is an agent's private space, and whether an agent may be
    // *granted* observer membership of another agent's notebook is a decision
    // task 37 deliberately leaves open. Until it is made, the panel offers no
    // new door there — app-core's rules are untouched, so nothing is
    // foreclosed either way.
    let (stores, core, _dir, space) = participants_scene(cx);
    let agent = core
        .runtime()
        .block_on(core.list_space_participants(space.clone()))
        .expect("participants")
        .into_iter()
        .find(|p| p.kind == "agent")
        .expect("the template seeds an agent")
        .id;
    let notebook = core
        .runtime()
        .block_on(core.promote_participant(agent.clone(), None, None))
        .expect("promotion")
        .notebook_space_id;

    let (window, view) = open_participants_inspector(cx, &stores, &notebook);
    wait_until(cx, "the notebook's settings land", |cx| {
        stores.space_settings.read_with(cx, |s, _| {
            s.settings(&notebook)
                .value()
                .is_some_and(|s| s.notebook_participant_id.is_some())
        })
    });
    assert!(
        !view.read_with(cx, |v, cx| v.inspector_offers_grant_door_for_test(cx)),
        "a notebook withholds the grant door"
    );

    // An ordinary conversation still offers it.
    let (_w2, view2) = open_participants_inspector(cx, &stores, &space);
    wait_until(cx, "the space's settings land", |cx| {
        stores
            .space_settings
            .read_with(cx, |s, _| s.settings(&space).value().is_some())
    });
    assert!(
        view2.read_with(cx, |v, cx| v.inspector_offers_grant_door_for_test(cx)),
        "an ordinary conversation offers the grant"
    );
    let _ = window;
    drain_runtime(&core);
}

#[gpui::test]
fn inspector_the_invite_list_roves_a_cursor_over_all_its_candidates(cx: &mut TestAppContext) {
    // The candidate list's half of the same rule: one tab stop, a roving
    // cursor, `scroll_to_item` materializing what it lands on — because the
    // candidates are every agent this reader could add, and per-row tab stops
    // reach only the dozen that painted (Codex review, PR #280).
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.participants = Some(("s".into(), vec![agent_participant("a-1", "Mara")]));
        s.space_settings = Some(("s".into(), eidola_app_core::SpaceSettings::default()));
    });
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.set_inspector_open_for_test(true, window, cx));
        view.update(cx, |v, cx| v.inspector_begin_invite(window, cx));
        view.update(cx, |v, cx| {
            v.seed_invite_candidates_for_test(
                (0..40)
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
    draw_window(cx, window);

    // The list is the tab stop; put the keyboard there as a Tab would.
    let list = view
        .read_with(cx, |v, _| v.invite_list_focus_handle())
        .expect("the form is open");
    cx.update_window(window, |_, window, cx| window.focus(&list, cx))
        .unwrap();
    view.read_with(cx, |v, _| {
        assert_eq!(v.invite_cursor_for_test(), Some(0), "starting at the top");
    });

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.simulate_keystrokes("down down");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(v.invite_cursor_for_test(), Some(2));
    });
    vcx.simulate_keystrokes("end");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.invite_cursor_for_test(),
            Some(39),
            "End reaches the last candidate — far past the visible slice"
        );
    });

    // Enter arms it: the grant statement names the agent the cursor reached.
    vcx.simulate_keystrokes("enter");
    vcx.run_until_parked();
    let (_, statement) = view
        .read_with(&vcx, |v, _| v.inspector_invite_for_test())
        .expect("the form is open");
    let statement = statement.expect("a candidate is armed");
    assert!(
        statement.contains("Agent 39"),
        "Enter armed what the cursor was on: {statement}"
    );

    // **The cursor is the row's focus identity, so it belongs to the row only
    // while the list holds the keyboard.** Gated on modality alone it painted a
    // ring on the first candidate the moment the read landed — the door's own
    // reveal focuses the *form* — and left it there after Tab reached Cancel:
    // two focus indications for one focus (Codex review, PR #280).
    let form = view
        .read_with(&vcx, |v, _| v.inspector_invite_focus_handle())
        .expect("the form is open");
    vcx.update(|window, cx| window.focus(&form, cx));
    let (stored, shown) = vcx.update(|window, cx| {
        view.read_with(cx, |v, _| {
            (
                v.invite_cursor_for_test(),
                v.invite_cursor_row_for_test(window),
            )
        })
    });
    assert_eq!(stored, Some(39), "the cursor is still where ↑/↓ left it");
    assert_eq!(
        shown, None,
        "…but no row claims it while the keyboard is elsewhere in the form"
    );
    vcx.update(|window, cx| window.focus(&list, cx));
    let shown =
        vcx.update(|window, cx| view.read_with(cx, |v, _| v.invite_cursor_row_for_test(window)));
    assert_eq!(shown, Some(39), "and it comes back with the keyboard");

    // And Escape is none of its business — the form's exit is its Cancel, and
    // the panel's Escape rungs belong to its dropdowns.
    vcx.simulate_keystrokes("escape");
    vcx.run_until_parked();
    view.read_with(&vcx, |v, _| {
        assert!(
            v.inspector_invite_for_test().is_some(),
            "the cursor consumed no Escape, and none of the rungs closes this form"
        );
    });
}

#[gpui::test]
fn inspector_withholds_the_grant_door_until_the_settings_say_what_this_space_is(
    cx: &mut TestAppContext,
) {
    // The withheld door is only as good as the question behind it. That
    // question is a **settings read**, and a cell that has not answered is not
    // an answer of "ordinary" — folding the two offered the grant in the window
    // before the read landed, and confirming it succeeds, because app-core
    // deliberately does not refuse a notebook grant (task 37 left that decision
    // open, which is exactly why the GUI's withholding is the only thing
    // standing between the reader and a notebook's whole history). Codex
    // review, PR #280.
    let (stores, core, _dir, space) = participants_scene(cx);
    let agent = core
        .runtime()
        .block_on(core.list_space_participants(space.clone()))
        .expect("participants")
        .into_iter()
        .find(|p| p.kind == "agent")
        .expect("the template seeds an agent")
        .id;
    let notebook = core
        .runtime()
        .block_on(core.promote_participant(agent.clone(), None, None))
        .expect("promotion")
        .notebook_space_id;

    let (window, view) = open_participants_inspector(cx, &stores, &notebook);
    assert!(
        stores
            .space_settings
            .read_with(cx, |s, _| s.settings(&notebook).value().is_none()),
        "the premise of this test: the panel has asked, and nothing has answered yet"
    );
    assert!(
        !view.read_with(cx, |v, cx| v.inspector_offers_grant_door_for_test(cx)),
        "unknown withholds the door — it comes back on the frame the read answers"
    );

    // And the form is a door left standing: one opened while the answer was
    // still unknown must not be confirmable once it arrives.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.inspector_begin_invite(window, cx));
    })
    .unwrap();
    assert!(
        view.read_with(cx, |v, _| v.inspector_invite_for_test().is_some()),
        "the form is open over a space nobody has classified yet"
    );

    wait_until(cx, "the notebook's settings land", |cx| {
        stores.space_settings.read_with(cx, |s, _| {
            s.settings(&notebook)
                .value()
                .is_some_and(|s| s.notebook_participant_id.is_some())
        })
    });
    draw_window(cx, window);
    assert!(
        view.read_with(cx, |v, _| v.inspector_invite_for_test().is_none()),
        "the answer retires the form, rather than leaving a grant one press away"
    );
    assert!(
        !view.read_with(cx, |v, cx| v.inspector_offers_grant_door_for_test(cx)),
        "…and the door stays withheld, now on evidence"
    );
    drain_runtime(&core);
}

// ---------------------------------------------------------------------------
// Disposing of untouched spaces (the last-window-close trigger)
// ---------------------------------------------------------------------------

/// Whether the core still holds `space`, archived or not.
fn space_exists(core: &std::sync::Arc<AppCore>, space: &str) -> bool {
    core.runtime()
        .block_on(core.list_spaces(true))
        .is_ok_and(|spaces| spaces.iter().any(|s| s.id == space))
}

/// Close a window the way a reader does — and drop the test's own handle on
/// its view first, because the close trigger rides the view's **release** and
/// a test holding an `Entity<SpaceView>` keeps it alive past its window.
fn close_space_window(cx: &mut TestAppContext, window: AnyWindowHandle, view: Entity<SpaceView>) {
    drop(view);
    cx.update_window(window, |_, window, _| window.remove_window())
        .unwrap();
    cx.run_until_parked();
}

/// Give a disposal that is *not* supposed to happen a real chance to happen.
fn settle(cx: &mut TestAppContext) {
    for _ in 0..8 {
        cx.run_until_parked();
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    cx.run_until_parked();
}

/// ⌘N, then close without doing anything: the space goes with the window.
///
/// A space is created when its window opens, so without this every abandoned
/// new window leaves a durable empty conversation in the Library. The view's
/// release is the close; the entity answers "was that the last window"; the
/// core decides — inside its own transaction — whether there is anything here
/// worth keeping.
#[gpui::test]
fn a_new_space_nobody_touched_goes_with_its_window(cx: &mut TestAppContext) {
    let (stores, core, _dir, _seeded) = participants_scene(cx);

    let (window, view) = open_space(cx, &stores, None);
    let space = view.read_with(cx, |v, cx| v.space().read(cx).id().to_string());
    wait_until(cx, "the space's row commits", |_| {
        space_exists(&core, &space)
    });

    close_space_window(cx, window, view);
    wait_until(cx, "the untouched space is disposed of", |_| {
        !space_exists(&core, &space)
    });
    drain_runtime(&core);
}

/// The mirror, at the same seam: a conversation someone actually wrote in stays
/// when its window closes. A wrongly-kept blank costs a Library row; a wrongly
/// reaped space costs the words in it.
#[gpui::test]
fn a_conversation_someone_wrote_in_survives_its_window(cx: &mut TestAppContext) {
    let (stores, core, _dir, _seeded) = participants_scene(cx);

    let (window, view) = open_space(cx, &stores, None);
    let entity = view.read_with(cx, |v, _| v.space().clone());
    let space = entity.read_with(cx, |s, _| s.id().to_string());
    assert!(entity.update(cx, |s, cx| {
        s.submit("The tide is the moon's doing.".into(), None, Vec::new(), cx)
    }));
    wait_until(cx, "the post lands durably", |_| {
        core.runtime()
            .block_on(core.get_space_messages(space.clone()))
            .is_ok_and(|m| m.len() == 1)
    });

    close_space_window(cx, window, view);
    settle(cx);
    assert!(
        space_exists(&core, &space),
        "the conversation outlives the window that wrote it"
    );
    drain_runtime(&core);
}

/// Two windows on one space: closing the first disposes of nothing, because the
/// conversation is still open. Closing the second is the last close, and the
/// untouched space goes then.
#[gpui::test]
fn closing_one_of_two_windows_disposes_of_nothing(cx: &mut TestAppContext) {
    let (stores, core, _dir, _seeded) = participants_scene(cx);

    let (first, view) = open_space(cx, &stores, None);
    let space = view.read_with(cx, |v, cx| v.space().read(cx).id().to_string());
    wait_until(cx, "the space's row commits", |_| {
        space_exists(&core, &space)
    });
    let (second, other) = open_space(cx, &stores, Some(space.clone()));

    close_space_window(cx, first, view);
    settle(cx);
    assert!(
        space_exists(&core, &space),
        "a conversation still on screen is not abandoned"
    );

    close_space_window(cx, second, other);
    wait_until(cx, "the last close disposes of it", |_| {
        !space_exists(&core, &space)
    });
    drain_runtime(&core);
}

/// **A save still on its way to the database holds off the disposal.**
///
/// A `bridge`d core call outlives the gpui task that issued it, so a Send
/// pressed a moment before ⌘W leaves a post travelling to the same database the
/// disposal is about to reserve the writer on. Whichever reaches it first wins:
/// if the disposal does, the space is still pristine and goes, and the post
/// lands on nothing — with the window gone, nobody is even told. Lost prose is
/// the one outcome the whole feature exists to prevent, so the close asks —
/// while the entity is still alive to answer — whether anything is still
/// writing, and refuses to dispose when it is.
///
/// The in-flight save is staged with the entity's own busy seam rather than a
/// real one, because a real save's core call settles in a millisecond and
/// "still travelling" would then be a timing hope rather than a fact. What the
/// test drives is the production path — the view's release, the cross-store
/// question, the store's refusal.
#[gpui::test]
fn a_save_still_in_flight_holds_off_the_disposal(cx: &mut TestAppContext) {
    let (stores, core, _dir, _seeded) = participants_scene(cx);

    let (window, view) = open_space(cx, &stores, None);
    let entity = view.read_with(cx, |v, _| v.space().clone());
    let space = entity.read_with(cx, |s, _| s.id().to_string());
    wait_until(cx, "the space's row commits", |_| {
        space_exists(&core, &space)
    });

    // A save whose core call has not come back yet.
    entity.update(cx, |s, cx| s.arm_post_runner_for_test(cx));
    assert!(
        entity.read_with(cx, |s, _| s.is_busy()),
        "the premise: a mutation is outstanding"
    );
    drop(entity);

    close_space_window(cx, window, view);
    settle(cx);
    assert!(
        space_exists(&core, &space),
        "the space a write is still travelling to is kept, whatever else it \
         does or does not contain"
    );
    drain_runtime(&core);
}

/// The mirror: with nothing outstanding, the same close disposes of the same
/// untouched space — so the guard above is a guard and not a disabling.
#[gpui::test]
fn a_settled_space_is_still_disposed_of_at_the_same_close(cx: &mut TestAppContext) {
    let (stores, core, _dir, _seeded) = participants_scene(cx);

    let (window, view) = open_space(cx, &stores, None);
    let space = view.read_with(cx, |v, cx| v.space().read(cx).id().to_string());
    wait_until(cx, "the space's row commits", |_| {
        space_exists(&core, &space)
    });

    close_space_window(cx, window, view);
    wait_until(cx, "the untouched space is disposed of", |_| {
        !space_exists(&core, &space)
    });
    drain_runtime(&core);
}

/// **A refusal a reader sees is copy, copy is localized, and a view renders it
/// from state rather than from a string it kept.**
///
/// `error_copy` answers the conversation's recovery notice, and its fallback is
/// `AppError`'s own `Display` — written for a log, and English in every locale.
/// App-core ships no user-facing strings by doctrine, so that fallback is a
/// known residual the localization extraction sweep owns; `SpaceArchived` is
/// the first variant taken out of it.
///
/// The locale is switched here **without re-emitting the failure**, which is
/// the whole point: the band holds the typed error and formats at render, so a
/// language change repaints the sentence that is already on screen. Holding the
/// formatted string instead would pass a test that re-failed after each switch
/// and leave a real reader looking at the old language until something else
/// went wrong.
#[gpui::test]
async fn an_archived_conversations_refusal_is_localized(cx: &mut gpui::TestAppContext) {
    let stores = stub_stores_with_agents(cx, "s");
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());

    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("a1", "the only post")], cx)
        });
    })
    .unwrap();
    cx.run_until_parked();

    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.apply_turn_failure_for_test(
                "agent-b",
                "a1",
                AppError::SpaceArchived {
                    space_id: "s".into(),
                },
                cx,
            )
        });
    })
    .unwrap();
    cx.run_until_parked();

    let english = view.read_with(cx, |v, cx| v.error_for_test(cx).expect("a notice is shown"));
    assert_eq!(
        english, "This conversation is archived, so it can’t take new replies.",
        "the source locale's own words, not the error's Display"
    );
    assert!(
        !english.contains("takes no new turns"),
        "the raw Display must not be what a reader sees: {english}"
    );

    // **One failure, three languages.** Nothing is re-emitted between these —
    // only the locale changes, and the band must follow it.
    for (tag, expected) in [
        (
            "fr",
            "Cette conversation est archivée : elle ne peut plus recevoir de réponses.",
        ),
        ("zh-Hans", "此对话已归档，无法接收新的回复。"),
        (
            "en",
            "This conversation is archived, so it can’t take new replies.",
        ),
    ] {
        cx.update(|cx| eidola_gui::i18n::apply(tag, cx));
        cx.run_until_parked();
        assert_eq!(
            view.read_with(cx, |v, cx| v.error_for_test(cx).expect("a notice is shown")),
            expected,
            "{tag} repaints the notice already on screen — no new failure was emitted"
        );
    }
}

/// **An affordance that cannot succeed is worse than none.**
///
/// Archival is what closes a conversation and there is no unarchive door
/// anywhere in the app, so a Retry on an archived-space failure could only
/// re-hit the same guard. The failure therefore records no `failed_turn`, which
/// is what arms Retry — while the notice still explains itself, because what
/// has nothing to retry still has something to say.
#[gpui::test]
async fn an_archived_conversation_explains_itself_without_offering_retry(
    cx: &mut gpui::TestAppContext,
) {
    let stores = stub_stores_with_agents(cx, "s");
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    let space = view.read_with(cx, |v, _| v.space().clone());

    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.set_post_tree_for_test(vec![fixture_user_post("a1", "the only post")], cx)
        });
    })
    .unwrap();
    cx.run_until_parked();

    // An ordinary failure is retryable — the control case, so this test fails
    // if the record simply stopped being written at all.
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.apply_turn_failure_for_test(
                "agent-b",
                "a1",
                AppError::Network {
                    message: "dns error".into(),
                },
                cx,
            )
        });
    })
    .unwrap();
    cx.run_until_parked();
    view.read_with(cx, |v, cx| {
        assert!(v.error_for_test(cx).is_some(), "the notice explains it");
        assert!(v.space().read(cx).can_retry(), "and offers another press");
    });

    // The permanent one does not — **and it arrives on top of that standing
    // record**, deliberately. Not clearing it is the subtler half of the same
    // rule: the band would explain the archived refusal while Retry re-asked on
    // the network failure's behalf, into a room that cannot reopen.
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| {
            s.apply_turn_failure_for_test(
                "agent-b",
                "a1",
                AppError::SpaceArchived {
                    space_id: "s".into(),
                },
                cx,
            )
        });
    })
    .unwrap();
    cx.run_until_parked();

    view.read_with(cx, |v, cx| {
        let msg = v
            .error_for_test(cx)
            .expect("the band still explains what happened");
        assert!(msg.contains("archived"), "and says what: {msg}");
        assert!(
            !v.space().read(cx).can_retry(),
            "but arms no Retry — it could only re-hit the guard that closed the room"
        );
        assert!(
            v.space().read(cx).failed_turn().is_none(),
            "and leaves nothing armed behind the notice either — including the record the \
             earlier retryable failure had left standing"
        );
    });

    // A sibling turn finishing must not take the explanation with it. The
    // notice's lifetime is normally owned by the `failed_turn` record, and a
    // permanent refusal records none — so this is exactly where the
    // explanation could have been blanked by somebody else's success.
    cx.update_window(window, |_, _, cx| {
        space.update(cx, |s, cx| s.finish_save_for_test(cx));
    })
    .unwrap();
    cx.run_until_parked();
    view.read_with(cx, |v, cx| {
        assert!(
            v.error_for_test(cx).is_some(),
            "a sibling ending does not erase a refusal that stands"
        );
    });
}

/// **Quoting composes, so it is gated where composing is.**
///
/// Quote and Quote in Reply open a populated, focused draft *in this
/// conversation*; for a reader who may only watch, that draft's submit is
/// refused by app-core, so offering them is the window inviting composition it
/// cannot accept — the same defect the Ask chips had.
///
/// **Quote Elsewhere… deliberately stays.** Its draft lands in whichever
/// conversation the reader picks, very likely one they take part in, and the
/// passage is theirs to quote because they can read it. Refusing it here would
/// deny a legitimate act on account of where they happened to be reading; the
/// destination answers for the destination.
#[gpui::test]
fn a_watching_reader_may_quote_elsewhere_but_not_here(cx: &mut TestAppContext) {
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.eidola_trust = Some(eidola_trust());
        // A roster that has answered without the reader, and a second cell
        // saying this is not a notebook: the reader is only watching.
        s.participants = Some((
            "s".to_string(),
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
        ));
        s.space_settings = Some((
            "s".to_string(),
            eidola_app_core::SpaceSettings {
                notebook_participant_id: None,
                ..Default::default()
            },
        ));
    });
    let (window, view) = open_space(cx, &stores, Some("s".into()));
    seed_quotable_space(
        &view,
        window,
        cx,
        vec![fixture_post_with_block("a1", "b1", "the quick brown fox")],
    );

    let mut vcx = VisualTestContext::from_window(window, cx);
    vcx.run_until_parked();
    let at = point_in_post(&view, &vcx, "a1");
    select_whole_post(&view, &mut vcx, "a1");
    right_click(&mut vcx, at);

    view.read_with(&vcx, |v, _| {
        assert_eq!(
            v.context_menu_items_for_test(),
            Some(vec![
                "Copy".to_string(),
                "Quote Elsewhere…".to_string(),
                "Select All".to_string(),
            ]),
            "the two that compose here are gone; quoting into another \
             conversation, and reading, are not"
        );
    });
    vcx.simulate_keystrokes("escape");
    vcx.run_until_parked();

    // And the handler refuses on its own — a keystroke reaches it without
    // passing either surface that offers it.
    let drafts_before = view.read_with(&vcx, |v, _| v.draft_parents_for_test().len());
    vcx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.quote(&eidola_gui::actions::Quote, window, cx);
            v.quote_in_reply(&eidola_gui::actions::QuoteInReply, window, cx);
        })
    })
    .unwrap();
    vcx.run_until_parked();
    assert_eq!(
        view.read_with(&vcx, |v, _| v.draft_parents_for_test().len()),
        drafts_before,
        "neither handler opened a draft for a reader who may not compose here"
    );
}
/// **A verb that unmounts itself hands the keyboard back.** The failed-download
/// row's two verbs are the only tab stops in it, and both take themselves away:
/// Dismiss removes the whole row, Retry replaces it with a downloading row whose
/// verb is Cancel. Left alone, the window keeps a handle nobody paints — no
/// keystroke reaches anything and Tab restarts from the window root, the class
/// `RecordView::close_detail` and the space bands already cure.
///
/// And it restores **only from a verb that was holding it**: a pointer press
/// moves focus nowhere on macOS, so a reader standing on another row must keep
/// their place.
#[gpui::test]
fn backends_a_failed_download_verb_hands_the_keyboard_back(cx: &mut TestAppContext) {
    use eidola_app_core::{LocalModelInfo, LocalModelStatus, LocalModelsState};
    use eidola_gui::backends_settings::BackendsTab;

    let failed = LocalModelInfo {
        id: "wisp@local".into(),
        slug: "wisp".into(),
        display_name: "Wisp".into(),
        file_name: "wisp.gguf".into(),
        size_bytes: None,
        source_url: Some("https://example.com/wisp.gguf".into()),
        status: LocalModelStatus::Available,
        last_error: Some("HTTP 500".into()),
        on_disk: false,
    };
    let intact = LocalModelInfo {
        id: "tiny@local".into(),
        slug: "tiny".into(),
        display_name: "Tiny".into(),
        file_name: "tiny.gguf".into(),
        size_bytes: Some(1_000_000),
        source_url: None,
        status: LocalModelStatus::Available,
        last_error: None,
        on_disk: true,
    };
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.eidola_trust = Some(eidola_trust());
        s.backends = backends_fixture(true);
        s.local_models = Some(LocalModelsState {
            engine_path: None,
            models: vec![failed, intact],
            external: Vec::new(),
        });
    });
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SettingsView::new(stores.clone(), window, cx))
    });
    let pane = view.read_with(cx, |v, _| v.backends_pane());
    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| v.select(SettingsPane::Backends, cx));
        pane.update(cx, |p, cx| p.select_tab(BackendsTab::Local, cx));
    })
    .unwrap();
    draw_window(cx, window);

    let root = pane.read_with(cx, |p, _| p.focus_handle());
    // **The target has to be a node the reader is told about.** gpui only
    // reports focus on a node the a11y tree actually has, and an element with a
    // tracked handle but no role never gets one — the adapter logs "it has an
    // id but no role" and leaves focus at the window root, which is the whole
    // of what a handback was supposed to prevent. So the pane names itself.
    {
        use eidola_gui::probe;
        probe::set_probes_enabled(true);
        probe::clear_window(window.window_id().as_u64());
        draw_window(cx, window);
        let entries = probe::window_entries(window.window_id().as_u64());
        probe::set_probes_enabled(false);
        let (_, entry) = entries
            .iter()
            .find(|(n, _)| n == "settings/backends/pane")
            .expect("the pane carries its own landmark");
        assert_eq!(
            entry.role,
            gpui::Role::Region,
            "the handback target is a named region, not a role-less div"
        );
        assert_eq!(entry.label, "Backends");
    }

    let row = |cx: &mut TestAppContext, id: &str| {
        pane.read_with(cx, |p, _| p.model_row_focus_for_test(id))
            .unwrap_or_else(|| panic!("row {id} painted"))
    };

    // Retry: its own button becomes the downloading row's Cancel.
    let failed_row = row(cx, "wisp@local");
    cx.update_window(window, |_, window, cx| window.focus(&failed_row, cx))
        .unwrap();
    cx.update_window(window, |_, window, cx| {
        pane.update(cx, |p, cx| {
            p.retry_failed_download(
                "wisp@local",
                "https://example.com/wisp.gguf".into(),
                window,
                cx,
            )
        });
    })
    .unwrap();
    assert!(
        cx.update_window(window, |_, window, _| root.is_focused(window))
            .unwrap(),
        "Retry unmounts under the keyboard exactly as Dismiss does"
    );

    // Dismiss: the row goes entirely.
    cx.update_window(window, |_, window, cx| window.focus(&failed_row, cx))
        .unwrap();
    cx.update_window(window, |_, window, cx| {
        pane.update(cx, |p, cx| {
            p.dismiss_failed_download("wisp@local", window, cx)
        });
    })
    .unwrap();
    assert!(
        cx.update_window(window, |_, window, _| root.is_focused(window))
            .unwrap(),
        "Dismiss took its own row away, so it owed the keyboard back"
    );

    // Standing somewhere else, the same press takes nothing.
    draw_window(cx, window);
    let other_row = row(cx, "tiny@local");
    cx.update_window(window, |_, window, cx| window.focus(&other_row, cx))
        .unwrap();
    cx.update_window(window, |_, window, cx| {
        pane.update(cx, |p, cx| {
            p.dismiss_failed_download("wisp@local", window, cx)
        });
    })
    .unwrap();
    assert!(
        cx.update_window(window, |_, window, _| other_row.is_focused(window))
            .unwrap(),
        "a reader standing on another row keeps their place"
    );
}

/// **Retry is not re-entrant while its own transfer is being started.** The two
/// activations are one request twice, not alternatives — but the store's keyed
/// op slot is keep-newest, so a second press superseded the first: either the
/// first continuation was dropped while the core call it issued ran on (app-core
/// then refusing the duplicate, "already downloading", published over a retry
/// that was working), or the first task had not been polled at all and its press
/// simply never happened. Both are the same defect — a control that can be
/// activated twice for one operation.
///
/// Driven against a real core, because the pending-ness being asserted *is* the
/// live op slot (a stub store starts no task at all). The URL points at a
/// listener that accepts and never answers, so the transfer stays in flight and
/// the window being tested is genuinely open.
///
/// The assertion that pins it is the **control**, not the outcome: while the
/// operation is pending the row paints no Retry at all, so there is nothing to
/// press a second time — no tab stop, nothing to activate. The op-error
/// assertion is the symptom the finding named, kept as the honest end state.
#[gpui::test]
fn backends_a_second_retry_press_does_not_race_the_first(cx: &mut TestAppContext) {
    use eidola_app_core::{LocalModelInfo, LocalModelStatus};
    use eidola_gui::backends_settings::BackendsTab;
    use eidola_gui::probe;

    cx.executor().allow_parking();
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    let dir = tempfile::tempdir().unwrap();
    let core = std::sync::Arc::new(
        AppCore::new(dir.path().to_path_buf(), dir.path().join("data")).expect("open core"),
    );
    let stores = cx.update(|cx| Stores::for_test(core.clone(), cx));

    // A listener that accepts and never answers, so the first transfer is
    // genuinely still in flight when the second press lands — a port that
    // refuses instantly would let the first finish and hide the window.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let url = format!("http://{}/wisp.gguf", listener.local_addr().unwrap());

    // The pane lists the registry, so the seeded singletons have to be in it
    // before the Local tab has anything to draw.
    stores.backends.update(cx, |s, cx| s.refresh(cx));
    for _ in 0..8 {
        cx.run_until_parked();
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    cx.run_until_parked();

    // The row the reader is looking at: a failed download, remembering its URL.
    stores.local_models.update(cx, |s, _| {
        s.set_state_for_test(eidola_app_core::LocalModelsState {
            engine_path: None,
            external: Vec::new(),
            models: vec![LocalModelInfo {
                id: "wisp@local".into(),
                slug: "wisp".into(),
                display_name: "Wisp".into(),
                file_name: "wisp.gguf".into(),
                size_bytes: None,
                source_url: Some(url.clone()),
                status: LocalModelStatus::Available,
                last_error: Some("HTTP 500".into()),
                on_disk: false,
            }],
        })
    });

    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SettingsView::new(stores.clone(), window, cx))
    });
    let pane = view.read_with(cx, |v, _| v.backends_pane());
    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| v.select(SettingsPane::Backends, cx));
        pane.update(cx, |p, cx| p.select_tab(BackendsTab::Local, cx));
    })
    .unwrap();

    probe::set_probes_enabled(true);
    probe::clear_window(window.window_id().as_u64());
    draw_window(cx, window);
    let names: Vec<String> = probe::window_entries(window.window_id().as_u64())
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert!(
        names.contains(&"settings/backends/local/installed/0/retry".to_string()),
        "the failed row offers its Retry before anything is pending: {names:?}"
    );

    // One press.
    cx.update_window(window, |_, window, cx| {
        pane.update(cx, |p, cx| {
            p.retry_failed_download("wisp@local", url.clone(), window, cx)
        });
    })
    .unwrap();
    assert!(
        stores
            .local_models
            .read_with(cx, |s, _| s.download_pending(&url)),
        "the press owns the operation while it runs"
    );

    // ...and the control is gone, so there is no second press to make.
    probe::clear_window(window.window_id().as_u64());
    draw_window(cx, window);
    let names: Vec<String> = probe::window_entries(window.window_id().as_u64())
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    probe::set_probes_enabled(false);
    assert!(
        !names.contains(&"settings/backends/local/installed/0/retry".to_string()),
        "a pending retry must not leave a second door onto itself: {names:?}"
    );

    // The programmatic door is shut too, and nothing is published over the
    // retry that is working.
    cx.update_window(window, |_, window, cx| {
        pane.update(cx, |p, cx| {
            p.retry_failed_download("wisp@local", url.clone(), window, cx)
        });
    })
    .unwrap();
    for _ in 0..12 {
        cx.run_until_parked();
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    cx.run_until_parked();
    assert!(
        stores
            .local_models
            .read_with(cx, |s, _| s.op_error().is_none()),
        "a doubled press must not publish `already downloading` over a working retry: {:?}",
        stores
            .local_models
            .read_with(cx, |s, _| s.op_error().map(str::to_string))
    );
}

/// **The catalog's Download and the failed row's Retry are one operation.** Both
/// hand `LocalModelsStore::download` the same URL, so both land in the same
/// keyed op slot — which is keep-newest, so pressing the second door while the
/// first door's transfer was being started superseded it: the first
/// continuation was dropped while the core call it had already issued ran on,
/// and app-core then refused the duplicate ("already downloading"), publishing
/// a failure over a transfer that was working. The cure is the store's rather
/// than the pane's — `download` is a no-op for a URL whose slot is pending — so
/// every door onto it is safe and the rows are free to be presentation.
///
/// The window is **staged, not hoped for**: one `tick` polls the retry's task,
/// which issues its core call on the runtime and parks on the answer, and
/// nothing polls it again — so the transfer is genuinely running while the slot
/// is still the first press's, which is exactly the window the catalog door used
/// to reach into. The URL is a listener that accepts and never answers, so the
/// transfer stays in flight for the rest of the test.
///
/// `op_error` is the assertion with teeth: without the store's guard the second
/// press supersedes a call that has already gone out, and app-core's refusal of
/// the duplicate is what lands in the dropped continuation's place.
#[gpui::test]
fn backends_a_catalog_press_cannot_race_a_pending_retry(cx: &mut TestAppContext) {
    use eidola_app_core::{LocalModelInfo, LocalModelStatus};
    use eidola_gui::backends_settings::BackendsTab;

    cx.executor().allow_parking();
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    let dir = tempfile::tempdir().unwrap();
    let core = std::sync::Arc::new(
        AppCore::new(dir.path().to_path_buf(), dir.path().join("data")).expect("open core"),
    );
    let stores = cx.update(|cx| Stores::for_test(core.clone(), cx));

    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let url = format!("http://{}/wisp.gguf", listener.local_addr().unwrap());

    stores.backends.update(cx, |s, cx| s.refresh(cx));
    for _ in 0..8 {
        cx.run_until_parked();
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    cx.run_until_parked();

    stores.local_models.update(cx, |s, _| {
        s.set_state_for_test(eidola_app_core::LocalModelsState {
            engine_path: None,
            external: Vec::new(),
            models: vec![LocalModelInfo {
                id: "wisp@local".into(),
                slug: "wisp".into(),
                display_name: "Wisp".into(),
                file_name: "wisp.gguf".into(),
                size_bytes: None,
                source_url: Some(url.clone()),
                status: LocalModelStatus::Available,
                last_error: Some("HTTP 500".into()),
                on_disk: false,
            }],
        })
    });

    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SettingsView::new(stores.clone(), window, cx))
    });
    let pane = view.read_with(cx, |v, _| v.backends_pane());
    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| v.select(SettingsPane::Backends, cx));
        pane.update(cx, |p, cx| p.select_tab(BackendsTab::Local, cx));
    })
    .unwrap();
    draw_window(cx, window);

    // Settle first, so the task the press leaves behind is the one the tick
    // below runs.
    cx.run_until_parked();
    cx.update_window(window, |_, window, cx| {
        pane.update(cx, |p, cx| {
            p.retry_failed_download("wisp@local", url.clone(), window, cx)
        });
    })
    .unwrap();
    assert!(
        stores
            .local_models
            .read_with(cx, |s, _| s.download_pending(&url)),
        "the retry owns the slot from the frame it is pressed on"
    );

    // Poll that task exactly far enough to issue its core call, and no further:
    // its continuation stays outstanding, which is the state a second press used
    // to drop.
    let running = |cx: &mut TestAppContext| {
        let _ = cx;
        core.runtime()
            .block_on(core.local_models_state())
            .map(|s| {
                s.models.iter().any(|m| {
                    m.slug == "wisp" && matches!(m.status, LocalModelStatus::Downloading { .. })
                })
            })
            .unwrap_or(false)
    };
    let mut in_flight = false;
    for _ in 0..8 {
        cx.executor().tick();
        std::thread::sleep(std::time::Duration::from_millis(60));
        in_flight = running(cx);
        if in_flight {
            break;
        }
    }
    assert!(
        in_flight,
        "the retry's transfer is genuinely running — the window this test is about"
    );
    assert!(
        stores
            .local_models
            .read_with(cx, |s, _| s.download_pending(&url)),
        "…and its continuation has not landed, so the slot is still the retry's"
    );

    // The catalog's door, onto that same URL.
    cx.update_window(window, |_, _, cx| {
        pane.update(cx, |p, cx| p.download_catalog(&url, cx));
    })
    .unwrap();
    for _ in 0..12 {
        cx.run_until_parked();
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    cx.run_until_parked();

    assert!(
        stores
            .local_models
            .read_with(cx, |s, _| s.op_error().is_none()),
        "a second door must start nothing rather than supersede a working transfer: {:?}",
        stores
            .local_models
            .read_with(cx, |s, _| s.op_error().map(str::to_string))
    );

    let live = core
        .runtime()
        .block_on(core.local_models_state())
        .expect("read the live listing");
    let downloading: Vec<&LocalModelInfo> = live
        .models
        .iter()
        .filter(|m| matches!(m.status, LocalModelStatus::Downloading { .. }))
        .collect();
    assert_eq!(
        downloading.len(),
        1,
        "two presses, one operation, one transfer: {:?}",
        live.models
    );
    assert_eq!(downloading[0].slug, "wisp");
}

/// **One transfer is one transfer however its URL is spelled.** App-core keys
/// its download map by the model's slug, and `normalize_model_url` folds
/// equivalent spellings onto it: a Hugging Face `/blob/` file page and its
/// `/resolve/` object are the same bytes, as is either with `?download=true`.
/// A pending key taken from the raw text separates what app-core joins — the
/// failed row remembers the spelling that was pasted, the catalog offers its
/// own — so the second door missed the lookup, reached app-core, and had its
/// "already downloading" refusal published over the transfer that was working.
/// The key is therefore derived, by `resolve_model_download`, from the same
/// identity app-core deduplicates on.
///
/// Staged like its sibling above: one `tick` leaves the first press's core call
/// genuinely out and its continuation outstanding, and the listener never
/// answers, so the window stays open for the rest of the test. The teeth are
/// both halves — `download_pending` answering for a spelling nobody pressed,
/// and `op_error` staying empty after two more doors open on the same model.
/// The URL is local but carries `huggingface.co/` in its path, which is what
/// the rewrite rule keys on, so the real normalization runs against a listener
/// this machine owns.
#[gpui::test]
fn backends_an_equivalent_url_spelling_is_the_same_transfer(cx: &mut TestAppContext) {
    use eidola_app_core::{LocalModelInfo, LocalModelStatus};
    use eidola_gui::backends_settings::BackendsTab;

    cx.executor().allow_parking();
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    let dir = tempfile::tempdir().unwrap();
    let core = std::sync::Arc::new(
        AppCore::new(dir.path().to_path_buf(), dir.path().join("data")).expect("open core"),
    );
    let stores = cx.update(|cx| Stores::for_test(core.clone(), cx));

    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let host = listener.local_addr().unwrap();
    // What the reader pastes out of the address bar, and what the catalog
    // would hold for the same file.
    let blob_url = format!("http://{host}/huggingface.co/wisp-gguf/blob/main/wisp.gguf");
    let resolve_url = format!("http://{host}/huggingface.co/wisp-gguf/resolve/main/wisp.gguf");
    let query_url = format!("{resolve_url}?download=true");

    stores.backends.update(cx, |s, cx| s.refresh(cx));
    for _ in 0..8 {
        cx.run_until_parked();
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    cx.run_until_parked();

    stores.local_models.update(cx, |s, _| {
        s.set_state_for_test(eidola_app_core::LocalModelsState {
            engine_path: None,
            external: Vec::new(),
            models: vec![LocalModelInfo {
                id: "wisp@local".into(),
                slug: "wisp".into(),
                display_name: "Wisp".into(),
                file_name: "wisp.gguf".into(),
                size_bytes: None,
                source_url: Some(blob_url.clone()),
                status: LocalModelStatus::Available,
                last_error: Some("HTTP 500".into()),
                on_disk: false,
            }],
        })
    });

    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SettingsView::new(stores.clone(), window, cx))
    });
    let pane = view.read_with(cx, |v, _| v.backends_pane());
    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| v.select(SettingsPane::Backends, cx));
        pane.update(cx, |p, cx| p.select_tab(BackendsTab::Local, cx));
    })
    .unwrap();
    draw_window(cx, window);

    cx.run_until_parked();
    cx.update_window(window, |_, window, cx| {
        pane.update(cx, |p, cx| {
            p.retry_failed_download("wisp@local", blob_url.clone(), window, cx)
        });
    })
    .unwrap();
    for (spelling, name) in [
        (&blob_url, "the spelling that was pressed"),
        (&resolve_url, "the object that spelling resolves to"),
        (&query_url, "the same object with a download query"),
    ] {
        assert!(
            stores
                .local_models
                .read_with(cx, |s, _| s.download_pending(spelling)),
            "the slot must answer for {name}: {spelling}"
        );
    }

    // Poll that task exactly far enough to issue its core call, and no further.
    let running = |cx: &mut TestAppContext| {
        let _ = cx;
        core.runtime()
            .block_on(core.local_models_state())
            .map(|s| {
                s.models.iter().any(|m| {
                    m.slug == "wisp" && matches!(m.status, LocalModelStatus::Downloading { .. })
                })
            })
            .unwrap_or(false)
    };
    let mut in_flight = false;
    for _ in 0..8 {
        cx.executor().tick();
        std::thread::sleep(std::time::Duration::from_millis(60));
        in_flight = running(cx);
        if in_flight {
            break;
        }
    }
    assert!(
        in_flight,
        "the retry's transfer is genuinely running — the window this test is about"
    );
    assert!(
        stores
            .local_models
            .read_with(cx, |s, _| s.download_pending(&resolve_url)),
        "…and the slot, still the retry's, is still reachable by the other spelling"
    );

    // Two more doors onto the same model, each under a spelling of its own.
    cx.update_window(window, |_, _, cx| {
        pane.update(cx, |p, cx| p.download_catalog(&resolve_url, cx));
        pane.update(cx, |p, cx| p.download_catalog(&query_url, cx));
    })
    .unwrap();
    for _ in 0..12 {
        cx.run_until_parked();
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    cx.run_until_parked();

    assert!(
        stores
            .local_models
            .read_with(cx, |s, _| s.op_error().is_none()),
        "an equivalent spelling must start nothing rather than refuse the transfer it names: {:?}",
        stores
            .local_models
            .read_with(cx, |s, _| s.op_error().map(str::to_string))
    );

    let live = core
        .runtime()
        .block_on(core.local_models_state())
        .expect("read the live listing");
    let downloading: Vec<&LocalModelInfo> = live
        .models
        .iter()
        .filter(|m| matches!(m.status, LocalModelStatus::Downloading { .. }))
        .collect();
    assert_eq!(
        downloading.len(),
        1,
        "three spellings, one operation, one transfer: {:?}",
        live.models
    );
    assert_eq!(downloading[0].slug, "wisp");
}

/// **The catalog row yields its verb the way the Retry verb does.** A curated
/// entry is `Installed` only once its file is on disk, so while a Retry-started
/// transfer of that very entry is being set up the row went on painting a live
/// Download onto the operation already in flight. The store makes pressing it
/// harmless (above); this is the other half — a control over a settled decision
/// is not a control, so the slot says what is happening instead, with no id and
/// no probe: no tab stop, nothing to activate.
///
/// Staged with the real catalog URL, because the URL *is* what joins the two
/// doors. The press is never meant to complete, so the entry's file is put on
/// disk first: `download_local_model` refuses a file it already has before it
/// opens any connection, which keeps the settle at teardown local to this
/// machine. The frame is taken synchronously — pumping the executor would run
/// the continuation and end the very state being drawn.
#[gpui::test]
fn backends_a_catalog_row_stands_down_while_its_transfer_starts(cx: &mut TestAppContext) {
    use eidola_app_core::{LOCAL_MODEL_CATALOG, LocalModelInfo, LocalModelStatus};
    use eidola_gui::backends_settings::BackendsTab;
    use eidola_gui::probe;

    cx.executor().allow_parking();
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    let dir = tempfile::tempdir().unwrap();
    let entry = &LOCAL_MODEL_CATALOG[0];
    let models_dir = dir.path().join("data").join("models");
    std::fs::create_dir_all(&models_dir).unwrap();
    std::fs::write(models_dir.join(entry.file_name), b"").unwrap();

    let core = std::sync::Arc::new(
        AppCore::new(dir.path().to_path_buf(), dir.path().join("data")).expect("open core"),
    );
    let stores = cx.update(|cx| Stores::for_test(core.clone(), cx));

    stores.backends.update(cx, |s, cx| s.refresh(cx));
    for _ in 0..8 {
        cx.run_until_parked();
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    cx.run_until_parked();

    // The reader's situation: this catalog entry's download failed and left
    // nothing behind, so the row remembers the entry's own URL — and the
    // catalog, seeing no file, still offers Download.
    let seed = |cx: &mut TestAppContext| {
        stores.local_models.update(cx, |s, _| {
            s.set_state_for_test(eidola_app_core::LocalModelsState {
                engine_path: None,
                external: Vec::new(),
                models: vec![LocalModelInfo {
                    id: "wisp@local".into(),
                    slug: "wisp".into(),
                    display_name: "Wisp".into(),
                    file_name: "wisp.gguf".into(),
                    size_bytes: None,
                    source_url: Some(entry.url.to_string()),
                    status: LocalModelStatus::Available,
                    last_error: Some("HTTP 500".into()),
                    on_disk: false,
                }],
            })
        });
    };
    seed(cx);

    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| SettingsView::new(stores.clone(), window, cx))
    });
    let pane = view.read_with(cx, |v, _| v.backends_pane());
    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| v.select(SettingsPane::Backends, cx));
        pane.update(cx, |p, cx| p.select_tab(BackendsTab::Local, cx));
    })
    .unwrap();
    draw_window(cx, window);
    seed(cx);

    let probes = |cx: &mut TestAppContext| -> Vec<String> {
        probe::set_probes_enabled(true);
        probe::clear_window(window.window_id().as_u64());
        // Synchronous: `draw_window` would run the pending continuation with it.
        cx.update_window(window, |_, window, cx| window.draw(cx).clear())
            .unwrap();
        let names: Vec<String> = probe::window_entries(window.window_id().as_u64())
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        probe::set_probes_enabled(false);
        names
    };

    let before = probes(cx);
    assert!(
        before.contains(&"settings/backends/local/catalog/0/download".to_string()),
        "the entry is not installed, so its door stands: {before:?}"
    );

    cx.update_window(window, |_, window, cx| {
        pane.update(cx, |p, cx| {
            p.retry_failed_download("wisp@local", entry.url.to_string(), window, cx)
        });
    })
    .unwrap();
    assert!(
        stores
            .local_models
            .read_with(cx, |s, _| s.download_pending(entry.url)),
        "the retry owns this entry's transfer"
    );
    assert!(
        stores.local_models.read_with(cx, |s, _| !s
            .models()
            .iter()
            .any(|m| m.file_name == entry.file_name && m.on_disk)),
        "nothing here is installed, so the row below can only be about pendingness"
    );

    let during = probes(cx);
    assert!(
        !during.contains(&"settings/backends/local/catalog/0/download".to_string()),
        "the catalog must not offer a second door onto a transfer already starting: {during:?}"
    );
    assert!(
        !during.contains(&"settings/backends/local/installed/0/retry".to_string()),
        "nor may the row that started it"
    );
    assert!(
        during.contains(&"settings/backends/local/catalog/1/download".to_string()),
        "only this entry's door yields — the rest of the catalog is untouched: {during:?}"
    );
}

/// **A transcript Retry hands the keyboard back, because pressing it takes the
/// surface away.** Both failure surfaces in the reading column carry exactly
/// one tab stop — their Retry (`probe(Role::Button)` derives a real one) — and
/// the press ends the state the surface renders: the read leaves `Failed` as
/// it restarts, so the centred panel and the stale strip each unmount under
/// the keyboard that activated them. Left alone the window points at a handle
/// nobody paints, no keystroke reaches anything, and Tab restarts from the
/// root — the class the bottom bands and `RecordView::close_detail` cure, and
/// this branch's failed-download verbs cured beside them.
///
/// The target is `keyboard_home()`, this window's one answer for a surface
/// that borrowed the keyboard going away — the view root where the panel left
/// nothing to stand on, the reader's own place among the posts the strip kept.
///
/// Real-core, because a stub never starts the read: the unmount *is* the cell
/// leaving `Failed`, and this asserts it alongside the focus rather than
/// taking the surface's disappearance on trust. Both failures are staged after
/// the window has settled, and the assertions are taken synchronously, so no
/// in-flight load can clear them instead of the press.
#[gpui::test]
fn space_a_transcript_retry_hands_the_keyboard_back(cx: &mut TestAppContext) {
    let (stores, core, _dir, space_id) = participants_scene(cx);
    let (window, view) = open_space(cx, &stores, Some(space_id));
    cx.run_until_parked();

    let space = view.read_with(cx, |v, _| v.space().clone());
    let root = view.read_with(cx, |v, _| v.focus_handle());
    let surface = view.read_with(cx, |v, _| v.transcript_retry_focus_for_test());

    // The centred panel: a failed initial read, nothing on the page. Seeded,
    // drawn — so the panel has painted and its handle is tracked — and seeded
    // again, since the window's own read can land during that pump.
    let seed_dead_end = |cx: &mut TestAppContext| {
        cx.update_window(window, |_, _, cx| {
            space.update(cx, |s, cx| s.fail_initial_transcript_load_for_test(cx));
        })
        .unwrap();
    };
    seed_dead_end(cx);
    draw_window(cx, window);
    seed_dead_end(cx);
    assert!(
        space.read_with(cx, |s, _| s.transcript_load_failure().is_some()),
        "the panel is standing, which is what the press is about"
    );
    cx.update_window(window, |_, window, cx| window.focus(&surface, cx))
        .unwrap();
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.retry_transcript_load(window, cx));
    })
    .unwrap();
    assert!(
        space.read_with(cx, |s, _| s.transcript_load_failure().is_none()),
        "the read restarted, so the panel the reader was standing in is gone"
    );
    assert!(
        cx.update_window(window, |_, window, _| root.is_focused(window))
            .unwrap(),
        "so the keyboard goes home rather than staying on a surface nobody paints"
    );

    // The stale strip: a failed refresh over posts this window still holds.
    // Seeded, drawn — so the strip has painted and its handle is tracked — and
    // then seeded again, because the real read the first Retry started lands
    // during that pump and would otherwise resolve the cell for us.
    cx.run_until_parked();
    let seed_stale = |cx: &mut TestAppContext| {
        cx.update_window(window, |_, _, cx| {
            space.update(cx, |s, cx| {
                s.set_post_tree_for_test(vec![fixture_user_post("a1", "the question")], cx);
                s.fail_transcript_refresh_for_test(cx);
            });
        })
        .unwrap();
    };
    seed_stale(cx);
    draw_window(cx, window);
    seed_stale(cx);
    assert!(
        space.read_with(cx, |s, _| s.transcript_refresh_failure().is_some()),
        "the strip is standing"
    );
    cx.update_window(window, |_, window, cx| window.focus(&surface, cx))
        .unwrap();
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.retry_transcript_load(window, cx));
    })
    .unwrap();
    assert!(
        space.read_with(cx, |s, _| s.transcript_refresh_failure().is_none()),
        "the strip stands down while the read it asked for is in flight"
    );
    assert!(
        cx.update_window(window, |_, window, _| root.is_focused(window))
            .unwrap(),
        "and its Retry owed the keyboard back exactly as the panel's did"
    );

    // …and a reader who was not standing in the surface keeps their place: a
    // pointer press moves focus nowhere, so an unconditional restore would take
    // a composing reader's caret away.
    cx.run_until_parked();
    seed_stale(cx);
    draw_window(cx, window);
    open_space_draft(&view, window, cx, Some("a1"));
    seed_stale(cx);
    let composer = view
        .read_with(cx, |v, _| v.composer_state_for_test())
        .expect("a draft is open");
    let caret = composer.read_with(cx, |e, cx| e.focus_handle(cx));
    cx.update_window(window, |_, window, cx| window.focus(&caret, cx))
        .unwrap();
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.retry_transcript_load(window, cx));
    })
    .unwrap();
    assert!(
        cx.update_window(window, |_, window, _| caret.is_focused(window))
            .unwrap(),
        "a reader composing beside the strip keeps their caret"
    );

    drain_runtime(&core);
}
