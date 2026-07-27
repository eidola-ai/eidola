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
use eidola_app_core::changes::Change;
use eidola_app_core::error::AppError;
use eidola_app_core::updates::{
    Claim, ClaimDelta, ClaimsComparison, UpdateCheckResult, UpdateCheckSnapshot, VerifiedRelease,
};
use eidola_app_core::{
    AttestationDetail, AttestationInfo, BalancesResult, ConfigState, CredentialLifecycleInfo,
    PostBlock, PostNode, PostParticipant, RequestDetail, RequestInfo, SpaceInfo, SpaceMessage,
};
use eidola_gui::about::AboutView;
use eidola_gui::account::AccountView;
use eidola_gui::actions::{PostOnly, Send};
use eidola_gui::library::LibraryView;
use eidola_gui::onboarding::{OnboardingView, Slide};
use eidola_gui::participants_view::{EditMode, ParticipantsView};
use eidola_gui::record::{RecordDetail, RecordSection, RecordView};
use eidola_gui::settings::{SettingsPane, SettingsView};
use eidola_gui::space_view::SpaceView;
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
    view.update(cx, |v, cx| v.cancel_rename(cx));
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
    let core = std::sync::Arc::new(AppCore::new(
        _dir.path().to_path_buf(),
        _dir.path().join("data"),
    ));
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
        domain_separator: "ACT-v1:eidola:inference:production:2026-03-05".into(),
        attestation_url: None,
        appearance: eidola_app_core::config::AppearanceSetting::System,
        time_of_day_tint: eidola_app_core::config::TimeOfDayTint::On,
        light_character: eidola_app_core::config::LightCharacter::Neutral,
        font_scale: 1.0,
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
fn participants_renders_with_scroll_indicator(cx: &mut TestAppContext) {
    // Seed enough eidola-catalog models that the model-picker dropdown
    // overflows its 220px max-height — the nested scroller whose own overlay
    // indicator this exercises (a Codex P1 on PR #232). The draw below opens
    // the add form + picker; a mis-bound picker handle would panic here.
    let stores = stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.eidola_trust = Some(eidola_trust());
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
    // Roster body indicator.
    draw_frame(cx, window);
    // Open the add form + its overflowing model picker; the picker's own
    // overlay indicator binds to `picker_scroll`.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| {
            v.begin_add(window, cx);
            v.open_add_picker_for_test(cx);
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
    // beneath a top reserve that holds the first post clear of the overlaid
    // titlebar — but an empty notebook has no post above the composer, so that
    // reserve was pure dead space (total = window + reserve). `doc_reserve` is
    // now zero when there are no posts, so a sole composer fills the window
    // exactly: no scroll until the content actually overflows. (Spaces *with*
    // posts keep the reserve, so the docked composer still reaches the window
    // top when scrolled to the bottom.)
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
                AppError::ChatFailed {
                    space_id: "s".into(),
                    source: Box::new(AppError::Network {
                        message: "connection reset".into(),
                    }),
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
        let msg = v.error_for_test().expect("the failed turn shows a notice");
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
        assert!(v.error_for_test().is_none(), "retry clears the notice");
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
                AppError::ChatFailed {
                    space_id: "s".into(),
                    source: Box::new(AppError::Network {
                        message: "connection reset".into(),
                    }),
                },
                cx,
            )
        });
    })
    .unwrap();
    cx.run_until_parked();
    view.read_with(cx, |v, cx| {
        assert!(
            v.error_for_test().is_some(),
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
            v.error_for_test().is_some(),
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
    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| v.dismiss_error(cx));
    })
    .unwrap();
    view.read_with(cx, |v, cx| {
        assert!(v.error_for_test().is_none(), "dismiss clears the notice");
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
    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| v.dismiss_cascade(cx));
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

    // Type a draft far taller than the ~310px floating viewport. `InsertText`
    // leaves the caret at the end of the inserted text — below the fold.
    let long = "line of the draft that carries some words\n".repeat(40);
    editor.update(&mut vcx, |e, cx| {
        e.apply_event_for_test(EditorEvent::InsertText(long), cx)
    });
    // Scroll the page to the top so the composer's slot sits far below the fold
    // and the composer floats (capped) rather than docking.
    view.read_with(&vcx, |v, _| v.scroll_page_to_top_for_test());
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

    // Full-reveal assertion (the off-by-one regression, Task C). The docked
    // branch computes the caret's DOCUMENT position as `page_slot_doc_top +
    // editor_top_offset + caret_content_bottom`, then scrolls the page to reveal
    // it. The final page-scroll value is gpui-clamped against a frame-lagged
    // content size (and races under parallel test load), so we assert the
    // frame-independent piece the branch recorded: the slot-relative offset it
    // folded in (`caret_doc_bottom − caret_content_bottom` = `page_slot_doc_top +
    // editor_top_offset`). For a blank ⌘N space `page_slot_doc_top == 0` (no
    // posts, the sole node is the draft leaf), so this must equal `POST_PAD_Y`
    // (= 2·half_pad = 40px) — the editor's content-top offset within the slot.
    // Before the fix it was 0, so the docked reveal aimed a pad-height too high
    // and never fully revealed the line.
    let slot_offset = view.read_with(&vcx, |v, _| v.docked_caret_slot_offset_for_test());
    let post_pad_y = 40.0_f32;
    assert!(
        (slot_offset - post_pad_y).abs() < 1.0,
        "the docked reveal must fold the editor's {post_pad_y}px content-top \
         offset into the caret's document position (slot-relative offset was \
         {slot_offset}; omitting it — a value near 0 — under-scrolls the line \
         out of view)",
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
    let wrapped = AppError::ChatFailed {
        space_id: "s".into(),
        source: Box::new(AppError::Network {
            message: "dns error: failed to look up address".into(),
        }),
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
    view.read_with(cx, |v, _| {
        let msg = v
            .error_for_test()
            .expect("a failure shows the recovery notice");
        assert!(msg.contains("dns error"), "notice carries the error: {msg}");
    });

    // Dismissing clears only the notice — the space is untouched.
    cx.update_window(window, |_, _, cx| {
        view.update(cx, |v, cx| v.dismiss_error(cx));
    })
    .unwrap();
    view.read_with(cx, |v, cx| {
        assert!(v.error_for_test().is_none(), "notice dismissed");
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
        assert!(v.error_for_test().is_none(), "retry clears the notice");
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
                AppError::ChatFailed {
                    space_id: "s".into(),
                    source: Box::new(AppError::Network {
                        message: "connection reset".into(),
                    }),
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
                AppError::ChatFailed {
                    space_id: "s".into(),
                    source: Box::new(AppError::Network {
                        message: "connection reset".into(),
                    }),
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
    let core = std::sync::Arc::new(AppCore::new(
        dir.path().to_path_buf(),
        dir.path().join("data"),
    ));
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

#[gpui::test]
fn participants_view_add_and_remove(cx: &mut TestAppContext) {
    let (stores, core, _dir, space) = participants_scene(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| ParticipantsView::new(stores.clone(), space.clone(), None, window, cx))
    });
    wait_until(cx, "participants load", |cx| {
        participant_labels(&stores, &space, cx).len() == 2
    });

    // Add a participant: open the form, set its name, save.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.begin_add(window, cx));
    })
    .unwrap();
    let label = view
        .read_with(cx, |v, _| v.adding_label_state())
        .expect("add form open");
    cx.update_window(window, |_, window, cx| {
        label.update(cx, |s, cx| s.set_value("Reviewer", window, cx));
    })
    .unwrap();
    view.update(cx, |v, cx| v.save_add(cx));
    wait_until(cx, "participant added", |cx| {
        participant_labels(&stores, &space, cx).contains(&"Reviewer".to_string())
    });
    assert_eq!(participant_labels(&stores, &space, cx).len(), 3);

    // Remove it.
    let reviewer_id = stores.participants.read_with(cx, |s, _| {
        s.list(&space)
            .iter()
            .find(|p| p.label == "Reviewer")
            .unwrap()
            .id
            .clone()
    });
    view.update(cx, |v, cx| v.remove(&reviewer_id, cx));
    wait_until(cx, "participant removed", |cx| {
        participant_labels(&stores, &space, cx).len() == 2
    });

    drain_runtime(&core);
}

/// The headline fork: editing a **referenced global** ("You") writes either the
/// shared config (edit everywhere) or a per-space override (override here). The
/// view routes to the right store method per its mode.
#[gpui::test]
fn participants_view_override_vs_edit_everywhere(cx: &mut TestAppContext) {
    let (stores, core, _dir, space) = participants_scene(cx);
    let (window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| ParticipantsView::new(stores.clone(), space.clone(), None, window, cx))
    });
    wait_until(cx, "participants load", |cx| {
        participant_labels(&stores, &space, cx).len() == 2
    });

    let you = eidola_app_core::HUMAN_PARTICIPANT_ID;

    // Override here: a referenced global defaults to the this-space-only mode.
    cx.update_window(window, |_, window, cx| {
        view.update(cx, |v, cx| v.begin_edit(you, window, cx));
    })
    .unwrap();
    assert_eq!(
        view.read_with(cx, |v, _| v.editing_mode()),
        Some(EditMode::OverrideHere)
    );
    let label = view.read_with(cx, |v, _| v.editing_label_state()).unwrap();
    cx.update_window(window, |_, window, cx| {
        label.update(cx, |s, cx| s.set_value("Me", window, cx));
    })
    .unwrap();
    view.update(cx, |v, cx| v.save_edit(cx));
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
        view.update(cx, |v, cx| v.begin_edit(you, window, cx));
        view.update(cx, |v, cx| {
            v.set_edit_mode(EditMode::Everywhere, window, cx)
        });
    })
    .unwrap();
    assert_eq!(
        view.read_with(cx, |v, _| v.editing_mode()),
        Some(EditMode::Everywhere)
    );
    let label = view.read_with(cx, |v, _| v.editing_label_state()).unwrap();
    cx.update_window(window, |_, window, cx| {
        label.update(cx, |s, cx| s.set_value("Myself", window, cx));
    })
    .unwrap();
    view.update(cx, |v, cx| v.save_edit(cx));
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

/// A failed initial participant load must render Retry (not a phantom-empty
/// roster), and Retry must actually re-fetch. `ensure` declines once a `Failed`
/// cell exists, so `retry_load` is the only path back.
#[gpui::test]
fn participants_view_retry_refetches_after_failed_load(cx: &mut TestAppContext) {
    let (stores, core, _dir, space) = participants_scene(cx);
    let (_window, view) = open_view(cx, |window, cx| {
        cx.new(|cx| ParticipantsView::new(stores.clone(), space.clone(), None, window, cx))
    });
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
    view.update(cx, |v, cx| v.retry_load(cx));
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
            .read_with(cx, |s, _| s.op_error(&space_a).is_some())
    });
    stores.participants.read_with(cx, |s, _| {
        assert!(s.op_error(&space_b).is_none(), "B must not see A's error");
    });

    let b = space_b.clone();
    stores
        .participants
        .update(cx, move |s, cx| s.add(b, bad(), cx));
    wait_until(cx, "B op_error", |cx| {
        stores
            .participants
            .read_with(cx, |s, _| s.op_error(&space_b).is_some())
    });
    stores.participants.read_with(cx, |s, _| {
        assert!(
            s.op_error(&space_a).is_some(),
            "starting B's op must not clear A's error (per-space keying)"
        );
    });

    drain_runtime(&core);
}

/// P1: a "New Space from Template" failure is owned by the store (not detached)
/// and surfaced in `new_space_error`, not silently discarded.
#[gpui::test]
fn spaces_store_create_from_template_surfaces_error(cx: &mut TestAppContext) {
    let (stores, core, _dir, _space) = participants_scene(cx);
    let stores_clone = stores.clone();
    stores.spaces.update(cx, move |s, cx| {
        s.create_from_template("does-not-exist".into(), stores_clone, cx);
    });
    wait_until(cx, "create error surfaced", |cx| {
        stores
            .spaces
            .read_with(cx, |s, _| s.new_space_error().is_some())
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
            snippet: Some("some passage".into()),
        },
        eidola_app_core::PostReference {
            antecedent_action_id: "x2".into(),
            ordinal: 2,
            content_block_id: Some("by".into()),
            range_start: Some(0),
            range_end: Some(4),
            annotation: None,
            // The stored range no longer maps — the rail says so rather than
            // guessing at a remap.
            snippet: None,
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
