//! Interactive UI driver — "Playwright for eidola-gui".
//!
//! A long-running session that lets an agent (or a human in a terminal) drive
//! real, offscreen-rendered eidola windows: open a view in a known fixture
//! state, list its named elements, click/type/keystroke against it, and
//! capture screenshots — with the deterministic test dispatcher underneath,
//! so there are no animation races and no real-desktop dependencies (no
//! Accessibility/Screen Recording permissions, no visible windows, parallel
//! sessions per worktree are fine).
//!
//! Run it:
//!
//! ```text
//! cargo run -p eidola-gui --example driver
//! # or: just driver
//! ```
//!
//! The protocol is JSON lines on stdin/stdout — one request per line, one
//! response per line, always `{"ok":true,...}` or `{"ok":false,"error":…}`.
//! A `hello` line with the scene catalog is printed at startup. Commands:
//!
//! ```text
//! {"cmd":"scenes"}
//! {"cmd":"open","scene":"space_conversation"}           // optional width/height
//! {"cmd":"windows"}
//! {"cmd":"elements","window":1}                          // named probe targets
//! {"cmd":"click","window":1,"target":"chat/model-label"} // or "x"/"y"; alt/command/shift bools
//!   //   optional "button":"right" for the context-menu gesture
//! {"cmd":"drag","window":1,"from_x":300,"from_y":320,"to_x":560,"to_y":660} // press-move-release
//!   //   optional "click_count":2|3 (double/triple-click selection); "hold":true pumps frames at
//!   //   `to` so host autoscroll-while-selecting runs before release
//! {"cmd":"type","window":1,"text":"Hello there"}
//! {"cmd":"keys","window":1,"keys":"cmd-enter"}           // space-separated keystrokes
//! {"cmd":"modifiers","window":1,"alt":true}              // hold/release modifiers
//! {"cmd":"scroll","window":1,"target":"chat/transcript","dy":-300}
//! {"cmd":"resize","window":1,"width":480,"height":700}
//! {"cmd":"screenshot","window":1}                        // optional "path"
//! {"cmd":"theme","mode":"night"}                         // or "day"; optional "character": cool|neutral|warm
//! {"cmd":"locale","tag":"zh-Hans"}                       // en|es|fr|zh-Hans|zh-Hant
//! {"cmd":"settle","ms":250}                              // advance test clock + park
//! {"cmd":"close","window":1}
//! {"cmd":"quit"}
//! ```
//!
//! Element targeting comes from the probe registry (`eidola_gui::probe`):
//! every `.probe(name, role, label)` annotation in the views is listed by
//! `elements` with its painted bounds, and `click`/`scroll` accept the probe
//! name as `target`. The same annotation feeds the AccessKit tree, so the
//! driver's selector vocabulary is exactly the app's accessible surface.
//! Elements annotated with `probe_value` also report a `value` — the content
//! channel (a settled post's text, a balance, an alert's message); it is
//! `null` for everything else.
//!
//! Scenes are stub-store fixtures (no backend, no network), mirroring the
//! visual snapshot cases — deterministic scenes the agent can interact with.
//! Store-backed flows stop at the stub guard exactly as behavior tests do;
//! local interaction (composer editing, submit's local append, picker, hover
//! reveals, navigation) is fully live.

// `VisualTestAppContext` wraps the real Mac platform (offscreen Metal
// rendering), so the driver is macOS-only — same gate as `tests/visual.rs`.
#[cfg(target_os = "macos")]
// The shared, backend-free post fixtures (also used by the visual snapshot
// cases) — a branched `PostNode` tree to exercise the space view. Declared at
// the file's top level so its `#[path]` resolves against `examples/` (a real
// directory) rather than the inline `driver` module's virtual subdir.
#[cfg(target_os = "macos")]
#[path = "../tests/visual/fixtures.rs"]
mod fixtures;

#[cfg(target_os = "macos")]
mod driver {
    use super::fixtures;
    use std::collections::HashMap;
    use std::io::{BufRead, Write as _};
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::mpsc;
    use std::time::Duration;

    use eidola_app_core::error::AppError;
    use eidola_app_core::updates::{UpdateCheckResult, UpdateCheckSnapshot, VerifiedRelease};
    use eidola_app_core::{
        AttestationInfo, BalancePoolInfo, BalancesResult, ConfigState, ModelInfo, ParticipantInfo,
        ParticipantReference, PriceInfo, SpaceInfo, SpaceMessage, SpaceTemplateInfo,
        SubscriptionInfo, SubscriptionState, TemplateParticipantInfo,
    };
    use eidola_gui::about::AboutView;
    use eidola_gui::library::LibraryView;
    use eidola_gui::loadable::Loadable;
    use eidola_gui::onboarding::OnboardingView;
    use eidola_gui::probe;
    use eidola_gui::record::RecordView;
    use eidola_gui::settings::{SettingsPane, SettingsView};
    use eidola_gui::space_view::SpaceView;
    use eidola_gui::stores::{Stores, StoresStub};
    use eidola_gui::updates::UpdatesView;
    use eidola_gui::window_input::WindowInput;
    use gpui::{
        AnyWindowHandle, App, AppContext, Capslock, Modifiers, ModifiersChangedEvent, MouseButton,
        MouseDownEvent, MouseMoveEvent, MouseUpEvent, Pixels, ScrollDelta, ScrollWheelEvent, Size,
        TouchPhase, VisualTestAppContext, point, px, size,
    };
    use gpui_component::{Root, ThemeMode};
    use gpui_component_assets::Assets;
    use serde::Deserialize;
    use serde_json::{Value, json};

    // ---------------------------------------------------------------------------
    // Protocol
    // ---------------------------------------------------------------------------

    fn one() -> usize {
        1
    }

    #[derive(Deserialize)]
    #[serde(tag = "cmd", rename_all = "snake_case", deny_unknown_fields)]
    enum Cmd {
        Scenes,
        Open {
            scene: String,
            width: Option<f32>,
            height: Option<f32>,
        },
        Windows,
        Elements {
            window: u64,
        },
        Click {
            window: u64,
            target: Option<String>,
            x: Option<f32>,
            y: Option<f32>,
            #[serde(default)]
            alt: bool,
            #[serde(default)]
            command: bool,
            #[serde(default)]
            shift: bool,
            /// `"right"` opens a context menu (the app's own — see the space
            /// view's `context_menu`); anything else is an ordinary left click.
            #[serde(default)]
            button: Option<String>,
        },
        Drag {
            window: u64,
            from_x: f32,
            from_y: f32,
            to_x: f32,
            to_y: f32,
            #[serde(default = "one")]
            click_count: usize,
            /// Hold the button at `to` (running any host autoscroll loop to a
            /// settled state) before releasing.
            #[serde(default)]
            hold: bool,
        },
        Type {
            window: u64,
            text: String,
        },
        Keys {
            window: u64,
            keys: String,
        },
        Modifiers {
            window: u64,
            #[serde(default)]
            alt: bool,
            #[serde(default)]
            command: bool,
            #[serde(default)]
            shift: bool,
            #[serde(default)]
            ctrl: bool,
        },
        Scroll {
            window: u64,
            target: Option<String>,
            x: Option<f32>,
            y: Option<f32>,
            #[serde(default)]
            dx: f32,
            #[serde(default)]
            dy: f32,
        },
        Resize {
            window: u64,
            width: f32,
            height: f32,
        },
        Screenshot {
            window: u64,
            path: Option<String>,
        },
        Theme {
            mode: String,
            /// Optional circadian light character: `cool` / `neutral`
            /// (default) / `warm` — renders the tinted palette variants
            /// (Sunrise/Sunset/Dawn/Dusk) that production derives from the
            /// sun.
            character: Option<String>,
        },
        /// Switch the display language to one of the shipped locales (`en`,
        /// `es`, `fr`, `zh-Hans`, `zh-Hant`). The driver starts on the source
        /// locale, exactly as tests do.
        Locale {
            tag: String,
        },
        Settle {
            ms: Option<u64>,
        },
        Close {
            window: u64,
        },
        Quit,
    }

    // ---------------------------------------------------------------------------
    // Scenes — stub-store fixtures mirroring tests/visual/cases.rs
    // ---------------------------------------------------------------------------

    struct Scene {
        name: &'static str,
        description: &'static str,
        default_size: Size<Pixels>,
        build: fn(&mut gpui::Window, &mut App) -> gpui::Entity<Root>,
    }

    fn scenes() -> Vec<Scene> {
        fn root<V: gpui::Render + 'static>(
            view: gpui::Entity<V>,
            window: &mut gpui::Window,
            cx: &mut App,
        ) -> gpui::Entity<Root> {
            cx.new(|cx| Root::new(view, window, cx))
        }

        vec![
            Scene {
                name: "space_blank",
                description: "Space view: a brand-new blank space (composer open at top)",
                default_size: size(px(760.), px(680.)),
                build: |window, cx| {
                    let stores = ready_stores(cx);
                    let view =
                        cx.new(|cx| SpaceView::new(stores, None, WindowInput::new(cx), window, cx));
                    root(view, window, cx)
                },
            },
            Scene {
                name: "space_conversation",
                description: "Space view: an existing conversation with a docked tail composer at the branch end",
                default_size: size(px(760.), px(680.)),
                build: |window, cx| {
                    let stores = ready_stores(cx);
                    // An *existing* space (Some id) opens without a composer.
                    let view = cx.new(|cx| {
                        SpaceView::new(
                            stores,
                            Some("demo".into()),
                            WindowInput::new(cx),
                            window,
                            cx,
                        )
                    });
                    let space = view.read(cx).space().clone();
                    space.update(cx, |s, cx| s.set_messages_for_test(conversation(), cx));
                    root(view, window, cx)
                },
            },
            Scene {
                name: "space_docked_composer",
                description: "Space view: a populated conversation settled at the document floor with its active composer fully docked",
                default_size: size(px(900.), px(720.)),
                build: |window, cx| {
                    let stores = ready_stores(cx);
                    let view = cx.new(|cx| {
                        SpaceView::new(
                            stores,
                            Some("demo".into()),
                            WindowInput::new(cx),
                            window,
                            cx,
                        )
                    });
                    view.update(cx, |view, cx| {
                        view.space().update(cx, |space, cx| {
                            space.set_post_tree_for_test(fixtures::kitchen_sink_posts(), cx)
                        });
                        view.seed_draft_quote_for_test(
                            Some("a9"),
                            "I want to preserve that distinction in the final paragraph.",
                            vec![],
                            window,
                            cx,
                        );
                        view.set_page_scroll_for_test(-100_000.0);
                    });
                    root(view, window, cx)
                },
            },
            Scene {
                name: "space_long_post",
                description: "Space view: a conversation whose assistant reply is far taller than the window (selection repro)",
                default_size: size(px(760.), px(680.)),
                build: |window, cx| {
                    let stores = ready_stores(cx);
                    let view = cx.new(|cx| {
                        SpaceView::new(
                            stores,
                            Some("demo".into()),
                            WindowInput::new(cx),
                            window,
                            cx,
                        )
                    });
                    let space = view.read(cx).space().clone();
                    space.update(cx, |s, cx| s.set_messages_for_test(long_conversation(), cx));
                    root(view, window, cx)
                },
            },
            Scene {
                name: "space_structured",
                description: "Space view: a branched space whose spine reply is markdown-heavy (tables, nested lists, quote, rules) — structural repro of a user-reported selection failure",
                default_size: size(px(760.), px(680.)),
                build: |window, cx| {
                    let stores = ready_stores(cx);
                    let view = cx.new(|cx| {
                        SpaceView::new(
                            stores,
                            Some("demo".into()),
                            WindowInput::new(cx),
                            window,
                            cx,
                        )
                    });
                    let space = view.read(cx).space().clone();
                    space.update(cx, |s, cx| s.set_post_tree_for_test(structured_posts(), cx));
                    root(view, window, cx)
                },
            },
            Scene {
                name: "space_long_metadata",
                description: "Space view: compact post metadata with long author and backend labels",
                default_size: size(px(760.), px(680.)),
                build: |window, cx| {
                    let stores = ready_stores(cx);
                    let view = cx.new(|cx| {
                        SpaceView::new(
                            stores,
                            Some("demo".into()),
                            WindowInput::new(cx),
                            window,
                            cx,
                        )
                    });
                    let space = view.read(cx).space().clone();
                    space.update(cx, |s, cx| {
                        s.set_post_tree_for_test(long_metadata_posts(), cx)
                    });
                    root(view, window, cx)
                },
            },
            Scene {
                name: "markdown_table",
                description: "Space view: an assistant reply carrying GFM tables (aligned columns, styled cells, a wide table whose cells wrap) — the table display-mode QA scene; the tail composer is live for edit-mode QA",
                default_size: size(px(760.), px(680.)),
                build: |window, cx| {
                    let stores = ready_stores(cx);
                    let view = cx.new(|cx| {
                        SpaceView::new(
                            stores,
                            Some("demo".into()),
                            WindowInput::new(cx),
                            window,
                            cx,
                        )
                    });
                    let space = view.read(cx).space().clone();
                    space.update(cx, |s, cx| {
                        s.set_messages_for_test(table_conversation(), cx)
                    });
                    root(view, window, cx)
                },
            },
            Scene {
                name: "space_thinking",
                description: "Space view: the thinking disclosure in all three states — a finished post whose persisted `thinking` block is collapsed, one already expanded, and a live streaming turn still saying \"Thinking…\"",
                default_size: size(px(860.), px(760.)),
                build: |window, cx| {
                    let stores = ready_stores(cx);
                    let view = cx.new(|cx| {
                        SpaceView::new(
                            stores,
                            Some("demo".into()),
                            WindowInput::new(cx),
                            window,
                            cx,
                        )
                    });
                    let space = view.read(cx).space().clone();
                    space.update(cx, |s, cx| {
                        s.set_post_tree_for_test(thinking_posts(), cx);
                        // The finished reply at index 3 opens its disclosure, so
                        // one screenshot carries the collapsed and expanded
                        // states side by side — both reading their *persisted*
                        // thinking block, not a live buffer.
                        s.toggle_message_reasoning(3, cx);
                        // …and a live turn, still producing.
                        s.push_streaming_turn_for_test(
                            None,
                            Some("a4".into()),
                            eidola_gui::space::StreamingResponse {
                                reasoning: "Weighing whether to bring up Rayleigh scattering \
                                            again or start from the geometry…"
                                    .into(),
                                content: "The short answer is that the same scattering that \
                                          makes the sky blue"
                                    .into(),
                                expanded: false,
                                error: None,
                            },
                            cx,
                        );
                    });
                    root(view, window, cx)
                },
            },
            Scene {
                name: "space_model_loading",
                description: "Space view: a turn waiting on its local engine — the streaming leaf reads \"Loading model…\" (task 29) instead of an unexplained silence",
                default_size: size(px(860.), px(760.)),
                build: |window, cx| {
                    // The responding agent runs an engine-served model that is
                    // still warming, which is exactly the correlation the
                    // readout is keyed on.
                    let stores = stub_stores(cx, |s| {
                        s.config_state = Some(config_state(true));
                        s.eidola_trust = Some(eidola_trust());
                        s.models = models();
                        s.backends = backends();
                        s.templates = templates_fixture();
                        let mut local = local_models_state();
                        for m in &mut local.models {
                            if m.id == "gemma-4-E4B_q4_0-it@local" {
                                m.status = eidola_app_core::LocalModelStatus::Loading;
                            }
                        }
                        s.local_models = Some(local);
                        let (space_id, mut people) = participants_fixture();
                        if let Some(agent) = people.iter_mut().find(|p| p.id == "agent-assistant") {
                            agent.model_ref = Some("gemma-4-E4B_q4_0-it@local".into());
                        }
                        s.participants = Some((space_id, people));
                    });
                    let view = cx.new(|cx| {
                        SpaceView::new(
                            stores,
                            Some("demo".into()),
                            WindowInput::new(cx),
                            window,
                            cx,
                        )
                    });
                    let space = view.read(cx).space().clone();
                    space.update(cx, |s, cx| {
                        s.set_messages_for_test(conversation(), cx);
                        s.push_streaming_turn_for_test(
                            Some("agent-assistant".into()),
                            None,
                            eidola_gui::space::StreamingResponse::default(),
                            cx,
                        );
                    });
                    root(view, window, cx)
                },
            },
            Scene {
                name: "space_transcript_failed",
                description: "Space view: a conversation whose initial transcript read failed — the reading column's own error and Retry, the one page with no composer to act from",
                default_size: size(px(760.), px(680.)),
                build: |window, cx| {
                    let stores = ready_stores(cx);
                    let view = cx.new(|cx| {
                        SpaceView::new(
                            stores,
                            Some("demo".into()),
                            WindowInput::new(cx),
                            window,
                            cx,
                        )
                    });
                    let space = view.read(cx).space().clone();
                    space.update(cx, |s, cx| s.fail_initial_transcript_load_for_test(cx));
                    root(view, window, cx)
                },
            },
            Scene {
                name: "space_transcript_stale",
                description: "Space view: a refresh that failed over a long conversation, parked at the tail — every post stays, and the quiet \"couldn't refresh\" strip floats where the reader is",
                default_size: size(px(760.), px(680.)),
                build: |window, cx| {
                    let stores = ready_stores(cx);
                    let view = cx.new(|cx| {
                        SpaceView::new(
                            stores,
                            Some("demo".into()),
                            WindowInput::new(cx),
                            window,
                            cx,
                        )
                    });
                    view.update(cx, |view, cx| {
                        view.space().update(cx, |space, cx| {
                            space.set_messages_for_test(long_conversation(), cx);
                            space.fail_transcript_refresh_for_test(cx);
                        });
                        // Where a conversation actually sits: at its tail.
                        view.set_page_scroll_for_test(-100_000.0);
                    });
                    root(view, window, cx)
                },
            },
            Scene {
                name: "space_branches",
                description: "Space view: a branched post tree with docked tail drafts (kitchen-sink fixture)",
                default_size: size(px(900.), px(700.)),
                build: |window, cx| {
                    let stores = ready_stores(cx);
                    let view = cx.new(|cx| {
                        SpaceView::new(
                            stores,
                            Some("demo".into()),
                            WindowInput::new(cx),
                            window,
                            cx,
                        )
                    });
                    let space = view.read(cx).space().clone();
                    space.update(cx, |s, cx| {
                        s.set_post_tree_for_test(fixtures::kitchen_sink_posts(), cx)
                    });
                    root(view, window, cx)
                },
            },
            Scene {
                name: "space_off_branch_composer",
                description: "Space view: a long active composer floating over a sibling branch whose own tail draft remains compact",
                default_size: size(px(760.), px(680.)),
                build: |window, cx| {
                    let stores = ready_stores(cx);
                    let view = cx.new(|cx| {
                        SpaceView::new(
                            stores,
                            Some("demo".into()),
                            WindowInput::new(cx),
                            window,
                            cx,
                        )
                    });
                    let long = (0..36)
                        .map(|i| {
                            format!(
                                "Paragraph {i} belongs only to the composer on the tangent branch."
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("\n\n");
                    view.update(cx, |view, cx| {
                        view.space().update(cx, |space, cx| {
                            space.set_post_tree_for_test(fixtures::kitchen_sink_posts(), cx)
                        });
                        view.seed_draft_quote_for_test(Some("a6"), &long, vec![], window, cx);
                    });
                    root(view, window, cx)
                },
            },
            Scene {
                name: "space_quotes",
                description: "Space view: quoted references — a source post with highlighted quoted passages (single + overlapping), two replies carrying embed quote blocks and their footnote rails",
                default_size: size(px(860.), px(760.)),
                build: |window, cx| {
                    let stores = ready_stores(cx);
                    let view = cx.new(|cx| {
                        SpaceView::new(
                            stores,
                            Some("demo".into()),
                            WindowInput::new(cx),
                            window,
                            cx,
                        )
                    });
                    let space = view.read(cx).space().clone();
                    let (posts, incoming) = fixtures::quoted_reference_posts();
                    space.update(cx, |s, cx| {
                        s.set_post_tree_for_test(posts, cx);
                        for (action_id, refs) in incoming {
                            s.seed_incoming_references_for_test(
                                action_id.clone(),
                                refs.iter()
                                    .map(|r| eidola_app_core::IncomingReference {
                                        action_id: r.action_id.clone(),
                                        item_id: format!("item-of-{}", r.action_id),
                                        space_id: "demo".into(),
                                        ordinal: 1,
                                        content_block_id: Some(r.block_id.clone()),
                                        range_start: Some(r.range.0),
                                        range_end: Some(r.range.1),
                                        annotation: None,
                                        created_at: 0,
                                        author_label: "Ada".into(),
                                        author_kind: "agent".into(),
                                        space_title: None,
                                    })
                                    .collect(),
                            );
                        }
                    });
                    root(view, window, cx)
                },
            },
            Scene {
                name: "space_cross_space_quote",
                description: "Space view: a persisted post quoting one passage from this conversation and two from elsewhere — the footnote rail naming each author from the source it has (task 68)",
                default_size: size(px(860.), px(760.)),
                build: |window, cx| {
                    let stores = ready_stores(cx);
                    let view = cx.new(|cx| {
                        SpaceView::new(
                            stores,
                            Some("demo".into()),
                            WindowInput::new(cx),
                            window,
                            cx,
                        )
                    });
                    let space = view.read(cx).space().clone();
                    space.update(cx, |s, cx| {
                        s.set_post_tree_for_test(fixtures::cross_space_reference_posts(), cx)
                    });
                    root(view, window, cx)
                },
            },
            Scene {
                name: "space_delegation_report",
                description: "Space view: a delegated conversation's report — the owning agent's post quoting each helper's finding, the footnote rail saying how that conversation stopped",
                default_size: size(px(860.), px(620.)),
                build: |window, cx| {
                    let stores = ready_stores(cx);
                    let view = cx.new(|cx| {
                        SpaceView::new(
                            stores,
                            Some("demo".into()),
                            WindowInput::new(cx),
                            window,
                            cx,
                        )
                    });
                    let space = view.read(cx).space().clone();
                    space.update(cx, |s, cx| {
                        s.set_post_tree_for_test(fixtures::delegation_report_posts(), cx)
                    });
                    root(view, window, cx)
                },
            },
            Scene {
                name: "space_inspector",
                description: "Space view: the inspector open beside the conversation (a real split — title, cascade stepper, router picker at its Off default)",
                default_size: size(px(1040.), px(760.)),
                build: |window, cx| {
                    let stores = inspector_stores(cx, None);
                    let view = cx.new(|cx| {
                        SpaceView::new(
                            stores,
                            Some("demo".into()),
                            WindowInput::new(cx),
                            window,
                            cx,
                        )
                    });
                    let space = view.read(cx).space().clone();
                    space.update(cx, |s, cx| s.set_messages_for_test(conversation(), cx));
                    view.update(cx, |v, cx| v.set_inspector_open_for_test(true, window, cx));
                    root(view, window, cx)
                },
            },
            Scene {
                name: "space_inspector_remote_router",
                description: "Space view: the inspector with a remote (eidola) router selected — the mandatory per-call cost copy under the row",
                default_size: size(px(1040.), px(760.)),
                build: |window, cx| {
                    let stores = inspector_stores(cx, Some("gemma4-31b@eidola"));
                    let view = cx.new(|cx| {
                        SpaceView::new(
                            stores,
                            Some("demo".into()),
                            WindowInput::new(cx),
                            window,
                            cx,
                        )
                    });
                    let space = view.read(cx).space().clone();
                    space.update(cx, |s, cx| s.set_messages_for_test(conversation(), cx));
                    view.update(cx, |v, cx| v.set_inspector_open_for_test(true, window, cx));
                    root(view, window, cx)
                },
            },
            Scene {
                name: "space_inspector_participants",
                description: "Space view: the inspector's Participants section with one member's disclosure open — the fork chips, model picker, system prompt and notify control",
                default_size: size(px(1040.), px(760.)),
                build: |window, cx| {
                    let stores = inspector_stores(cx, None);
                    let view = cx.new(|cx| {
                        SpaceView::new(
                            stores,
                            Some("demo".into()),
                            WindowInput::new(cx),
                            window,
                            cx,
                        )
                    });
                    let space = view.read(cx).space().clone();
                    space.update(cx, |s, cx| s.set_messages_for_test(conversation(), cx));
                    view.update(cx, |v, cx| {
                        v.set_inspector_open_for_test(true, window, cx);
                        v.inspector_toggle_participant("agent-assistant", window, cx);
                    });
                    root(view, window, cx)
                },
            },
            Scene {
                name: "space_new_inspector",
                description: "Space view: a brand-new (⌘N) space with zero posts — its Participants section is live from birth, because the space is created when its window opens",
                default_size: size(px(1040.), px(760.)),
                build: |window, cx| {
                    let stores = inspector_stores(cx, None);
                    let view = cx.new(|cx| {
                        SpaceView::new(stores.clone(), None, WindowInput::new(cx), window, cx)
                    });
                    // ⌘N mints the id, so the fixtures are seeded onto the
                    // space this window just created rather than a constant.
                    let space_id = view.read(cx).space().read(cx).id().to_string();
                    stores.participants.update(cx, |p, _| {
                        p.seed_for_test(&space_id, participants_fixture().1)
                    });
                    stores.space_settings.update(cx, |s, _| {
                        s.seed_for_test(&space_id, eidola_app_core::SpaceSettings::default())
                    });
                    view.update(cx, |v, cx| {
                        v.set_inspector_open_for_test(true, window, cx);
                        v.inspector_toggle_participant("agent-assistant", window, cx);
                    });
                    root(view, window, cx)
                },
            },
            Scene {
                name: "space_inspector_share_agent",
                description: "Space view: the inspector's Participants section with a space-owned agent's disclosure open and its share confirmation armed — the one-way promote affordance (task 36)",
                default_size: size(px(1040.), px(760.)),
                build: |window, cx| {
                    let stores = inspector_stores(cx, None);
                    let view = cx.new(|cx| {
                        SpaceView::new(
                            stores,
                            Some("demo".into()),
                            WindowInput::new(cx),
                            window,
                            cx,
                        )
                    });
                    let space = view.read(cx).space().clone();
                    space.update(cx, |s, cx| s.set_messages_for_test(conversation(), cx));
                    view.update(cx, |v, cx| {
                        v.set_inspector_open_for_test(true, window, cx);
                        v.inspector_toggle_participant("agent-assistant", window, cx);
                        v.inspector_begin_promote(window, cx);
                    });
                    root(view, window, cx)
                },
            },
            Scene {
                name: "space_traces",
                description: "Space view: trace disclosures — an answered turn's tool rounds under its reply, and three declines stacked in the gap under the post they answered (two agents, one of them asked twice)",
                default_size: size(px(860.), px(760.)),
                build: |window, cx| {
                    let stores = ready_stores(cx);
                    let view = cx.new(|cx| {
                        SpaceView::new(
                            stores,
                            Some("demo".into()),
                            WindowInput::new(cx),
                            window,
                            cx,
                        )
                    });
                    let space = view.read(cx).space().clone();
                    let (posts, traces) = fixtures::trace_posts();
                    space.update(cx, |s, cx| {
                        s.set_post_tree_for_test(posts, cx);
                        s.seed_traces_for_test(traces);
                    });
                    root(view, window, cx)
                },
            },
            Scene {
                name: "space_quote_draft",
                description: "Space view: composing with a quote — a pending reference in the active draft (the embed block in the body, the footnote below it with its remove affordance)",
                default_size: size(px(860.), px(760.)),
                build: |window, cx| {
                    let stores = ready_stores(cx);
                    let view = cx.new(|cx| {
                        SpaceView::new(
                            stores,
                            Some("demo".into()),
                            WindowInput::new(cx),
                            window,
                            cx,
                        )
                    });
                    let space = view.read(cx).space().clone();
                    let (posts, _) = fixtures::quoted_reference_posts();
                    // Just the source post: the composing draft under it is
                    // what this scene is about.
                    space.update(cx, |s, cx| {
                        s.set_post_tree_for_test(posts.into_iter().take(1).collect(), cx)
                    });
                    view.update(cx, |v, cx| {
                        v.seed_draft_quote_for_test(
                            Some("q1"),
                            "That's the sentence I keep snagging on:\n\n{{ embed 1 }}\n\nIf \
                             the care is real, doesn't the shepherd analogy quietly concede \
                             Socrates' point?",
                            vec![(1, "kimi-k2", "the care is real")],
                            window,
                            cx,
                        )
                    });
                    root(view, window, cx)
                },
            },
            Scene {
                name: "space_quote_destination",
                description: "Space view: quoting into another conversation — the destination picker over the page (task 37's creation UI)",
                default_size: size(px(860.), px(760.)),
                build: |window, cx| {
                    let stores = stub_stores(cx, |s| {
                        s.config_state = Some(config_state(true));
                        s.eidola_trust = Some(eidola_trust());
                        s.models = models();
                        s.backends = backends();
                        s.participants = Some(participants_fixture());
                        s.spaces = fixtures::destination_spaces();
                    });
                    let view = cx.new(|cx| {
                        SpaceView::new(
                            stores,
                            Some("demo".into()),
                            WindowInput::new(cx),
                            window,
                            cx,
                        )
                    });
                    let space = view.read(cx).space().clone();
                    let (posts, _) = fixtures::quoted_reference_posts();
                    space.update(cx, |s, cx| {
                        s.set_post_tree_for_test(posts.into_iter().take(1).collect(), cx)
                    });
                    view.update(cx, |v, cx| {
                        v.seed_quote_destination_for_test(
                            "q1",
                            "blk-1",
                            "kimi-k2",
                            "the care is real",
                            None,
                            cx,
                        )
                    });
                    let _ = window;
                    root(view, window, cx)
                },
            },
            Scene {
                name: "space_quote_visibility",
                description: "Space view: the visibility statement — a chosen destination and what quoting into it means, the step the reader confirms (task 37)",
                default_size: size(px(860.), px(760.)),
                build: |window, cx| {
                    let stores = stub_stores(cx, |s| {
                        s.config_state = Some(config_state(true));
                        s.eidola_trust = Some(eidola_trust());
                        s.models = models();
                        s.backends = backends();
                        s.participants = Some(participants_fixture());
                        s.spaces = fixtures::destination_spaces();
                    });
                    let view = cx.new(|cx| {
                        SpaceView::new(
                            stores,
                            Some("demo".into()),
                            WindowInput::new(cx),
                            window,
                            cx,
                        )
                    });
                    let space = view.read(cx).space().clone();
                    let (posts, _) = fixtures::quoted_reference_posts();
                    space.update(cx, |s, cx| {
                        s.set_post_tree_for_test(posts.into_iter().take(1).collect(), cx)
                    });
                    view.update(cx, |v, cx| {
                        v.seed_quote_destination_for_test(
                            "q1",
                            "blk-1",
                            "kimi-k2",
                            "the care is real",
                            Some(("tides", "Tides and the moon")),
                            cx,
                        )
                    });
                    let _ = window;
                    root(view, window, cx)
                },
            },
            Scene {
                name: "space_reference_denied",
                description: "Space view: a quote followed into a conversation you take no part in — the quiet, non-leaking refusal (task 37)",
                default_size: size(px(860.), px(760.)),
                build: |window, cx| {
                    let stores = ready_stores(cx);
                    let view = cx.new(|cx| {
                        SpaceView::new(
                            stores,
                            Some("demo".into()),
                            WindowInput::new(cx),
                            window,
                            cx,
                        )
                    });
                    let space = view.read(cx).space().clone();
                    let (posts, _) = fixtures::quoted_reference_posts();
                    space.update(cx, |s, cx| s.set_post_tree_for_test(posts, cx));
                    view.update(cx, |v, cx| {
                        v.report_navigation_failure_for_test(
                            eidola_app_core::error::AppError::NotAParticipant {
                                participant_id: "p-ada".into(),
                                action_id: "a-private".into(),
                            },
                            cx,
                        )
                    });
                    root(view, window, cx)
                },
            },
            Scene {
                name: "space_inspector_invite",
                description: "Space inspector: the grant — inviting an agent from another conversation as an observer, with the sentence that says what sharing costs (task 37)",
                default_size: size(px(900.), px(760.)),
                build: |window, cx| {
                    let stores = ready_stores(cx);
                    let view = cx.new(|cx| {
                        SpaceView::new(
                            stores,
                            Some("demo".into()),
                            WindowInput::new(cx),
                            window,
                            cx,
                        )
                    });
                    let space = view.read(cx).space().clone();
                    space.update(cx, |s, cx| {
                        s.set_post_tree_for_test(fixtures::kitchen_sink_posts(), cx)
                    });
                    view.update(cx, |v, cx| {
                        v.set_inspector_open_for_test(true, window, cx);
                        v.inspector_begin_invite(window, cx);
                        v.seed_invite_candidates_for_test(fixtures::grantable_agents(), cx);
                        v.inspector_arm_invite("agent-mara", window, cx);
                    });
                    root(view, window, cx)
                },
            },
            Scene {
                name: "onboarding",
                description: "Onboarding window: the first-run 'Get Started' slide flow (scroll-snap, branching CTAs)",
                default_size: size(px(640.), px(760.)),
                build: |window, cx| {
                    let stores = ready_stores(cx);
                    let view = cx.new(|cx| OnboardingView::new(stores, window, cx));
                    root(view, window, cx)
                },
            },
            Scene {
                name: "space_templates",
                description: "Settings → Space Templates pane: the template registry (Default + a saved multi-agent template)",
                default_size: size(px(620.), px(520.)),
                build: |window, cx| {
                    let stores = ready_stores(cx);
                    let view = cx.new(|cx| SettingsView::new(stores, window, cx));
                    view.update(cx, |v, cx| v.select(SettingsPane::Templates, cx));
                    root(view, window, cx)
                },
            },
            Scene {
                name: "settings_agents",
                description: "Settings → Agents pane: the shared agent library (task 36) — each row with its notebook, edit and retire verbs",
                default_size: size(px(620.), px(520.)),
                build: |window, cx| {
                    let stores = ready_stores(cx);
                    let view = cx.new(|cx| SettingsView::new(stores, window, cx));
                    view.update(cx, |v, cx| v.select(SettingsPane::Agents, cx));
                    root(view, window, cx)
                },
            },
            Scene {
                name: "settings_agents_retire",
                description: "Settings → Agents pane with a retirement armed — the confirmation that says the notebook goes too",
                default_size: size(px(620.), px(520.)),
                build: |window, cx| {
                    let stores = ready_stores(cx);
                    let view = cx.new(|cx| SettingsView::new(stores, window, cx));
                    view.update(cx, |v, cx| {
                        v.select(SettingsPane::Agents, cx);
                        v.agents_pane()
                            .update(cx, |p, cx| p.arm_retire("agent-ada", window, cx));
                    });
                    root(view, window, cx)
                },
            },
            Scene {
                name: "about",
                description: "About window — the localization pilot surface; pair it with the `locale` command",
                default_size: size(px(360.), px(420.)),
                build: |window, cx| {
                    let view = cx.new(|cx| AboutView::new(window, cx));
                    root(view, window, cx)
                },
            },
            Scene {
                name: "library",
                description: "Library window with six spaces (hover/rename/archive)",
                default_size: size(px(520.), px(620.)),
                build: |window, cx| {
                    let stores = stub_stores(cx, |s| s.spaces = library_spaces());
                    let view = cx.new(|cx| LibraryView::new(stores, window, cx));
                    root(view, window, cx)
                },
            },
            Scene {
                name: "settings",
                description: "Settings window: funded account, plans, wallet history",
                default_size: size(px(620.), px(520.)),
                build: |window, cx| {
                    let stores = settings_stores(cx);
                    let view = cx.new(|cx| SettingsView::new(stores, window, cx));
                    root(view, window, cx)
                },
            },
            Scene {
                name: "settings_backends_local_failed_download",
                description: "Settings ▸ Backends ▸ Local: a download that failed and left nothing on disk — the row affords Retry and Dismiss",
                default_size: size(px(620.), px(560.)),
                build: |window, cx| {
                    use eidola_gui::backends_settings::{BackendsSettingsView, BackendsTab};
                    let stores = stub_stores(cx, |s| {
                        s.config_state = Some(config_state(true));
                        s.eidola_trust = Some(eidola_trust());
                        s.backends = backends();
                        s.local_models = Some(local_models_with_failed_download());
                    });
                    let view = cx.new(|cx| BackendsSettingsView::new(stores, window, cx));
                    view.update(cx, |v, cx| v.select_tab(BackendsTab::Local, cx));
                    root(view, window, cx)
                },
            },
            Scene {
                name: "settings_account_unsubscribed",
                description: "Settings ▸ Account: an account that has never transacted — the answer and a re-check, no billing door, the full plans list",
                default_size: size(px(620.), px(620.)),
                build: |window, cx| {
                    let stores =
                        account_stores(cx, Some(subscription(SubscriptionState::NoCustomer)));
                    let view = cx.new(|cx| SettingsView::new(stores, window, cx));
                    view.update(cx, |v, cx| v.select(SettingsPane::Account, cx));
                    root(view, window, cx)
                },
            },
            Scene {
                name: "settings_account_subscribed",
                description: "Settings ▸ Account: an active subscription — Manage subscription replaces the recurring plans, one-time top-ups stay",
                default_size: size(px(620.), px(620.)),
                build: |window, cx| {
                    let stores = account_stores(cx, Some(subscription(SubscriptionState::Active)));
                    let view = cx.new(|cx| SettingsView::new(stores, window, cx));
                    view.update(cx, |v, cx| v.select(SettingsPane::Account, cx));
                    root(view, window, cx)
                },
            },
            Scene {
                name: "settings_account_lapsed",
                description: "Settings ▸ Account: a payment customer with nothing in force — billing and receipts, and every plan still offered",
                default_size: size(px(620.), px(620.)),
                build: |window, cx| {
                    let stores =
                        account_stores(cx, Some(subscription(SubscriptionState::Inactive)));
                    let view = cx.new(|cx| SettingsView::new(stores, window, cx));
                    view.update(cx, |v, cx| v.select(SettingsPane::Account, cx));
                    root(view, window, cx)
                },
            },
            Scene {
                name: "settings_account_subscription_checking",
                description: "Settings ▸ Account: the subscription read is still in flight",
                default_size: size(px(620.), px(620.)),
                build: |window, cx| {
                    let stores = account_stores(cx, None);
                    stores.account.update(cx, |s, cx| {
                        s.set_subscription_for_test(Loadable::Loading, cx)
                    });
                    let view = cx.new(|cx| SettingsView::new(stores, window, cx));
                    view.update(cx, |v, cx| v.select(SettingsPane::Account, cx));
                    root(view, window, cx)
                },
            },
            Scene {
                name: "settings_account_subscription_failed",
                description: "Settings ▸ Account: the subscription read failed — the honest retry, with every plan still offered",
                default_size: size(px(620.), px(620.)),
                build: |window, cx| {
                    let stores = account_stores(cx, None);
                    stores.account.update(cx, |s, cx| {
                        s.set_subscription_for_test(
                            Loadable::Failed {
                                error: AppError::Network {
                                    message: "connection reset by peer".into(),
                                },
                                prior: None,
                            },
                            cx,
                        )
                    });
                    let view = cx.new(|cx| SettingsView::new(stores, window, cx));
                    view.update(cx, |v, cx| v.select(SettingsPane::Account, cx));
                    root(view, window, cx)
                },
            },
            Scene {
                name: "settings_account_subscription_stale",
                description: "Settings ▸ Account: a re-read failed over a known answer — the subscription stays on screen with a quiet check-again",
                default_size: size(px(620.), px(620.)),
                build: |window, cx| {
                    let stores = account_stores(cx, None);
                    stores.account.update(cx, |s, cx| {
                        s.set_subscription_for_test(
                            Loadable::Failed {
                                error: AppError::Network {
                                    message: "connection reset by peer".into(),
                                },
                                prior: Some(subscription(SubscriptionState::Active)),
                            },
                            cx,
                        )
                    });
                    let view = cx.new(|cx| SettingsView::new(stores, window, cx));
                    view.update(cx, |v, cx| v.select(SettingsPane::Account, cx));
                    root(view, window, cx)
                },
            },
            Scene {
                name: "updates_available",
                description: "Updates window: a verified release is available",
                default_size: size(px(480.), px(360.)),
                build: |window, cx| {
                    let stores = stub_stores(cx, |s| {
                        s.update_check = Some(UpdateCheckSnapshot {
                        checked_at_ms: eidola_app_core::now_ms() - 23 * 60 * 1000,
                        result: UpdateCheckResult::UpdateAvailable {
                            release: VerifiedRelease {
                                version: "0.2.0".into(),
                                tag: "v0.2.0".into(),
                                release_url: Some(
                                    "https://github.com/eidola-ai/eidola/releases/tag/v0.2.0"
                                        .into(),
                                ),
                                published_at: Some("2026-06-01T12:00:00Z".into()),
                                ci_identity: "https://github.com/eidola-ai/eidola/.github/workflows/tinfoil-build.yml@refs/tags/v0.2.0".into(),
                                rekor_log_index: 168_338_903,
                                manifest_sha256: "ab".repeat(32),
                                claims_accepted: false,
                            },
                        },
                    });
                    });
                    let view = cx.new(|cx| UpdatesView::new(stores, window, cx));
                    root(view, window, cx)
                },
            },
            Scene {
                name: "record",
                description: "Record window (seeded attestation rows, live section strip)",
                default_size: size(px(860.), px(640.)),
                build: |window, cx| {
                    let stores = stub_stores(cx, |_| {});
                    let view = cx.new(|cx| RecordView::new(stores, window, cx));
                    // Seed a full listing so the section scrolls (the stub has no
                    // backend, so the fetch is a no-op) — lets the scroll
                    // indicator be exercised in the driver.
                    view.update(cx, |v, _| {
                        let rows: Vec<AttestationInfo> = (0..40)
                            .map(|i| AttestationInfo {
                                hash: format!("{i:064x}"),
                                pcr_digest: None,
                                created_at: 1_700_000_000_000 - (i as i64) * 3_600_000,
                                doc_bytes: 4096 + i as i64,
                                connection_count: 1 + (i as i64 % 3),
                            })
                            .collect();
                        v.set_attestations_for_test(rows, true);
                    });
                    root(view, window, cx)
                },
            },
        ]
    }

    fn stub_stores(cx: &mut App, setup: impl FnOnce(&mut StoresStub)) -> Stores {
        let mut fixture = StoresStub::default();
        setup(&mut fixture);
        Stores::stub_with(fixture, cx)
    }

    /// `ready_stores` plus this space's own settings row and a Library title,
    /// so the inspector renders real values rather than placeholders.
    fn inspector_stores(cx: &mut App, router: Option<&str>) -> Stores {
        let router = router.map(str::to_string);
        stub_stores(cx, |s| {
            s.config_state = Some(config_state(true));
            s.eidola_trust = Some(eidola_trust());
            s.models = models();
            s.backends = backends();
            s.local_models = Some(local_models_state());
            s.participants = Some(participants_fixture());
            s.templates = templates_fixture();
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
                    router_model: router,
                    ..Default::default()
                },
            ));
        })
    }

    /// A funded, ready account with a populated model list (so the ⌥ model
    /// reveal and picker are live), the backend registry (so the Backends
    /// settings pane and the picker's per-backend groups render), and the
    /// engine fixtures.
    fn ready_stores(cx: &mut App) -> Stores {
        stub_stores(cx, |s| {
            s.config_state = Some(config_state(true));
            s.eidola_trust = Some(eidola_trust());
            s.models = models();
            s.backends = backends();
            s.local_models = Some(local_models_state());
            s.participants = Some(participants_fixture());
            s.templates = templates_fixture();
            s.agents = Some(agents_fixture());
        })
    }

    /// The shared agent library (Settings → Agents): two promoted agents, each
    /// with the notebook a promotion creates.
    fn agents_fixture() -> Vec<eidola_app_core::GlobalAgentInfo> {
        vec![
            eidola_app_core::GlobalAgentInfo {
                id: "agent-ada".into(),
                label: "Ada".into(),
                model_ref: Some("gemma4-31b".into()),
                system_prompt: Some(
                    "Answer plainly, cite what you rely on, and say when you are unsure.".into(),
                ),
                notify_policy: "human".into(),
                notebook_space_id: Some("nb-ada".into()),
            },
            eidola_app_core::GlobalAgentInfo {
                id: "agent-critic".into(),
                label: "Critic".into(),
                model_ref: Some("qwen3-8b@my-vllm".into()),
                system_prompt: Some("Look for the weakest step in an argument.".into()),
                notify_policy: "explicit".into(),
                notebook_space_id: Some("nb-critic".into()),
            },
        ]
    }

    /// Participants fixture for the Participants view: the referenced global
    /// "You" (with its base/override detail so the fork renders) plus two owned
    /// agents on different backends and notify policies.
    fn participants_fixture() -> (String, Vec<ParticipantInfo>) {
        let you = ParticipantInfo {
            id: eidola_app_core::HUMAN_PARTICIPANT_ID.into(),
            scope: "global".into(),
            source: "referenced".into(),
            kind: "human".into(),
            // The **wire** label (task 64); every surface shows "You".
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
        let assistant = ParticipantInfo {
            id: "agent-assistant".into(),
            scope: "space".into(),
            source: "owned".into(),
            kind: "agent".into(),
            label: "Assistant".into(),
            model_ref: Some("gemma4-31b".into()),
            system_prompt: Some("Be concise and cite sources.".into()),
            notify_policy: "human".into(),
            role: "member".into(),
            reference: None,
        };
        let critic = ParticipantInfo {
            id: "agent-critic".into(),
            scope: "space".into(),
            source: "owned".into(),
            kind: "agent".into(),
            label: "Critic".into(),
            model_ref: Some("qwen3-8b@my-vllm".into()),
            system_prompt: Some("Challenge assumptions and find weak points.".into()),
            notify_policy: "all".into(),
            role: "member".into(),
            reference: None,
        };
        ("demo".into(), vec![you, assistant, critic])
    }

    /// Space-template fixture for the Templates settings pane: the built-in
    /// Default plus a multi-agent saved template.
    fn templates_fixture() -> Vec<SpaceTemplateInfo> {
        vec![
            SpaceTemplateInfo {
                id: eidola_app_core::DEFAULT_TEMPLATE_ID.into(),
                title: "Default".into(),
                cascade_limit: 4,
                router_model: None,
                participants: vec![TemplateParticipantInfo {
                    id: "t-default-1".into(),
                    label: "Assistant".into(),
                    model_ref: Some("gemma4-31b".into()),
                    system_prompt: None,
                    notify_policy: "human".into(),
                }],
                referenced: Vec::new(),
            },
            SpaceTemplateInfo {
                id: "tmpl-research".into(),
                title: "Research panel".into(),
                cascade_limit: 6,
                router_model: None,
                participants: vec![
                    TemplateParticipantInfo {
                        id: "t-research-1".into(),
                        label: "Analyst".into(),
                        model_ref: Some("gemma4-31b".into()),
                        system_prompt: Some("Lay out the evidence.".into()),
                        notify_policy: "all".into(),
                    },
                    TemplateParticipantInfo {
                        id: "t-research-2".into(),
                        label: "Skeptic".into(),
                        model_ref: Some("qwen3-8b@my-vllm".into()),
                        system_prompt: Some("Push back hard.".into()),
                        notify_policy: "human".into(),
                    },
                ],
                // Saved from a space, so it carries the shared "You" by
                // reference — listed read-only in the editor.
                referenced: vec![eidola_app_core::TemplateReferencedParticipant {
                    id: eidola_app_core::HUMAN_PARTICIPANT_ID.into(),
                    kind: "human".into(),
                    label: "User".into(),
                    model_ref: None,
                    system_prompt: Some("Keep me honest and ask before assuming.".into()),
                    notify_policy: "explicit".into(),
                }],
            },
        ]
    }

    /// Backend-registry fixture: the two singletons plus one external
    /// OpenAI-compatible server, mirroring a configured multi-backend setup.
    fn backends() -> Vec<eidola_app_core::BackendInfo> {
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
                id: "my-vllm".into(),
                kind: BackendKind::OpenAi,
                display_name: "My vLLM box".into(),
                enabled: true,
                base_url: Some("http://192.168.1.20:8000".into()),
                has_api_key: true,
                models_dir: None,
                model_overrides: Some(vec!["qwen3-8b".into()]),
                engine_path: None,
                auto_start: true,
                created_at: 1,
            },
        ]
    }

    /// Local-inference fixture: one model loaded and serving, one merely
    /// downloaded, one mid-download — every state the Models pane renders,
    /// with the loaded one surfacing in the space view's model picker.
    fn local_models_state() -> eidola_app_core::LocalModelsState {
        use eidola_app_core::{LocalModelInfo, LocalModelStatus, LocalModelsState};
        LocalModelsState {
            engine_path: Some("/opt/homebrew/bin/llama-server".into()),
            external: Vec::new(),
            models: vec![
                LocalModelInfo {
                    id: "gemma-4-12b-it-qat-q4_0@local".into(),
                    slug: "gemma-4-12b-it-qat-q4_0".into(),
                    display_name: "Gemma 4 12B".into(),
                    file_name: "gemma-4-12b-it-qat-q4_0.gguf".into(),
                    size_bytes: Some(6_975_877_728),
                    source_url: None,
                    status: LocalModelStatus::Downloading {
                        received: 3_100_000_000,
                        total: Some(6_975_877_728),
                    },
                    last_error: None,
                    on_disk: false,
                },
                LocalModelInfo {
                    id: "gemma-4-E2B_q4_0-it@local".into(),
                    slug: "gemma-4-E2B_q4_0-it".into(),
                    display_name: "Gemma 4 E2B".into(),
                    file_name: "gemma-4-E2B_q4_0-it.gguf".into(),
                    size_bytes: Some(3_349_514_112),
                    source_url: None,
                    status: LocalModelStatus::Loaded {
                        port: 51_432,
                        context_tokens: 8192,
                        pinned: false,
                    },
                    last_error: None,
                    on_disk: true,
                },
                LocalModelInfo {
                    id: "gemma-4-E4B_q4_0-it@local".into(),
                    slug: "gemma-4-E4B_q4_0-it".into(),
                    display_name: "Gemma 4 E4B".into(),
                    file_name: "gemma-4-E4B_q4_0-it.gguf".into(),
                    size_bytes: Some(5_154_939_136),
                    source_url: None,
                    status: LocalModelStatus::Available,
                    last_error: None,
                    on_disk: true,
                },
            ],
        }
    }

    /// [`local_models_state`] plus the row a **failed download** leaves: no
    /// file, no size, an error, and the URL a retry re-runs.
    fn local_models_with_failed_download() -> eidola_app_core::LocalModelsState {
        use eidola_app_core::{LocalModelInfo, LocalModelStatus};
        let mut state = local_models_state();
        state.models.insert(
            0,
            LocalModelInfo {
                id: "gemma-4-27b-it-qat-q4_0@local".into(),
                slug: "gemma-4-27b-it-qat-q4_0".into(),
                display_name: "Gemma 4 27B".into(),
                file_name: "gemma-4-27b-it-qat-q4_0.gguf".into(),
                size_bytes: None,
                source_url: Some(
                    "https://huggingface.co/google/gemma-4-27b-it-qat-q4_0-gguf/resolve/main/\
                     gemma-4-27b-it-qat-q4_0.gguf"
                        .into(),
                ),
                status: LocalModelStatus::Available,
                last_error: Some("download failed: connection reset by peer".into()),
                on_disk: false,
            },
        );
        state
    }

    fn config_state(has_account: bool) -> ConfigState {
        ConfigState {
            default_template: "00000000-0000-7000-8000-000000000010".into(),
            has_account,
            has_account_secret: has_account,
            account_id: has_account.then(|| "00000000-0000-7000-8000-000000000111".into()),
            account_secret: has_account.then(|| "driver-account-secret".into()),
            domain_separator: "ACT-v1:eidola:inference:production:2026-03-05".into(),
            appearance: eidola_app_core::config::AppearanceSetting::System,
            time_of_day_tint: eidola_app_core::config::TimeOfDayTint::On,
            light_character: eidola_app_core::config::LightCharacter::Neutral,
            font_scale: 1.0,
            language: None,
        }
    }

    fn eidola_trust() -> eidola_app_core::EidolaTrust {
        eidola_app_core::EidolaTrust {
            base_url: "https://eidola.example/v1".into(),
            base_url_pin: "https://eidola.example/v1".into(),
            base_url_is_override: false,
            trusted_measurements: Vec::new(),
            trusted_measurements_are_override: false,
            pinned_measurement: eidola_app_core::MeasurementInfo {
                snp: "1122334455667788112233445566778811223344556677881122334455667788".into(),
                tdx_rtmr1: "aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899"
                    .into(),
                tdx_rtmr2: "99887766554433221100ffeeddccbbaa99887766554433221100ffeeddccbbaa"
                    .into(),
            },
            has_hardware_root_ca: false,
            hardware_root_ca_pem: None,
            has_hardware_intermediate_ca: false,
            hardware_intermediate_ca_pem: None,
        }
    }

    fn models() -> Vec<ModelInfo> {
        vec![
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
            // Enough further models that the picker dropdown overflows its
            // max-height and scrolls independently of the roster body — the QA
            // fixture for the picker's own scroll indicator.
            ModelInfo {
                id: "llama4-scout".into(),
                context_length: 131_072,
                prompt_credits_per_token: 0.8,
                completion_credits_per_token: 2.4,
                request_credits: None,
            },
            ModelInfo {
                id: "deepseek-v3".into(),
                context_length: 131_072,
                prompt_credits_per_token: 1.2,
                completion_credits_per_token: 3.0,
                request_credits: None,
            },
            ModelInfo {
                id: "mistral-large-3".into(),
                context_length: 131_072,
                prompt_credits_per_token: 1.5,
                completion_credits_per_token: 4.5,
                request_credits: None,
            },
            ModelInfo {
                id: "gpt-oss-120b".into(),
                context_length: 131_072,
                prompt_credits_per_token: 1.1,
                completion_credits_per_token: 3.3,
                request_credits: None,
            },
            ModelInfo {
                id: "phi5-moe".into(),
                context_length: 131_072,
                prompt_credits_per_token: 0.4,
                completion_credits_per_token: 1.2,
                request_credits: None,
            },
            ModelInfo {
                id: "command-r-plus-2".into(),
                context_length: 131_072,
                prompt_credits_per_token: 1.3,
                completion_credits_per_token: 3.9,
                request_credits: None,
            },
            ModelInfo {
                id: "yi-large-2".into(),
                context_length: 131_072,
                prompt_credits_per_token: 0.9,
                completion_credits_per_token: 2.7,
                request_credits: None,
            },
            ModelInfo {
                id: "glm-5-air".into(),
                context_length: 131_072,
                prompt_credits_per_token: 0.6,
                completion_credits_per_token: 1.8,
                request_credits: None,
            },
        ]
    }

    fn prices() -> Vec<PriceInfo> {
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
            // A one-time price, so the plans surfaces show both kinds — it is
            // what stays offered once a subscription is in force.
            PriceInfo {
                id: "price_topup".into(),
                product_name: "Top-up".into(),
                product_description: Some("Credit that keeps for a year".into()),
                amount_display: "10.00 USD".into(),
                recurrence: "".into(),
                credits: 10_000_000,
            },
        ]
    }

    /// A linear conversation whose two assistant replies each carry a persisted
    /// `thinking` content block — what a *reopened* space looks like now that
    /// reasoning is durable (it used to come back with no disclosure at all).
    fn thinking_posts() -> Vec<eidola_app_core::PostNode> {
        let mut posts = Vec::new();
        let mut push = |action_id: &str,
                        kind: &str,
                        label: &str,
                        atype: &str,
                        parent: Option<&str>,
                        thinking: Option<&str>,
                        text: &str| {
            let mut n = fixtures::fixture_post(action_id, kind, label, atype, text, 0, false, 1);
            n.parent_action_id = parent.map(String::from);
            n.relation = parent.map(|_| "reply".to_string());
            if let Some(t) = thinking {
                n.blocks.insert(
                    0,
                    eidola_app_core::PostBlock {
                        id: format!("cb-think-{action_id}"),
                        block_type: "thinking".into(),
                        text: Some(t.into()),
                        tool_name: None,
                        tool_call_id: None,
                        data: None,
                    },
                );
            }
            posts.push(n);
        };
        push(
            "a1",
            "human",
            "user",
            "user_input",
            None,
            None,
            "Why is the sky blue?",
        );
        push(
            "a2",
            "agent",
            "kimi-k2",
            "inference",
            Some("a1"),
            Some(
                "The user wants the physical mechanism, not a metaphor. Rayleigh scattering, \
                 then the 1/λ⁴ dependence, then why we don't see violet.",
            ),
            "Sunlight is a fairly even mix across the visible spectrum. As it crosses the \
             atmosphere it meets molecules far smaller than its wavelength, and those \
             scatter short (blue) wavelengths far more strongly than long (red) ones.",
        );
        push(
            "a3",
            "human",
            "user",
            "user_input",
            Some("a2"),
            None,
            "And at sunset?",
        );
        push(
            "a4",
            "agent",
            "kimi-k2",
            "inference",
            Some("a3"),
            Some("Same mechanism, longer path."),
            "Near sunset the light skims a long, slanted path through the air, the blue is \
             scattered away entirely, and what survives to reach you is the warm red-orange \
             of a low sun.",
        );
        posts
    }

    fn conversation() -> Vec<SpaceMessage> {
        vec![
            SpaceMessage {
                role: "user".into(),
                content: "Why is the sky blue?".into(),
            },
            SpaceMessage {
                role: "assistant".into(),
                content: "Sunlight is a fairly even mix across the visible spectrum. As it \
                      crosses the atmosphere it meets molecules far smaller than its \
                      wavelength, and those scatter short (blue) wavelengths far more \
                      strongly than long (red) ones."
                    .into(),
            },
            SpaceMessage {
                role: "user".into(),
                content: "And at sunset?".into(),
            },
            SpaceMessage {
                role: "assistant".into(),
                content: "Near sunset the light skims a long, slanted path through the air, \
                      the blue is scattered away entirely, and what survives to reach you \
                      is the warm red-orange of a low sun."
                    .into(),
            },
        ]
    }

    fn table_conversation() -> Vec<SpaceMessage> {
        vec![
            SpaceMessage {
                role: "user".into(),
                content: "Compare the local models I can run.".into(),
            },
            SpaceMessage {
                role: "assistant".into(),
                content: "Here is a comparison of the curated local models:\n\n\
                    | Model | Params | Context | Disk |\n\
                    | :-- | --: | --: | --: |\n\
                    | Gemma 4 E2B | 2B | 32k | 1.6 GB |\n\
                    | Gemma 4 4B | 4B | 128k | 3.2 GB |\n\
                    | Gemma 4 12B | 12B | 128k | 8.1 GB |\n\
                    | Gemma 4 27B | 27B | 128k | 17 GB |\n\n\
                    Styling composes inside cells:\n\n\
                    | Feature | Status |\n\
                    | :-- | :-- |\n\
                    | **Bold** and `inline code` | ~~cut~~ kept |\n\
                    | a\\|b literal pipe | plain text |\n\n\
                    And a deliberately wide table whose columns shrink and wrap \
                    within the reading column:\n\n\
                    | A rather long header cell one | Header two with more words | Third header column | Fourth column header | Fifth and final header |\n\
                    | --- | --- | --- | --- | --- |\n\
                    | some content here | more cell content | further content | yet more words | the last cell |\n\n\
                    Everything above should read as a quiet, hairline-ruled book table."
                    .into(),
            },
        ]
    }

    fn long_conversation() -> Vec<SpaceMessage> {
        // One short question and one enormous multi-paragraph answer, far taller
        // than any test window — the readonly-selection repro fixture.
        let mut body = String::new();
        body.push_str("# Rayleigh scattering, in depth\n\n");
        for i in 1..=20 {
            body.push_str(&format!(
                "Paragraph {i}. Sunlight is a fairly even mix across the visible spectrum, \
                 and as it crosses the atmosphere it meets molecules far smaller than its \
                 wavelength; those scatter short blue wavelengths far more strongly than \
                 long red ones, which is the whole story of why the daytime sky is blue.\n\n"
            ));
            if i == 5 {
                body.push_str(
                    "```python\ndef rayleigh(wavelength):\n    # intensity scales as 1/lambda^4\n    return 1.0 / wavelength ** 4\n```\n\n",
                );
            }
            if i == 10 {
                body.push_str("- short wavelengths scatter more\n- long wavelengths pass through\n- the eye integrates the result\n\n");
            }
        }
        let body = body.trim_end().to_string();
        // The long reply sits *mid-conversation* (not the leaf), so it renders
        // nested inside single-child `render_strip` horizontal scrollers — the
        // real structure a multi-turn conversation produces.
        vec![
            SpaceMessage {
                role: "user".into(),
                content: "Explain the sky at length.".into(),
            },
            SpaceMessage {
                role: "assistant".into(),
                content: body,
            },
            SpaceMessage {
                role: "user".into(),
                content: "Thanks — and what about at sunset?".into(),
            },
            SpaceMessage {
                role: "assistant".into(),
                content: "Near sunset the light skims a long, slanted path through the air, \
                          the blue is scattered away entirely, and what survives to reach you \
                          is the warm red-orange of a low sun."
                    .into(),
            },
        ]
    }

    /// Placeholder prose of roughly `n` bytes (word-salad, never real user
    /// text) — used by [`structured_posts`] to mirror a real post's *markdown
    /// structure* without its content.
    fn lorem(n: usize) -> String {
        const WORDS: &[&str] = &[
            "lorem",
            "ipsum",
            "dolor",
            "sit",
            "amet",
            "consetetur",
            "sadipscing",
            "elitr",
            "sed",
            "diam",
            "nonumy",
            "eirmod",
            "tempor",
            "invidunt",
            "ut",
            "labore",
            "et",
            "dolore",
            "magna",
            "aliquyam",
            "erat",
        ];
        let mut out = String::new();
        let mut i = 0;
        while out.len() < n {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(WORDS[i % WORDS.len()]);
            i += 1;
        }
        out.truncate(n);

        out.trim_end().to_string()
    }

    /// A `PostNode` tree mirroring the *structure* of a real user-reported
    /// space where in-view selection failed on certain posts: a branched root
    /// (two replies — spine + branch) whose spine reply is markdown-heavy
    /// (headings, rules, nested lists, a long blockquote, bold + inline code,
    /// and a trailing table). All text is placeholder.
    fn structured_posts() -> Vec<eidola_app_core::PostNode> {
        let post = |action_id: &str,
                    kind: &str,
                    label: &str,
                    atype: &str,
                    depth: usize,
                    is_branch: bool,
                    parent: Option<&str>,
                    content: String,
                    at: i64|
         -> eidola_app_core::PostNode {
            let mut n = fixtures::fixture_post(
                action_id, kind, label, atype, &content, depth, is_branch, 1,
            );
            n.parent_action_id = parent.map(String::from);
            n.relation = parent.map(|_| "reply".to_string());
            n.created_at = at;
            n
        };

        // The markdown-heavy spine reply (line-structure mirror of the real
        // failing post: paragraphs, ---, ##/###, tight heading→para, bulleted
        // lists with bold lead-ins + inline code, a para immediately followed
        // by a long quote, nested ordered list, and a 3-column table).
        let mut big = String::new();
        big.push_str(&format!("{}\n\n", lorem(319)));
        big.push_str(&format!(
            "{} **{}** {}\n\n",
            lorem(120),
            lorem(20),
            lorem(158)
        ));
        big.push_str(&format!(
            "{}\n\n---\n\n## {}\n\n{}\n\n",
            lorem(229),
            lorem(52),
            lorem(123)
        ));
        big.push_str(&format!("### {}\n{}\n\n", lorem(48), lorem(150)));
        for len in [140usize, 160, 220, 90] {
            big.push_str(&format!(
                "- **{}:** {} `{}` {}\n",
                lorem(14),
                lorem(len / 2),
                lorem(8),
                lorem(len / 2)
            ));
        }
        big.push('\n');
        big.push_str(&format!(
            "**{}:**\n> {} `{}` {} `{}` {}\n\n",
            lorem(8),
            lorem(160),
            lorem(10),
            lorem(140),
            lorem(9),
            lorem(100)
        ));
        big.push_str(&format!("### {}\n{}\n\n", lorem(56), lorem(115)));
        big.push_str(&format!("- **{}:** {}\n", lorem(10), lorem(60)));
        for len in [55usize, 90, 85] {
            big.push_str(&format!("    1. **{}:** {}\n", lorem(8), lorem(len)));
        }
        big.push_str(&format!("- **{}:** {}\n", lorem(10), lorem(155)));
        big.push_str(&format!(
            "    - {}\n    - **{}:** {}\n\n",
            lorem(170),
            lorem(9),
            lorem(95)
        ));
        big.push_str(&format!(
            "### {}\n{} **{}** {} **{}** {}\n\n---\n\n",
            lorem(20),
            lorem(90),
            lorem(15),
            lorem(90),
            lorem(14),
            lorem(80)
        ));
        big.push_str(&format!("## {}\n\n{}\n\n", lorem(48), lorem(115)));
        for (h, p, l1, l2) in [
            (64usize, 131usize, 270usize, 225usize),
            (67, 128, 230, 265),
            (41, 154, 325, 0),
        ] {
            big.push_str(&format!("### {}\n{}\n\n", lorem(h), lorem(p)));
            big.push_str(&format!(
                "- **{}:** {} `{}` {}\n",
                lorem(12),
                lorem(l1 / 2),
                lorem(6),
                lorem(l1 / 2)
            ));
            if l2 > 0 {
                big.push_str(&format!("- **{}:** {}\n", lorem(12), lorem(l2)));
            }
            big.push('\n');
        }
        big.push_str(&format!("## {}\n\n", lorem(12)));
        big.push_str(&format!(
            "| {} | {} | {} |\n| --- | --- | --- |\n",
            lorem(8),
            lorem(12),
            lorem(14)
        ));
        for _ in 0..5 {
            big.push_str(&format!(
                "| **{}** | {} | {} |\n",
                lorem(12),
                lorem(48),
                lorem(52)
            ));
        }
        let big = big.trim_end().to_string();

        vec![
            // Root: a three-line, two-paragraph user question.
            post(
                "s1",
                "human",
                "You",
                "user_input",
                0,
                false,
                None,
                format!("{}\n\n{}", lorem(212), lorem(205)),
                1,
            ),
            // Spine: the markdown-heavy reply, then a short follow-up exchange.
            post(
                "s2",
                "agent",
                "gemma",
                "inference",
                0,
                false,
                Some("s1"),
                big,
                2,
            ),
            post(
                "s3",
                "human",
                "You",
                "user_input",
                0,
                false,
                Some("s2"),
                lorem(61),
                3,
            ),
            post(
                "s4",
                "agent",
                "gemma",
                "inference",
                0,
                false,
                Some("s3"),
                format!(
                    "{}\n\n1. **{}:** {}\n2. **{}:** {}\n3. **{}:** {}\n4. **{}:** {}\n5. **{}:** {}",
                    lorem(48),
                    lorem(10),
                    lorem(165),
                    lorem(10),
                    lorem(200),
                    lorem(10),
                    lorem(195),
                    lorem(10),
                    lorem(195),
                    lorem(10),
                    lorem(215),
                ),
                4,
            ),
            // Branch off the root: a tiny aside exchange.
            post(
                "s5",
                "human",
                "You",
                "user_input",
                1,
                true,
                Some("s1"),
                lorem(18),
                5,
            ),
            post(
                "s6",
                "agent",
                "gemma",
                "inference",
                1,
                false,
                Some("s5"),
                lorem(77),
                6,
            ),
        ]
    }

    fn long_metadata_posts() -> Vec<eidola_app_core::PostNode> {
        let mut posts = fixtures::kitchen_sink_posts();
        let post = posts
            .iter_mut()
            .find(|post| post.action_id == "a2")
            .expect("the fixture includes its first assistant reply");
        let selection = "A deliberately descriptive participant identity that remains recognizable@private-inference-workstation-in-the-west-studio";
        post.participant.label = selection.into();
        post.model = Some(selection.into());
        posts
    }

    fn library_spaces() -> Vec<SpaceInfo> {
        fn space(id: &str, title: Option<&str>, snippet: Option<&str>, days_ago: i64) -> SpaceInfo {
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
        vec![
            space("s1", Some("Tides and the moon"), None, 0),
            space(
                "s2",
                Some("Borrow checker, closures, and lifetimes"),
                None,
                1,
            ),
            space("s3", None, Some("what is a monad, really?"), 3),
            space("s4", Some("Reading list for distributed systems"), None, 12),
            space("s5", Some("Why is the sky blue?"), None, 30),
            space("s6", None, None, 400),
        ]
    }

    fn settings_stores(cx: &mut App) -> Stores {
        stub_stores(cx, |s| {
            s.config_state = Some(config_state(true));
            s.eidola_trust = Some(eidola_trust());
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
            s.prices = prices();
            s.backends = backends();
            s.local_models = Some(local_models_state());
        })
    }

    /// `settings_stores` with the subscription cell set explicitly — the
    /// Account pane's four subscription states are exactly what varies
    /// across the scenes below.
    fn account_stores(cx: &mut App, subscription: Option<SubscriptionInfo>) -> Stores {
        stub_stores(cx, |s| {
            s.config_state = Some(config_state(true));
            s.eidola_trust = Some(eidola_trust());
            s.balances = Some(BalancesResult {
                available: 4_200_000,
                pools: vec![BalancePoolInfo {
                    amount: 4_200_000,
                    source: "subscription".into(),
                    expires_at: Some(eidola_app_core::now_ms() + 23 * 24 * 60 * 60 * 1000),
                }],
            });
            s.prices = prices();
            s.backends = backends();
            s.local_models = Some(local_models_state());
            s.subscription = subscription;
        })
    }

    fn subscription(state: SubscriptionState) -> SubscriptionInfo {
        SubscriptionInfo {
            state,
            status: (state == SubscriptionState::Active).then(|| "active".to_string()),
            current_period_end: (state == SubscriptionState::Active)
                .then(|| eidola_app_core::now_ms() + 23 * 24 * 60 * 60 * 1000),
        }
    }

    // ---------------------------------------------------------------------------
    // Session
    // ---------------------------------------------------------------------------

    struct Session {
        windows: HashMap<u64, OpenWindow>,
        shot_dir: PathBuf,
        shot_counter: u32,
        quit: bool,
    }

    struct OpenWindow {
        handle: AnyWindowHandle,
        scene: String,
    }

    impl Session {
        fn new() -> Self {
            let shot_dir =
                std::env::temp_dir().join(format!("eidola-driver-{}", std::process::id()));
            Self {
                windows: HashMap::new(),
                shot_dir,
                shot_counter: 0,
                quit: false,
            }
        }

        fn handle_line(&mut self, cx: &mut VisualTestAppContext, line: &str) -> Value {
            let cmd: Cmd = match serde_json::from_str(line) {
                Ok(cmd) => cmd,
                Err(e) => return json!({"ok": false, "error": format!("bad command: {e}")}),
            };
            match self.handle(cx, cmd) {
                Ok(data) => {
                    let mut resp = json!({"ok": true});
                    if let Value::Object(extra) = data {
                        resp.as_object_mut().unwrap().extend(extra);
                    }
                    resp
                }
                Err(e) => json!({"ok": false, "error": e}),
            }
        }

        fn window(&self, id: u64) -> Result<AnyWindowHandle, String> {
            self.windows
                .get(&id)
                .map(|w| w.handle)
                .ok_or_else(|| format!("no open window {id} (see {{\"cmd\":\"windows\"}})"))
        }

        /// Resolve a click/scroll position: an explicit x/y wins, otherwise the
        /// center of the named probe's last-painted bounds.
        fn position(
            &self,
            cx: &mut VisualTestAppContext,
            window: u64,
            target: Option<&str>,
            x: Option<f32>,
            y: Option<f32>,
        ) -> Result<gpui::Point<Pixels>, String> {
            if let (Some(x), Some(y)) = (x, y) {
                return Ok(point(px(x), px(y)));
            }
            let Some(target) = target else {
                return Err("provide either \"target\" or both \"x\" and \"y\"".into());
            };
            // Refresh the registry so the bounds are from the current frame.
            let entries = self.fresh_elements(cx, window)?;
            entries
                .iter()
                .find(|(name, _)| name == target)
                .map(|(_, entry)| entry.bounds.center())
                .ok_or_else(|| {
                    let known: Vec<&str> = entries.iter().map(|(n, _)| n.as_str()).collect();
                    format!("no element \"{target}\"; known: {}", known.join(", "))
                })
        }

        /// Clear the window's probe entries, force a fresh frame, and return the
        /// re-recorded entries — so unmounted elements never linger as targets.
        fn fresh_elements(
            &self,
            cx: &mut VisualTestAppContext,
            window: u64,
        ) -> Result<Vec<(String, probe::ProbeEntry)>, String> {
            let handle = self.window(window)?;
            let id = handle.window_id().as_u64();
            probe::clear_window(id);
            cx.update_window(handle, |_, window, _| window.refresh())
                .map_err(|e| format!("window update failed: {e}"))?;
            cx.run_until_parked();
            Ok(probe::window_entries(id))
        }

        fn handle(&mut self, cx: &mut VisualTestAppContext, cmd: Cmd) -> Result<Value, String> {
            match cmd {
                Cmd::Scenes => Ok(json!({"scenes": scene_catalog()})),

                Cmd::Open {
                    scene,
                    width,
                    height,
                } => {
                    let def = scenes()
                        .into_iter()
                        .find(|s| s.name == scene)
                        .ok_or_else(|| {
                            let known: Vec<String> =
                                scenes().iter().map(|s| s.name.to_string()).collect();
                            format!("unknown scene \"{scene}\"; known: {}", known.join(", "))
                        })?;
                    let mut sz = def.default_size;
                    if let Some(w) = width {
                        sz.width = px(w);
                    }
                    if let Some(h) = height {
                        sz.height = px(h);
                    }
                    let handle = cx
                        .open_offscreen_window(sz, def.build)
                        .map_err(|e| format!("open failed: {e}"))?;
                    cx.run_until_parked();
                    let handle: AnyWindowHandle = handle.into();
                    let id = handle.window_id().as_u64();
                    self.windows.insert(
                        id,
                        OpenWindow {
                            handle,
                            scene: scene.clone(),
                        },
                    );
                    Ok(json!({
                        "window": id,
                        "scene": scene,
                        "width": sz.width.as_f32(),
                        "height": sz.height.as_f32(),
                    }))
                }

                Cmd::Windows => {
                    let mut list: Vec<Value> = Vec::new();
                    for (id, w) in &self.windows {
                        let sz = cx
                            .update_window(w.handle, |_, window, _| window.viewport_size())
                            .ok();
                        list.push(json!({
                            "window": id,
                            "scene": w.scene,
                            "width": sz.map(|s| s.width.as_f32()),
                            "height": sz.map(|s| s.height.as_f32()),
                        }));
                    }
                    list.sort_by_key(|v| v["window"].as_u64());
                    Ok(json!({"windows": list}))
                }

                Cmd::Elements { window } => {
                    let entries = self.fresh_elements(cx, window)?;
                    let list: Vec<Value> = entries
                        .into_iter()
                        .map(|(name, e)| {
                            let b = e.bounds;
                            json!({
                                "name": name,
                                "role": format!("{:?}", e.role),
                                "label": e.label.to_string(),
                                // The content channel (`aria_value`), when the
                                // call site sets one — a post's text, a
                                // balance, an alert's message. `null` otherwise.
                                "value": e.value.as_ref().map(|v| v.to_string()),
                                "x": b.origin.x.as_f32(),
                                "y": b.origin.y.as_f32(),
                                "width": b.size.width.as_f32(),
                                "height": b.size.height.as_f32(),
                                "center": {
                                    "x": b.center().x.as_f32(),
                                    "y": b.center().y.as_f32(),
                                },
                            })
                        })
                        .collect();
                    Ok(json!({"elements": list}))
                }

                Cmd::Click {
                    window,
                    target,
                    x,
                    y,
                    alt,
                    command,
                    shift,
                    button,
                } => {
                    let pos = self.position(cx, window, target.as_deref(), x, y)?;
                    let handle = self.window(window)?;
                    let modifiers = Modifiers {
                        alt,
                        platform: command,
                        shift,
                        ..Default::default()
                    };
                    if button.as_deref() == Some("right") {
                        // The context-menu gesture is a right *press* — that
                        // is what the editor listens for.
                        cx.simulate_event(
                            handle,
                            MouseDownEvent {
                                button: MouseButton::Right,
                                position: pos,
                                modifiers,
                                click_count: 1,
                                first_mouse: false,
                            },
                        );
                    } else {
                        cx.simulate_click(handle, pos, modifiers);
                    }
                    Ok(json!({"clicked": {"x": pos.x.as_f32(), "y": pos.y.as_f32()}}))
                }

                Cmd::Drag {
                    window,
                    from_x,
                    from_y,
                    to_x,
                    to_y,
                    click_count,
                    hold,
                } => {
                    let handle = self.window(window)?;
                    let from = point(px(from_x), px(from_y));
                    let to = point(px(to_x), px(to_y));
                    cx.simulate_event(
                        handle,
                        MouseDownEvent {
                            button: MouseButton::Left,
                            position: from,
                            modifiers: Modifiers::default(),
                            click_count,
                            first_mouse: false,
                        },
                    );
                    // A few interpolated moves so the drag crosses intervening
                    // rows the way a real pointer would.
                    for step in 1..=4 {
                        let t = step as f32 / 4.0;
                        let pos = point(
                            px(from_x + (to_x - from_x) * t),
                            px(from_y + (to_y - from_y) * t),
                        );
                        cx.simulate_event(
                            handle,
                            MouseMoveEvent {
                                position: pos,
                                pressed_button: Some(MouseButton::Left),
                                modifiers: Modifiers::default(),
                            },
                        );
                    }
                    if hold {
                        // Hold the button at `to` and pump frames so any host
                        // autoscroll-while-selecting loop advances. Re-issuing
                        // the (unchanged) move each iteration forces a render —
                        // the harness doesn't drive the continuous frame loop a
                        // real window would — and keeps the pointer in the edge
                        // margin so autoscroll keeps stepping until it clamps.
                        for _ in 0..120 {
                            cx.simulate_event(
                                handle,
                                MouseMoveEvent {
                                    position: to,
                                    pressed_button: Some(MouseButton::Left),
                                    modifiers: Modifiers::default(),
                                },
                            );
                            cx.run_until_parked();
                        }
                    }
                    cx.simulate_event(
                        handle,
                        MouseUpEvent {
                            button: MouseButton::Left,
                            position: to,
                            modifiers: Modifiers::default(),
                            click_count,
                        },
                    );
                    cx.run_until_parked();
                    Ok(json!({"dragged": {"from": [from_x, from_y], "to": [to_x, to_y]}}))
                }

                Cmd::Type { window, text } => {
                    let handle = self.window(window)?;
                    cx.simulate_input(handle, &text);
                    Ok(json!({}))
                }

                Cmd::Keys { window, keys } => {
                    let handle = self.window(window)?;
                    cx.simulate_keystrokes(handle, &keys);
                    Ok(json!({}))
                }

                Cmd::Modifiers {
                    window,
                    alt,
                    command,
                    shift,
                    ctrl,
                } => {
                    let handle = self.window(window)?;
                    let modifiers = Modifiers {
                        alt,
                        platform: command,
                        shift,
                        control: ctrl,
                        ..Default::default()
                    };
                    cx.simulate_event(
                        handle,
                        ModifiersChangedEvent {
                            modifiers,
                            capslock: Capslock::default(),
                        },
                    );
                    Ok(json!({}))
                }

                Cmd::Scroll {
                    window,
                    target,
                    x,
                    y,
                    dx,
                    dy,
                } => {
                    let pos = self.position(cx, window, target.as_deref(), x, y)?;
                    let handle = self.window(window)?;
                    cx.simulate_event(
                        handle,
                        ScrollWheelEvent {
                            position: pos,
                            delta: ScrollDelta::Pixels(point(px(dx), px(dy))),
                            modifiers: Modifiers::default(),
                            touch_phase: TouchPhase::Moved,
                        },
                    );
                    Ok(json!({}))
                }

                Cmd::Resize {
                    window,
                    width,
                    height,
                } => {
                    let handle = self.window(window)?;
                    cx.update_window(handle, |_, window, _| {
                        window.resize(size(px(width), px(height)))
                    })
                    .map_err(|e| format!("window update failed: {e}"))?;
                    cx.run_until_parked();
                    Ok(json!({}))
                }

                Cmd::Screenshot { window, path } => {
                    let handle = self.window(window)?;
                    cx.run_until_parked();
                    let img = cx
                        .capture_screenshot(handle)
                        .map_err(|e| format!("capture failed: {e}"))?;
                    let path = match path {
                        Some(p) => PathBuf::from(p),
                        None => {
                            std::fs::create_dir_all(&self.shot_dir)
                                .map_err(|e| format!("create {}: {e}", self.shot_dir.display()))?;
                            self.shot_counter += 1;
                            self.shot_dir
                                .join(format!("shot-{:03}.png", self.shot_counter))
                        }
                    };
                    img.save(&path).map_err(|e| format!("save: {e}"))?;
                    Ok(json!({
                        "path": path.display().to_string(),
                        "width": img.width(),
                        "height": img.height(),
                    }))
                }

                Cmd::Theme { mode, character } => {
                    let mode = match mode.as_str() {
                        "day" | "light" => ThemeMode::Light,
                        "night" | "dark" => ThemeMode::Dark,
                        other => return Err(format!("unknown theme mode \"{other}\" (day|night)")),
                    };
                    let character = match character.as_deref() {
                        None | Some("neutral") => eidola_gui::theme::LightCharacter::Neutral,
                        Some("cool") => eidola_gui::theme::LightCharacter::Cool,
                        Some("warm") => eidola_gui::theme::LightCharacter::Warm,
                        Some(other) => {
                            return Err(format!(
                                "unknown theme character \"{other}\" (cool|neutral|warm)"
                            ));
                        }
                    };
                    cx.update(|cx| eidola_gui::theme::apply_fixed(mode, character, cx));
                    for w in self.windows.values() {
                        cx.update_window(w.handle, |_, window, _| window.refresh())
                            .ok();
                    }
                    cx.run_until_parked();
                    Ok(json!({}))
                }

                Cmd::Locale { tag } => {
                    let known: Vec<&str> = eidola_gui::i18n::available_locales().collect();
                    if !known.iter().any(|k| *k == tag) {
                        return Err(format!("unknown locale \"{tag}\" ({})", known.join("|")));
                    }
                    cx.update(|cx| eidola_gui::i18n::apply(&tag, cx));
                    for w in self.windows.values() {
                        cx.update_window(w.handle, |_, window, _| window.refresh())
                            .ok();
                    }
                    cx.run_until_parked();
                    Ok(json!({ "locale": tag }))
                }

                Cmd::Settle { ms } => {
                    cx.advance_clock(Duration::from_millis(ms.unwrap_or(100)));
                    cx.run_until_parked();
                    Ok(json!({}))
                }

                Cmd::Close { window } => {
                    let handle = self.window(window)?;
                    cx.update_window(handle, |_, window, _| window.remove_window())
                        .map_err(|e| format!("window update failed: {e}"))?;
                    cx.run_until_parked();
                    self.windows.remove(&window);
                    Ok(json!({}))
                }

                Cmd::Quit => {
                    self.quit = true;
                    Ok(json!({}))
                }
            }
        }
    }

    fn scene_catalog() -> Vec<Value> {
        scenes()
            .iter()
            .map(|s| {
                json!({
                    "name": s.name,
                    "description": s.description,
                    "width": s.default_size.width.as_f32(),
                    "height": s.default_size.height.as_f32(),
                })
            })
            .collect()
    }

    pub fn run() {
        // Probes feed the element registry; without this the `elements` command
        // would always come back empty.
        probe::set_probes_enabled(true);

        let platform = gpui_platform::current_platform(false);
        let mut cx = VisualTestAppContext::with_asset_source(platform, Arc::new(Assets));
        cx.update(|cx| {
            gpui_component::init(cx);
            eidola_gui::theme::install(cx);
            // The real app's keymap, so simulated keystrokes resolve identically
            // (⌘↩ submit, ⌥⌘M picker, editor motion …). App-global action
            // *handlers* (⌘N, ⌘L …) are not installed — they require the real
            // AppGlobal/backend; window-level actions all work.
            eidola_gui::install_keybindings(cx);
        });

        // Stdin is read on a side thread; commands execute on the main thread
        // (AppKit requirement) as they arrive.
        let (tx, rx) = mpsc::channel::<String>();
        std::thread::spawn(move || {
            let stdin = std::io::stdin();
            for line in stdin.lock().lines() {
                let Ok(line) = line else { break };
                if tx.send(line).is_err() {
                    break;
                }
            }
        });

        let mut session = Session::new();
        let hello = json!({
            "ok": true,
            "hello": "eidola-driver",
            "protocol": 1,
            "scenes": scene_catalog(),
        });
        println!("{hello}");
        std::io::stdout().flush().ok();

        while let Ok(line) = rx.recv() {
            if line.trim().is_empty() {
                continue;
            }
            let resp = session.handle_line(&mut cx, &line);
            println!("{resp}");
            std::io::stdout().flush().ok();
            if session.quit {
                break;
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn main() {
    driver::run();
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("the eidola UI driver requires macOS (real offscreen Metal rendering)");
    std::process::exit(1);
}
