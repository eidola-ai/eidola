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
//! {"cmd":"open","scene":"chat_conversation"}            // optional width/height
//! {"cmd":"windows"}
//! {"cmd":"elements","window":1}                          // named probe targets
//! {"cmd":"click","window":1,"target":"chat/model-label"} // or "x"/"y"; alt/command/shift bools
//! {"cmd":"type","window":1,"text":"Hello there"}
//! {"cmd":"keys","window":1,"keys":"cmd-enter"}           // space-separated keystrokes
//! {"cmd":"modifiers","window":1,"alt":true}              // hold/release modifiers
//! {"cmd":"scroll","window":1,"target":"chat/transcript","dy":-300}
//! {"cmd":"resize","window":1,"width":480,"height":700}
//! {"cmd":"screenshot","window":1}                        // optional "path"
//! {"cmd":"theme","mode":"night"}                         // or "day"
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
//!
//! Scenes are stub-store fixtures (no backend, no network), mirroring the
//! visual snapshot cases — deterministic scenes the agent can interact with.
//! Store-backed flows stop at the stub guard exactly as behavior tests do;
//! local interaction (composer editing, submit's local append, picker, hover
//! reveals, navigation) is fully live.

use std::collections::HashMap;
use std::io::{BufRead, Write as _};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use eidola_app_core::updates::{UpdateCheckResult, UpdateCheckSnapshot, VerifiedRelease};
use eidola_app_core::{
    BalancePoolInfo, BalancesResult, ConfigState, ModelInfo, PriceInfo, SpaceInfo, SpaceMessage,
};
use eidola_gui::chat::ChatView;
use eidola_gui::library::LibraryView;
use eidola_gui::probe;
use eidola_gui::record::RecordView;
use eidola_gui::settings::SettingsView;
use eidola_gui::stores::{Stores, StoresStub};
use eidola_gui::updates::UpdatesView;
use eidola_gui::window_input::WindowInput;
use gpui::{
    AnyWindowHandle, App, AppContext, Capslock, Modifiers, ModifiersChangedEvent, Pixels,
    ScrollDelta, ScrollWheelEvent, Size, TouchPhase, VisualTestAppContext, point, px, size,
};
use gpui_component::{Root, Theme, ThemeMode};
use gpui_component_assets::Assets;
use serde::Deserialize;
use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// Protocol
// ---------------------------------------------------------------------------

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
            name: "onboarding_welcome",
            description: "Chat window, no account: the welcome page (Begin button)",
            default_size: size(px(705.), px(705.)),
            build: |window, cx| {
                let stores = stub_stores(cx, |s| {
                    s.config_state = Some(config_state(false));
                });
                let view =
                    cx.new(|cx| ChatView::new(stores, None, WindowInput::new(cx), window, cx));
                root(view, window, cx)
            },
        },
        Scene {
            name: "onboarding_plans",
            description: "Chat window, account with zero balance: the plans page",
            default_size: size(px(705.), px(705.)),
            build: |window, cx| {
                let stores = stub_stores(cx, |s| {
                    s.config_state = Some(config_state(true));
                    s.balances = Some(BalancesResult {
                        available: 0,
                        pools: Vec::new(),
                    });
                    s.prices = prices();
                });
                let view =
                    cx.new(|cx| ChatView::new(stores, None, WindowInput::new(cx), window, cx));
                root(view, window, cx)
            },
        },
        Scene {
            name: "chat_empty",
            description: "Ready chat window with an empty page and live composer",
            default_size: size(px(705.), px(705.)),
            build: |window, cx| {
                let stores = ready_stores(cx);
                let view =
                    cx.new(|cx| ChatView::new(stores, None, WindowInput::new(cx), window, cx));
                root(view, window, cx)
            },
        },
        Scene {
            name: "chat_conversation",
            description: "Ready chat window with a four-turn transcript",
            default_size: size(px(760.), px(620.)),
            build: |window, cx| {
                let stores = ready_stores(cx);
                let view = cx.new(|cx| {
                    let mut view = ChatView::new(stores, None, WindowInput::new(cx), window, cx);
                    view.set_messages_for_test(conversation(), cx);
                    view
                });
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
                let view = cx.new(|cx| SettingsView::new(stores, WindowInput::new(cx), window, cx));
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
            description: "Record window (stub data: empty listings, live section strip)",
            default_size: size(px(860.), px(640.)),
            build: |window, cx| {
                let stores = stub_stores(cx, |_| {});
                let view = cx.new(|cx| RecordView::new(stores, window, cx));
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

/// A funded, ready account with a populated model list (so the ⌥ model
/// reveal and picker are live).
fn ready_stores(cx: &mut App) -> Stores {
    stub_stores(cx, |s| {
        s.config_state = Some(config_state(true));
        s.models = models();
    })
}

fn config_state(has_account: bool) -> ConfigState {
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
    ]
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
    })
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
        let shot_dir = std::env::temp_dir().join(format!("eidola-driver-{}", std::process::id()));
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
            } => {
                let pos = self.position(cx, window, target.as_deref(), x, y)?;
                let handle = self.window(window)?;
                let modifiers = Modifiers {
                    alt,
                    platform: command,
                    shift,
                    ..Default::default()
                };
                cx.simulate_click(handle, pos, modifiers);
                Ok(json!({"clicked": {"x": pos.x.as_f32(), "y": pos.y.as_f32()}}))
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

            Cmd::Theme { mode } => {
                let mode = match mode.as_str() {
                    "day" | "light" => ThemeMode::Light,
                    "night" | "dark" => ThemeMode::Dark,
                    other => return Err(format!("unknown theme mode \"{other}\" (day|night)")),
                };
                cx.update(|cx| Theme::change(mode, None, cx));
                for w in self.windows.values() {
                    cx.update_window(w.handle, |_, window, _| window.refresh())
                        .ok();
                }
                cx.run_until_parked();
                Ok(json!({}))
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

fn main() {
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
