# apps/gui — gpui-based Eidola

Guidance for AI coding agents working on the gpui macOS app. Cross-cutting workspace context (server, app-core, build-system, conventions) lives in the top-level `AGENTS.md`.

## What this app is

A native Rust client for Eidola, built on [gpui](https://github.com/zed-industries/zed/tree/main/crates/gpui) (the immediate-mode UI framework Zed is built on) and [gpui-component](https://github.com/longbridge/gpui-component) (a shadcn-style widget library on top of gpui). It mirrors the SwiftUI `apps/macos/` app feature-for-feature, sharing the same `crates/eidola-app-core/` backend. macOS-only today; Linux is the next target.

## Core wrapper (`src/core.rs`)

`AppCore` lives in `crates/eidola-app-core/`; gpui can't host it directly because gpui's executor is smol-based and `AppCore` has its own tokio multi-thread runtime. The bridge:

- `Core` is a gpui `Entity<Core>` that holds an `Option<Arc<AppCore>>` (None for stub-mode tests) plus cached snapshots — `config_state`, `balances`, `prices`, `credentials`, `models`. Views hold `Entity<Core>` and re-render reactively via `cx.notify()` after each cache mutation.
- Async ops are bridged with `tokio::sync::oneshot` channels: the call is `spawn`ed on `AppCore::runtime()` (tokio) and the receiver is awaited from gpui's executor. `oneshot::Receiver` is runtime-agnostic, which is what makes this safe.
- `Core::stub()` builds an `Entity<Core>` with `inner: None` for tests. Views that hit `Core::app_core()` early-return after the local state mutation. `ChatView::submit` exemplifies the pattern: it pushes the user message and sets `thinking=true`, then bails before the actual chat call.

## View structure

| File | Window root | Purpose |
|---|---|---|
| `chat.rs` | `ChatView` | Main chat window — message list + input + Send action |
| `settings.rs` | `SettingsView` | Settings window — custom three-button tab strip switching between... |
| `general.rs` | `GeneralView` | ...base URL + attestation state (read-mostly) |
| `account.rs` | `AccountView` | ...account/balance/allocate/prices |
| `wallet.rs` | `WalletView` | ...credential list |

All view roots are wrapped in `gpui_component::Root` before being handed to `cx.open_window`. **Root is required** — `gpui_component::Input` calls `Root::read(window, cx)` to track the focused input, and panics if the window's root view type isn't `Root`.

## Theme — Circadian (`src/theme.rs`)

Two `ThemeConfig`s, "Circadian Day" (Light) and "Circadian Night" (Dark), installed onto the global `gpui_component::Theme` after `gpui_component::init`. Switching is driven by OS appearance:

- `Theme::sync_system_appearance` reads the appearance via `cx.window_appearance()` (or `window.appearance()` if a window is passed) and applies the matching config.
- Each opened window subscribes to appearance changes via `theme::observe_window_appearance(window)` so toggling macOS Light/Dark updates live.

The starting palette is lifted from the marketing site (`../www.eidola.ai/index.html`); treat it as a historical seed, not a contract.

**Body font is Newsreader** (variable TTF, SIL OFL 1.1) — bundled at `apps/gui/assets/fonts/Newsreader.ttf` + `Newsreader-Italic.ttf` and embedded into the binary via `include_bytes!`, then registered with `cx.text_system().add_fonts`. License at `assets/fonts/OFL.txt`. gpui's macOS text system loads TTF/OTF only — the website's WOFF2 files won't work, which is why we ship the canonical TTFs from `google/fonts`.

## Window model

**Chat windows are non-singleton.** Every `NewSpace` invocation opens a fresh `ChatView`, each owning its own `space_id` so they're independent conversations sharing the same `Core`. `open_main_window` calls `cx.activate(true)` after `cx.open_window` so a window opened from another app's context (dock right-click while a different app is foreground) brings Eidola to the front rather than opening behind.

**Settings is a singleton.** `AppGlobal.settings_window: Option<WindowHandle<Root>>` caches the handle, and `OpenSettings` raises the existing window via `window.activate_window()` if it's still open. Both open paths are **synchronous** (via `App::open_window`) so the cache is populated before the handler returns. Liveness is checked by matching the cached `WindowId` against `cx.windows()` (the authoritative live list) — borrowing Zed's pattern, except Zed can use `AnyWindowHandle::downcast::<SettingsWindow>` directly because their settings root is uniquely typed; ours is `gpui_component::Root` (shared with chat windows), so we match by id instead. A stale id self-heals on the next invocation — no `on_release` bookkeeping needed.

## Edge-to-edge titlebar

`transparent_titlebar()` returns `TitlebarOptions { appears_transparent: true, title: None, traffic_light_position: Some(point(14, 11)) }`. macOS extends the content view under the traffic-light buttons and stops painting a separate titlebar background. Each view leaves room at the top so the lights don't land on real UI:

- **`chat::TITLE_BAR_RESERVE`** (36px on macOS): vertical reserve, plus a `theme.background → transparent` linear-gradient overlay (`title_bar_overlay`) painted absolutely over the scroll area. Messages scrolling up under the band fade smoothly into the chrome instead of clipping at a hard edge.
- **`settings::TAB_STRIP_LEFT_PAD`** (80px on macOS): horizontal pad on the tab strip. The tab row doubles as the title bar — the lights live to its left on a shared `theme.background` band. 80px matches gpui-component's own `TITLE_BAR_LEFT_PADDING`.

## macOS UX — menus, keybindings, action dispatch

All wired in `src/lib.rs::install_menus`, `install_keybindings`, `install_action_handlers`. **Order of those calls matters** — see [Ordering invariant](#ordering-invariant) below.

### Menus (`cx.set_menus`)

- **Eidola**: About / Settings… / Hide / Hide Others / Show All / Quit
- **File**: New Space / Close Window
- **Edit**: Undo / Redo / Cut / Copy / Paste / Select All — Cut/Copy/Paste/Select All declared via `MenuItem::os_action(_, _, OsAction::*)` so they bind to the standard macOS selectors `cut:` / `copy:` / `paste:` / `selectAll:` and route through the responder chain to whatever has focus
- **Window**: Minimize / Zoom

`cx.set_dock_menu` adds "New Space" for the dock right-click menu.

**The "Window" menu name is special.** gpui_macos detects a menu literally named `"Window"` and calls `app.setWindowsMenu_(menu)` — which is how AppKit recognizes the app as a fully-wired macOS app and reliably dispatches menu key-equivalents in edge cases (no key window after ⌘Tab back, all windows closed). The Hide / Hide Others / Show All trio play the same "I'm a real app" signaling role.

### Keybindings (`cx.bind_keys`)

⌘, (Settings), ⌘N (NewSpace), ⌘W (CloseWindow), ⌘Q (Quit), ⌘H (Hide), ⌥⌘H (HideOthers), ⌘M (Minimize), and ⌘↩ for `Send` in the `ChatView` key-context.

### Ordering invariant

`install_keybindings(cx)` **must** run before `install_menus(cx)`. `cx.set_menus` snapshots the keymap when it builds NSMenuItems and attaches each item's `keyEquivalent` from `keymap.bindings_for_action(action)`. Setting menus before binding keys leaves the keymap empty at lookup time, no keystroke is attached, and macOS can't intercept the shortcut at the menu level — which then breaks ⌘N / ⌘Q etc. when no window has key focus (the only path that *requires* the menu-level intercept; with a window focused, gpui's per-window binding dispatch handles it independently). **Diagnostic signal**: items appear in the menu without their shortcut text on the right side.

### Action handlers (`cx.on_action`)

Most handlers are global (registered on `App`). Two notable patterns:

- **Window-targeting handlers** (`Minimize`, `Zoom`) capture `cx.active_window()` and call `cx.defer` to invoke `window.minimize_window()` / `zoom_window()` *after* the current update completes. Without `defer`, a direct `handle.update(cx, …)` on the same window we were dispatched inside fails (its slot is already taken), `.ok()` swallows the Err, and nothing happens.
- **`CloseWindow` is registered per-view**, not globally. Each view does `.on_action(cx.listener(|_, _: &CloseWindow, window, _| window.remove_window()))` on its root v_flex, and `track_focus`es a handle that's `focus()`ed in the view's constructor (so the dispatch path reaches the listener even before the user clicks anything). The intentional consequence: `is_action_available` returns true only when a window with the listener is alive, so macOS auto-disables the "Close Window" menu item (and ⌘W) when no window is open.

### Lifecycle

- `cx.activate(true)` at launch: brings the app to the foreground from frame 0 so the menu bar is fully connected before the user interacts.
- `Application::on_reopen` (registered on the `Application` builder *before* `run()` — the method takes `&self` and returns `&Self`, while `run()` consumes by value, so it can't be chained inline; bind the application to a local first): when the dock icon is clicked with no windows open, opens a new chat window. Without this, closing the last window leaves the app running but unreachable.

## .app bundling — required, not cosmetic

A bare `cargo run -p eidola-gui` binary launches as a command-line tool from AppKit's perspective, not a real app. `setActivationPolicy(Regular)` papers over the common path (menu shows, items enable, mouse clicks dispatch), but **menu key-equivalents fail to dispatch when no window is key**, even with the keymap-ordering fix above. The diagnostic signal is the menu bar showing the binary name (`eidola-gui`) instead of the app name (`Eidola`).

The fix is a proper macOS bundle:

- `apps/gui/Support/Info.plist` — `CFBundleIdentifier = tech.m6i.eidola-gpui` (distinct from the SwiftUI app's `tech.m6i.eidola` so they coexist), `CFBundleExecutable = Eidola`, `NSPrincipalClass = NSApplication`, `NSHighResolutionCapable = true`.
- `scripts/package-gui-app.sh` — copies `target/{debug,release}/eidola-gui` to `Contents/MacOS/Eidola` (renamed to match `CFBundleExecutable` — mismatch falls back to tool-mode), copies the Info.plist, ad-hoc codesigns. Output at `apps/gui/build/Eidola.app` (gitignored).
- `just build gui` runs `cargo build` then the package script on macOS. `just run gui` builds + `open`s the .app. Non-macOS falls back to `cargo run`.

## Testing — two tiers

Both run via `cargo test -p eidola-gui`. `apps/gui` has both `[lib]` and `[[bin]]` so the integration tests can import view modules.

### Behavior tests (`tests/behavior.rs`) — the regression gate

Built on `gpui::TestAppContext` (mocked rendering, deterministic dispatcher) so they run on libtest's worker thread without AppKit. Pattern:

1. Build a `Core::stub()` entity with whatever fixture state you need.
2. Open a window with the view under test (via `cx.open_window`).
3. Drive interactions through the view's `focus_handle()` — the same path keystrokes take in production.
4. Assert against the view/core's public state with `read_with`.

Stub cores have `inner: None`, so `Core::app_core()` returns `None`; views that hit that path early-return after the local state mutation. HTTP-mocked tests (real `AppCore` against a `wiremock` server) are the natural next layer.

### Visual snapshots (`tests/visual.rs`) — local-only debug aid

**Not** a regression gate. Built on `gpui::VisualTestAppContext` (real Metal renderer, offscreen window at -10000,-10000, deterministic dispatcher). Configured as `[[test]] harness = false` so `fn main()` runs on the macOS main thread (libtest's worker-thread harness would SIGABRT inside AppKit).

Cases live in `tests/visual/cases.rs`; the harness in `tests/visual/harness.rs` wraps each user view in a `Root` and renders it **twice — once in Circadian Day, once in Circadian Night** — by calling `Theme::change` between renders. Each case writes/compares two files: `tests/snapshots/<name>-day.png` and `<name>-night.png`. Case build closures must be `Fn` (invoked once per mode); they construct fresh entities each call.

The PNGs are **gitignored** — pixels are platform- and machine-bound (Metal+CoreText vs wgpu+cosmic-text on Linux; font hinting differs across macOS minor versions), so committing them would mean false-positive regressions in CI and on every other developer's machine. Their value is local: agents/humans can `Read` a PNG to "feel" a view at a state, and a developer iterating on a UI change can re-render and eyeball-diff their previous run.

Behavior:
- Missing PNG → write it and report `written`.
- Mismatch against a previously-written local PNG → write `<name>-<mode>.new.png` for review and fail.
- `UPDATE_SNAPSHOTS=1` overwrites.

Recipes: `just render-snapshots` (verify/write), `just render-snapshots-update` (accept).

### Why both tiers?

Behavior tests catch logic regressions (clicking X must call `core.Y(z)`; an empty Send must be a no-op) and survive across platforms. Visual snapshots are the "did I accidentally change the layout?" check that's only meaningful to the dev making the change. Together they let agents make UI changes confidently: behavior tests gate the merge, visual snapshots let the agent verify the change *looks* right by reading the freshly-written PNG.

## gpui / gpui-component pinning

`gpui-component` (longbridge, rev `01a116a15e9660963a2aa07d0192e38785d8b9ad`) pulls `gpui` and `gpui_platform` from `zed-industries/zed` without a rev. We mirror that exact spec in `Cargo.toml` so cargo unifies on a single `gpui` copy. `Cargo.lock` is the canonical pin for the resolved zed commit.

## Non-Rust dependencies

System frameworks gpui already pulls in (Cocoa, AppKit, CoreFoundation, CoreGraphics, CoreText, CoreVideo, Metal, Foundation) — no GTK/Qt/node/python. Build deps require Xcode Command Line Tools (`xcode-select --install`).
