# crates/eidola-gui — gpui-based Eidola

Guidance for AI coding agents working on the gpui macOS app. Cross-cutting workspace context (server, app-core, build-system, conventions) lives in the top-level `AGENTS.md`.

## What this app is

A native Rust client for Eidola, built on [gpui](https://github.com/zed-industries/zed/tree/main/crates/gpui) (the immediate-mode UI framework Zed is built on) and [gpui-component](https://github.com/longbridge/gpui-component) (a shadcn-style widget library on top of gpui). The sole macOS GUI for the project; the CLI in `crates/eidola-cli/` shares the same `crates/eidola-app-core/` backend. macOS-only today; Linux is the next target.

## Core wrapper (`src/core.rs`)

`AppCore` lives in `crates/eidola-app-core/`; gpui can't host it directly because gpui's executor is smol-based and `AppCore` has its own tokio multi-thread runtime. The bridge:

- `Core` is a gpui `Entity<Core>` that holds an `Option<Arc<AppCore>>` (None for stub-mode tests) plus cached snapshots — `config_state`, `balances`, `prices`, `credentials`, `models`. Views hold `Entity<Core>` and re-render reactively via `cx.notify()` after each cache mutation.
- Async ops are bridged with `tokio::sync::oneshot` channels: the call is `spawn`ed on `AppCore::runtime()` (tokio) and the receiver is awaited from gpui's executor. `oneshot::Receiver` is runtime-agnostic, which is what makes this safe.
- `Core::stub()` builds an `Entity<Core>` with `inner: None` for tests. Views that hit `Core::app_core()` early-return after the local state mutation. `ChatView::submit` exemplifies the pattern: it pushes the user message and sets `thinking=true`, then bails before the actual chat call.

## View structure

| File | Window root | Purpose |
|---|---|---|
| `chat.rs` | `ChatView` | Main chat window — message list + input + Send action, plus the onboarding empty states (see [Onboarding](#onboarding--the-chat-windows-empty-states)) |
| `settings.rs` | `SettingsView` | Settings window — custom three-button tab strip switching between... |
| `general.rs` | `GeneralView` | ...base URL + attestation state (read-mostly) |
| `account.rs` | `AccountView` | ...account/balance/allocate/prices |
| `wallet.rs` | `WalletView` | ...credential list |

All view roots are wrapped in `gpui_component::Root` before being handed to `cx.open_window`. **Root is required** — `gpui_component::Input` calls `Root::read(window, cx)` to track the focused input, and panics if the window's root view type isn't `Root`.

**`Core::spawn` debounces on a single `busy` flag** — sequential entity-method calls (`fetch_models` then `fetch_balances`) silently drop all but the first. Two consequences baked into the API: multi-fetch operations are *combined* into one spawn (`fetch_chat_startup` = models + balances + credentials; `fetch_plans_data` = prices + balances), and flows that must never be dropped (onboarding's account creation / checkout / balance poll) bypass the entity methods entirely via static oneshot helpers (`Core::account_create`, `Core::account_checkout`, `Core::account_balances` — same shape as `Core::chat`) so the calling view owns its own typed in-flight/error state.

## Onboarding — the chat window's empty states

From zero to first answer without leaving the page: onboarding is not a wizard window, it's what the chat window's *empty state* renders until the account is usable. The state machine lives in `ChatView` (`chat.rs`):

- **Stage is derived, not stored.** `ChatView::onboarding_stage(core, composer_empty)` returns `Welcome` (no account in `config_state`), `Plans` (account + *known-zero* balance + empty wallet credentials), or `Ready`. Any page content — messages, an in-flight stream, an error band, or composer text — short-circuits to `Ready` so the onboarding pages only ever replace a genuinely empty page. A `None` balance (not yet fetched) is `Ready`, never `Plans`: we don't claim the user is unfunded on an assumption. The only stored bridge is `OnboardingFlow.entered_plans`, which holds the page on `Plans` between account creation and the first balances fetch so the flow doesn't flash through the blank page.
- **`OnboardingFlow`** holds the local in-flight bits — `creating_account`, `checkout_pending`, `awaiting_checkout`, errors, `dismissed` — and every flag corresponds to a real request currently running or an explicit user choice (the no-fake-states rule). Async work goes through the static `Core` oneshot helpers, giving the view typed `AppError` results.
- **Welcome** — wordmark + three sentences + a single "Begin" button → `account_create` (anonymous; nothing to fill in). Success refreshes `config_state`, sets `entered_plans`, and kicks `fetch_plans_data`; failure renders inline with Begin available for retry.
- **Plans** — `prices` rendered as hairline-rule rows (name · price, credits underneath) in the prose column; clicking a row → `account_checkout` → `cx.open_url` → `awaiting_checkout` + a 3-second balance poll (`start_balance_poll`, each tick a real `GET /v1/account/balances`; poll errors surface inline and polling continues). A positive balance flips the derived stage to `Ready`. A quiet "I'll do this later" link sets `dismissed`.
- **Degraded honesty** — a later submit failing with `AppError::InsufficientBalance` (typed routing, no string matching — see `apply_chat_failure`) sets `show_plans_after_error`, which renders the same plans list below the transcript's error band. Not a modal.
- Silent credential provisioning itself is app-core's job (see the workspace `AGENTS.md`): a funded account never sees any of this — `chat()` auto-allocates from balance.

Behavior tests cover the stage derivation and each transition's in-flight state (stub core: handlers early-return after the local mutation, mirroring `submit`); visual snapshots cover `onboarding_welcome`, `onboarding_plans`, `onboarding_plans_waiting`, and `chat_insufficient_balance_plans`.

## Theme — Circadian (`src/theme.rs`)

Two `ThemeConfig`s, "Circadian Day" (Light) and "Circadian Night" (Dark), installed onto the global `gpui_component::Theme` after `gpui_component::init`. Switching is driven by OS appearance:

- `Theme::sync_system_appearance` reads the appearance via `cx.window_appearance()` (or `window.appearance()` if a window is passed) and applies the matching config.
- Each opened window subscribes to appearance changes via `theme::observe_window_appearance(window)` so toggling macOS Light/Dark updates live.

The starting palette is lifted from the marketing site (`../www.eidola.ai/index.html`); treat it as a historical seed, not a contract.

**Body font is Newsreader 16pt** (SIL OFL 1.1) — five static TTF instances (Regular, Italic, SemiBold, Bold, BoldItalic) from `productiontype/Newsreader`, bundled in `crates/eidola-gui/assets/fonts/Newsreader16pt-*.ttf`, embedded via `include_bytes!`, and registered with `cx.text_system().add_fonts`. License at `assets/fonts/OFL.txt`.

We ship statics rather than the variable upright + italic because **gpui's macOS text system does not apply variable-font weight axes**: `gpui_macos::text_system::add_fonts` registers each TTF as one face with the properties of its default instance, and `font_kit::matching::find_best_match` picks the closest face per weight request. With only the variable TTFs registered, every weight request — `**strong**` (BOLD), headings (SEMIBOLD/BOLD), etc. — resolved to the Regular default and rendered un-bold. Five static faces make `find_best_match` pick correctly. Family name in the theme is `"Newsreader 16pt"` (the typographic family — nid 16 — that all five faces report; SemiBold sets nid 16 explicitly to override its nid 1 = `Newsreader 16pt SemiBold`, the canonical workaround for the Windows OS/2 4-style-per-family limit).

## Chat typography — book metaphor

The chat is the long-form reading view, so it gets typography of its own rather than inheriting the UI baseline. Settings/wallet/account stay at the 16px UI size; chat body is governed by three constants in `chat.rs` (`PROSE_FONT_SIZE`, `PROSE_LINE_HEIGHT`, `PROSE_MAX_WIDTH_REM`) plus a custom `markdown_style` for `TextView::markdown`.

- **Body 17px / line-height 1.65×.** Newsreader at 17 sits in proportion with a real book page; 1.65 is the readability sweet spot for serifs at this size. Both are applied on a `prose()` wrapper around every `TextView::markdown` call so the markdown renderer inherits them through the normal text-style cascade — gpui-component's `TextViewStyle` itself doesn't expose body size or leading.
- **Centered measure, ~640px wide** (`max_w(rems(40.))`, anchored to the theme's 16px rem so it stays absolute regardless of the prose font bump). Lands at ~65–72 chars/line, the canonical long-form measure. Centering uses a `prose_row()` h-flex with `justify_center` rather than `mx_auto`, because v_flex children stretch to full width by default.
- **Heading scale anchored to body, gentler ramp.** `heading_base_font_size = PROSE_FONT_SIZE` (instead of the gpui default 14px), and a callback returns h1 1.5× / h2 1.25× / h3 1.125× / h4–h6 1.0×. The default scale (h1=2×) reads like a marketing page; a book has a flatter type ramp where weight (BOLD/SEMIBOLD) carries most of the hierarchy.
- **Paragraph gap 1.5 rem** in the markdown renderer (vs gpui-component's default 1.0). About 85% of a body line — clear paragraph breaks without the run-on tightness of a chat tool.
- **Chapter delimiters between turns**, not alternating backgrounds. `chapter_delim()` renders a hairline `theme.border` rule across the prose column, broken in the middle by a small italic participant label ("You" / "Eidola" / "Error") in `text_sm` at `theme.muted_foreground`. Errors swap the label color for `theme.danger`. The whole chat sits on one uniform `theme.background` — speaker differentiation is carried by the rule and label, not by tinted bands. Rows themselves no longer carry vertical padding; the delim's `pt_8 + pb_6` is the entire inter-turn rhythm. The messages column's trailing `pb_8` keeps the last row off the input border.
  - **The first message has no leading delim** (`if idx > 0 { … }` in the message render loop). The user's text is the *start* of the page; only the first speaker change (the assistant's response) needs a label to introduce the new voice. Subsequent same-role turns (rare but possible after errors) still get their delim so the rhythm holds.
  - Layout: an outer `h_flex().w_full().justify_center()` centers an inner `h_flex().w_full().max_w(rems(40))` capped at the prose column. The inner flex has `flex_1` rule divs on either side of the label, sharing the leftover space.
  - **Scroll container invariant**: the `div().id("scroll").flex_1().overflow_y_scroll()` wrapping `messages_col` in `Render` *must* also have `.w_full()`. Without it, taffy content-sizes the scroll div instead of stretching it cross-axis — every descendant inherits that collapsed width, and the delim's `flex_1` rules disappear because there's no leftover space to grow into. The body's row stays full-width because its own content (code blocks) forces the scroll div to expand to that intrinsic width, but a delim with no wide content collapses. Diagnostic signal: in narrow windows the delim rule is visible only on the left side of the label, OR not at all — even though wider windows render fine because some other content has forced the scroll div wide enough. Snapshot regression coverage at 480 / 680 / 820 / 1400 px in `tests/visual/cases.rs`.

Knobs we don't touch yet: list-item spacing (gpui-component's renderer doesn't expose a per-item gap — would need a fork), heading top breath specifically (h2's `pb(rems(0.3))` is hardcoded; headings inherit the previous block's `paragraph_gap` as their top space), code-block padding, and OpenType figure features (`onum` for oldstyle figures would be book-y for prose but conflicts with technical content like version numbers).

The chapter-delimiter approach is intentionally unscrolled: it identifies the speaker only at the boundary, not persistently. Once a long assistant response is partway scrolled past the delim, the reader has no in-view cue to whose turn this is. A future "persistent participant indicator" — likely a small label pinned to the gradient title-bar overlay, or a subtle margin glyph that travels with the column — is the natural follow-up. Don't add it without removing the assumption that the chapter delim alone is enough; the two have to be designed together so they don't compete.

## Inline composer — WYSIWYG markdown editor

The chat has no separate input bar. The composer is a `MarkdownEditor` entity from `crates/gpui-markdown-editor/`, dropped in as the last child of `messages_col` inside the scroll container. It shares the prose column, prose typography (Newsreader 17px / 1.65× / 1.5 rem paragraph gap / gentle heading scale — see [`composer_markdown_style`](src/chat.rs)), and the same `theme.background` as the rest of the page. What the user types is rendered with full markdown styling — bold, italic, headings, lists, fenced code, links, math, … — and shapes pixel-for-pixel like the assistant's reply will when it lands in the transcript.

A "You" `chapter_delim` is rendered above the composer **only when there is preceding content** (`!messages.is_empty() || streaming.is_some() || error.is_some()`). On a fresh, empty page the cursor sits at the top with no header — like opening a blank notebook.

There is no Send button. ⌘↩ is the sole submit path.

**Indefinite growth, single scroll surface.** The editor renders one gpui block per markdown block in a vertical flex column; there's no internal scrollbar. The editor grows naturally with content, and the *outer* `div().id("scroll").overflow_y_scroll()` handles overflow as one continuous unit (editor + all preceding messages).

**Empty-state floor.** With markdown `""`, the editor's render pipeline emits no blocks and its container collapses to zero height — which would make the composer un-clickable after losing focus. `prose().min_h(PROSE_FONT_SIZE * PROSE_LINE_HEIGHT)` puts a one-body-line floor on the editor's wrapper so the empty state stays clickable. Once any character is typed, normal content height takes over.

**Half-viewport bottom padding.** The composer's wrapper carries `pb(window.viewport_size().height * 0.5)` (computed at render time). The cursor never sits pinned to the bottom edge — there's always a half-page of empty space below the active line, so the parent scroll can keep the typing zone in the comfortable middle of the viewport as content grows. `messages_col` no longer carries `pb_8`; the wrapper's pb is the bottom breath. Computed in `Render` rather than as a constant because it tracks live window resizes.

**Cmd+Return dispatch.** ⌘↩ resolves to `Self::submit` via gpui's normal action dispatch. The editor's `MarkdownEditor` key context (see `install_markdown_editor_keybindings` in `lib.rs`) **does not bind `cmd-enter`** — only plain `enter` (newline insertion) and `shift-enter` (line break). So the ChatView-context `cmd-enter → Send` binding remains the innermost matching entry in the focus chain whenever the editor has focus, and `Self::submit` (registered with `cx.listener(Self::submit)` on the v_flex root) fires.

`Self::submit` is idempotent under streaming (`if self.streaming.is_some() { return; }`). The composer remains visible and editable during streaming, but submit is silently a no-op until the stream completes — known UX hole, fine for first iteration. `submit` reads `editor.state.markdown.trim()`, sends the raw markdown source upstream, and resets `editor.state = EditorState::default()` to clear the composer.

**Focus on construction.** `ChatView::new` calls `window.focus(&editor_focus, cx)` against the editor's own focus handle so the cursor lands in the composer when a window opens. The view's own `focus_handle` is still `track_focus`ed by the root v_flex (it's what behavior tests dispatch through), but no longer focused at construction — the editor takes that role.

**Key context routing.** `install_markdown_editor_keybindings` in `lib.rs` binds the editor's actions (Backspace, Delete, Enter, Tab/Shift-Tab, arrow keys, word- and line-granular motion and deletion, ⌘A/X/C/V/⌘⇧V) under `Some("MarkdownEditor")`. Settings/account still use `gpui_component::Input`, which establishes its own `Input` context with parallel bindings, so the two surfaces coexist without conflict. The Edit menu items remain wired to `gpui_component::input::*` actions for the Input widgets in settings/account — menu-driven Cut/Copy/Paste from the chat composer is a known gap (the keyboard works; the menu falls through).

**Test wrapping invariant.** Behavior tests still wrap the view in `gpui_component::Root` (see `tests/behavior.rs::open_view`) even though the chat composer no longer uses `gpui_component::Input` — keeping Root mirrors production (`lib.rs::open_main_window`) and is required for any test that exercises SettingsView. The chat tests no longer *require* Root strictly, but the helper is shared across views so removing the wrap would split the harness.

## Streaming chat

`ChatView` drives chat via `Core::chat_stream` (which wraps `AppCore::chat_stream` from `crates/eidola-app-core/`). The core's streaming method posts `stream: true` to the OpenAI-compatible upstream and forwards each SSE chunk's `delta.content` as `ChatStreamEvent::ContentDelta` and `delta.reasoning_content` / `delta.reasoning` (vLLM-style) as `ChatStreamEvent::ReasoningDelta`. The terminal `ChatResult` is the function's return value, not an event — the channel just closes.

While streaming, `ChatView::streaming: Option<StreamingResponse>` holds the live `reasoning` + `content` buffers and a disclosure-`expanded` flag. It renders as a single row below the user message: a clickable "Thinking…" / "Answering…" / "Thinking (N chars)" header (a `Button` styled with a chevron), the reasoning body when expanded, and the partial markdown content as it grows. On `Done`, `streaming` is dropped and `messages` is re-fetched from the space — only the final content is persisted; reasoning is ephemeral by design (it is not written to the local DB).

The CLI uses streaming too: `crates/eidola-cli/src/main.rs` pumps `ContentDelta` to stdout and `ReasoningDelta` to stderr (dimmed and prefixed with `thinking: ` when stderr is a TTY) so a piped stdout still captures only the final answer.

Refund handling for streaming differs from blocking only in *where* the refund token comes from: SSE responses have no inline JSON body to carry it, so the streaming path always goes through the `/v1/credentials/refund` recovery endpoint after the stream ends. Same recovery endpoint as the existing network-error fallback in the blocking `chat()`.

## Tail-on-bottom scroll

`ChatView` holds a `gpui::ScrollHandle` on the messages-list scroll div (`.track_scroll(&self.scroll_handle)`). Tail policy: if the user is at the bottom (within `TAIL_TOLERANCE = 24px`) just before a content mutation, we re-pin them to the new bottom afterward; if they've scrolled up, we leave their position alone.

The non-obvious bit is the *timing* of the re-pin. `set_offset(-max_offset)` writes a value the next paint reads — but `max_offset` is recomputed *during* paint, after layout. So `cx.defer` (which fires at end-of-effect-cycle, **before** the next paint) sees the stale `max_offset` and undershoots by one chunk's height. `Window::on_next_frame` runs at the start of the *next* frame — i.e. *after* the paint that reflects the latest content — so `max_offset` has already been updated. `ChatView::schedule_tail` uses `on_next_frame`. Diagnostic signal if this regresses: tail looks "sticky-but-one-chunk-behind" — auto-scroll lags the most recent token by exactly one render.

User-initiated submit is a special case: `submit` always sets `pending_tail = true` regardless of where the user was, so a fresh prompt brings the new exchange into view even if the user had been scrolled up — matching ChatGPT/Claude convention.

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

- `crates/eidola-gui/Support/Info.plist` — `CFBundleIdentifier = tech.m6i.eidola-gpui`, `CFBundleExecutable = Eidola`, `NSPrincipalClass = NSApplication`, `NSHighResolutionCapable = true`.
- `scripts/package-gui-app.sh` — copies `target/{debug,release}/eidola-gui` to `Contents/MacOS/Eidola` (renamed to match `CFBundleExecutable` — mismatch falls back to tool-mode), copies the Info.plist, ad-hoc codesigns. Output at `crates/eidola-gui/build/Eidola.app` (gitignored). This is the **dev path** — fast cargo iteration, ad-hoc signed for local launch only.
- `just build gui` runs `cargo build` then the package script on macOS. `just run gui` builds + `open`s the .app. Non-macOS falls back to `cargo run`.
- **Release path:** `nix build .#eidola-gui-macos-universal` (flake.nix) builds an aarch64 + x86_64 universal binary, zeros LC_UUID for reproducibility, and assembles the same `Eidola.app` layout into a Nix store output that's recorded as `eidola-gui-macos-universal` in `artifact-manifest.json` (Nix `narHash`, `darwin/universal` platform). Built locally via `just update-manifest` (alongside the CLI universal binary) and in CI by the `apple` job. Two reproducibility tricks make this work inside the Nix sandbox: (1) gpui's Metal shaders are compiled at app startup via `runtime_shaders` (passed through `gpui_platform/runtime_shaders` → `gpui_macos/runtime_shaders`) instead of via Apple's closed-source `metal` compiler at build time; (2) `flake.nix`'s `commonArgs.preBuild` stitches a missing `assets/assets/icons` sibling layout into the cargo-vendored gpui-component dep, working around an upstream packaging bug where the `icon_named!` proc-macro reads `../assets/assets/icons` relative to its `CARGO_MANIFEST_DIR` (a path that only resolves in the upstream workspace layout, not in `cargo vendor`'s flattened tree).

## Testing — two tiers

Both run via `cargo test -p eidola-gui`. `crates/eidola-gui` has both `[lib]` and `[[bin]]` so the integration tests can import view modules.

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

`gpui-component` (longbridge, rev `dadfca97fec7221acf3ce7047bccdc1eac0506b9`) pulls `gpui` and `gpui_platform` from `zed-industries/zed` without a rev. We mirror that exact spec in `Cargo.toml` so cargo unifies on a single `gpui` copy. `Cargo.lock` is the canonical pin for the resolved zed commit. gpui-component and gpui move in lockstep: a given gpui-component rev expects a matching gpui (e.g. the `flex_shrink_1`/`flex_grow_1` helpers it calls landed in gpui commit `8982fb17`), so bump both together.

**The zed pin is held at `969a67fc`, deliberately *behind* `main`.** The next commit, `39f7849a` ("Log worst hanging tasks and actions", zed #57835), added a global action-timing profiler whose process-wide `ACTION_STATISTICS` spinlock isn't isolated per thread. It's harmless in the single-threaded UI at runtime, but libtest runs our `gpui-markdown-editor` behavior tests in parallel — concurrent `dispatch_action` calls clobber the one global `running` slot and `save_action_timing()` panics on `None`. `969a67fc` is the parent of that commit: it has the flex helpers gpui-component needs but predates the profiler. Because the deps are rev-less, the lock is the only thing holding this — **do not `cargo update` zed past `969a67fc` until the upstream profiler race is fixed** (re-test the editor suite under parallelism before advancing the pin).

## Non-Rust dependencies

System frameworks gpui already pulls in (Cocoa, AppKit, CoreFoundation, CoreGraphics, CoreText, CoreVideo, Metal, Foundation) — no GTK/Qt/node/python. Build deps require Xcode Command Line Tools (`xcode-select --install`).
