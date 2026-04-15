# gpui-markdown-editor — Agent Development Guide

A WYSIWYG markdown editor as a `gpui-component`-style widget. Targets
`apps/gui` (the chat composer) but is intentionally generic so other gpui
applications can drop it in.

This crate is the Rust / GPUI successor to the SwiftUI / TextKit 2 editor in
`apps/macos/Packages/MarkdownEditor`. The foundational goals haven't changed
— the implementation target has. **Read that crate's `AGENTS.md` first** for
the full design rationale. This file records what's specific to the gpui port
and what's deliberately *different* from the Swift implementation.

## Foundational Goals (unchanged from Swift)

1. **Valid, compliant markdown.** The buffer is always valid CommonMark
   (with one exception: two consecutive newlines are preserved as user-visible
   paragraph separation rather than collapsed). The editor *may* normalize on
   input (e.g. setext → ATX) but never invents non-spec syntax.
2. **A single editable document.** Selections cross spans and blocks; the
   user thinks in markdown.
3. **Block composability.** Lists / blockquotes nest arbitrarily; leaf blocks
   (code, math) are inert islands.

## Target Behavior (unchanged from Swift)

The user edits markdown source but sees rich formatting — *except around their
cursor*. Delimiters hide when the cursor is outside the construct and reveal
(dimmed) when the cursor or an active selection enters it. Formatting applies
to content, never to delimiters; the underlying markdown is never modified to
achieve a visual effect; copy/paste always produces raw markdown.

The full per-construct expectations table from
`apps/macos/Packages/MarkdownEditor/AGENTS.md` applies verbatim. Any divergence
is a bug.

## Pixel-fidelity goal with chat rendering

The chat in `apps/gui` already uses `gpui-component`'s `TextView::markdown`
to render messages. The composer uses this editor. **The two must
match pixel-for-pixel** — what the user types is what they see in the
transcript after they send. That implies sharing typography, paragraph
spacing, list indentation, and code-block styling with the chat renderer.
Practically, this editor will lift the same `TextViewStyle`-equivalent
inputs (paragraph_gap, heading_base_font_size, heading_font_size callback,
highlight theme, code-block style refinement) so callers configure both
sides identically. The editor's output is *not* `TextView` though — that
widget can't host a cursor or selection — so the two implementations need
to walk lockstep on the styled-text side without sharing the rendering code
itself.

If a future change forces a fork between the two (e.g. the editor needs a
gpui-component capability that lands in our crate first), fork forward in the
chat renderer too rather than letting the surfaces drift.

## Architectural pivot from Swift

The Swift editor used TextKit 2 with one giant `NSAttributedString` and a
custom layout-fragment subclass for full-width decorations. The gpui version
is structurally different:

```
EditorState + EditorEvent  →  update()  →  new EditorState
                                                 ↓
                                            parse()  →  SyntaxTree
                                                 ↓
                                            render(state, tree)  →  RenderSpec
                                                 ↓
                                            BlockElement (gpui Element, one per block)
```

Key differences from Swift / TextKit 2:

- **Per-block GPUI `Element`s** instead of one attributed string. Each block
  is its own painter, which makes full-width code-block backgrounds and
  blockquote borders trivial — they're per-block decorations, not custom
  layout fragments.
- **`display_to_source` per shaped line** replaces the ZWSP length-matching
  trick from Swift. We *can* shape display strings shorter than their source
  range (delimiters genuinely removed from the shaped line) because gpui's
  `WrappedLine` returns positions in display-byte coordinates. We translate
  back at hit-test / cursor-paint time via a per-line map, mirroring the
  pattern from `gpui::Editor`.
- **No `NSTextView` subclass / responder chain.** Keyboard input goes
  through gpui actions. IME / dead-key composition uses
  `EntityInputHandler` (the gpui input-handler trait) — see the
  `examples/input.rs` upstream pattern.
- **No focus-anchor associated objects.** Shift-arrow extension state lives
  on `Selection::Range::anchor`, which gpui can update from a cursor-state
  member directly without runtime tricks.

## Minimum viable scope (current)

The first cut covers:

- ATX headings (`# `..`###### `): font size + weight, dimmed `# ` prefix
  when cursor is on the heading line.
- Bold (`**`): bold trait + dim/hide of `**` delimiters.
- Italic (`*`): italic trait + dim/hide of `*` delimiters.
- Strikethrough (`~~`): strikethrough decoration + dim/hide of `~~`.
- Body paragraphs (no-op styling).
- Soft-wrap.
- Cursor + selection geometry, mouse hit-test, basic keyboard navigation
  (arrows / home / end / doc start / doc end), basic editing (insert text,
  backspace / delete, newline / line break), select-all.

Explicitly *out* of this first phase: setext-heading normalization, ordered
list renumbering, code blocks, blockquotes, lists (all kinds), inline code,
links, images, thematic rules, tables, HTML, IME marked-text, word /
line-aware delete, indent / outdent, and tab-trapped focus traversal. Each
will land as a follow-up.

## Module map

| File | Purpose |
|------|---------|
| `state.rs` | `EditorState` (markdown + selection), `Selection` |
| `event.rs` | `EditorEvent` enum — every user action |
| `update.rs` | Pure `update(state, event) -> state` |
| `parser.rs` | pulldown-cmark walker → `Vec<SyntaxNode>` |
| `syntax.rs` | `SyntaxNode`, `NodeKind` (only the constructs we render) |
| `render.rs` | Pure `render(state, tree, style) -> RenderSpec` |
| `render_spec.rs` | `RenderSpec`, `RenderBlock`, `InlineRun`, `InlineStyle` |
| `style.rs` | `MarkdownStyle` — derived from `gpui_component::Theme` |
| `element.rs` | `BlockElement` — paints one block, owns a `display_to_source` map per shaped line |
| `editor.rs` | `MarkdownEditor` — gpui `Render` view, owns state, dispatches actions |
| `bin/demo.rs` | Standalone demo window |

## Theme integration

The editor does **not** carry its own color palette. `MarkdownStyle::from_theme`
derives every color (text, secondary, delimiter, background) from
`gpui_component::Theme`. Day / Night just work because they're the theme's
job. Callers can override individual fields after construction (font size,
heading callback, paragraph gap) the same way `apps/gui::chat::markdown_style`
overrides `TextViewStyle`.

## Testing — two tiers (mirrors `apps/gui`)

### Behavior tests (`tests/behavior.rs`) — the regression gate

Built on `gpui::TestAppContext`. They run on libtest's worker thread with
mocked rendering, so they're cheap and deterministic.

- Construct an `Entity<MarkdownEditor>` with a known initial state.
- Drive interactions through the view's `focus_handle` (the production
  dispatch path).
- Assert against `EditorState`, `RenderSpec`, or `MarkdownEditor` public
  state with `read_with`.

These cover state transitions and the renderer's pure-function decisions
(delimiter hide vs. dim, inline runs, block kinds). They do **not** verify
geometry — that's the visual tier.

### Visual snapshots (`tests/visual.rs`) — local debug aid

Built on `gpui::VisualTestAppContext`. Configured `harness = false` so
`fn main()` runs on the macOS main thread (libtest's worker harness would
SIGABRT inside AppKit). Renders each case **twice** — once Day, once Night —
and writes/compares `tests/snapshots/<name>-{day,night}.png`.

Mirrors `apps/gui/tests/visual.rs` exactly:

- Missing PNG → write it and report `written`.
- Mismatch → save `<name>-<mode>.new.png` for review and fail.
- `UPDATE_SNAPSHOTS=1` overwrites.

The PNGs are **gitignored** because they're platform-bound. They're a debug
aid for the developer making a UI change, not a CI gate.

### Required cursor-position coverage

The Swift `AGENTS.md` mandates that every visual test exercise the cursor at
many positions, not just where typing left it. That requirement carries over.
Each construct's snapshot suite must include cursor:

1. Inside the construct (delimiters dimmed/visible).
2. Just outside, on either side (delimiters hidden).
3. On a separate line (delimiters hidden).
4. With an active selection that overlaps the construct.

The Kitchen Sink case combines all supported constructs in one document and
moves the cursor to varied positions — it's the safety net for interaction
bugs between features.

## Build & run

```bash
# library + binary build
cargo build -p gpui-markdown-editor

# behavior tests (the gate)
cargo test -p gpui-markdown-editor

# visual snapshots (write goldens on first run, then compare)
cargo test -p gpui-markdown-editor --test visual
UPDATE_SNAPSHOTS=1 cargo test -p gpui-markdown-editor --test visual

# demo binary
cargo run -p gpui-markdown-editor --bin demo
```

## Process for adding a markdown feature (carried over from Swift)

1. **Discover.** Build the feature against the demo, *read every snapshot*,
   identify deviations from expected behavior. Think like a user, not a code
   reviewer.
2. **Articulate tests.** Turn discovered issues into behavior tests
   (state / render-spec assertions) and visual snapshot cases (with cursor
   placed at the positions listed above).
3. **Fix.** Update implementation, re-read snapshots, iterate.

### Where to make changes

| What | Where |
|------|-------|
| Keyboard behavior (Enter, Backspace, Tab) | `update.rs` |
| New events | `event.rs` + `update.rs` + `editor.rs` action wiring |
| New construct (parsing) | `parser.rs` + `syntax.rs` |
| Cursor-aware delimiter visibility | `render.rs` |
| Visual styling (fonts, colors, spacing) | `style.rs` (derived from theme) |
| Glyph substitution / hidden ranges | `render.rs` (RenderBlock fields) + `element.rs` (consumed at shape time) |
| Full-width decorations (code bg, blockquote border) | `render.rs::Decoration` + `element.rs::paint_decoration` |

## Known design notes

- **`gpui-component` pin is shared with `apps/gui`.** Same git rev. If we
  bump `apps/gui`, we bump here in lockstep so cargo unifies.
- **The crate has no Eidola-specific symbols.** It only depends on `gpui`,
  `gpui-component`, `gpui-component-assets`, `pulldown-cmark`,
  `unicode-segmentation`, and `smallvec`. Other gpui apps can use it
  without pulling in the rest of the workspace.
- **No async / no I/O.** Everything is synchronous and pure except for the
  gpui paint hooks themselves. No tokio, no spawned tasks.
