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
- Fenced code blocks (` ``` ` / `~~~`): mono font, full-width rounded
  background, no soft-wrap (horizontal scroll for overflow), dim/hide of
  fence chars and info string per cursor.
- Blockquotes (`> `): per-level left border bar + cumulative left
  indent, dim/hide of every `>` marker per blockquote level. Composes
  via the `containers` chain on each leaf — any leaf (paragraph,
  heading, code block, future list item) inside N nested blockquotes
  carries N `Container::BlockQuote` entries; the element layer applies
  `N * blockquote_indent` of left padding and paints N stacked border
  bars. Code-block backgrounds inset *inside* the blockquote indent so
  the border bar stays visible.
- Lists (top-level + nested-inside-blockquote, MVP): unordered
  (`-` / `*` / `+`) and ordered (`1.`, `2.`, …). Each *paragraph* of
  each item parses to one leaf carrying a `Container::ListItem`
  entry — multi-paragraph items render as multiple leaves, all
  inside the same item container. The first paragraph's source
  range includes the item's marker; subsequent paragraphs include
  their leading indent. The element layer adds `list_indent` of
  left padding once per item regardless of paragraph count. Bullet
  glyph substitution and ordered-list renumbering on edit are
  follow-ups. Lists nested inside lists are not yet wired.

  **Whitespace rules `enforce_invariants` enforces inside lists**
  (the analog of the blockquote pairs / soft-break discipline):

  - Lists are always rendered tight *between* items: a `\n\n+` run
    between two items collapses to one `\n`. *Inside* an item,
    `\n\n` is preserved as a paragraph break — multi-paragraph
    items are first-class. The pixel-fidelity divergence with the
    chat renderer's loose-list spacing is the documented cost.
  - Two consecutive hard breaks (`  \n` + scope-continuation +
    `  \n`, in any scope) collapse to a paragraph break in the
    same scope. The trailing-marker `  ` of each hard break is
    dropped; the scope-continuation between them (BQ `> `, list
    indent, …) is preserved. So at top level `foo  \n  \nbar` →
    `foo\n\nbar`; inside a blockquote the depth-D pair shape
    regenerates; inside a list item it produces a paragraph
    break in source. This is what enables Shift+Enter twice as
    the "create a paragraph break inside this item" gesture
    without a dedicated event.
  - No lazy continuations: continuation lines carry exactly
    `marker_width` spaces of indent (3 for `1. ` or `- `, 4 for
    `10. ` etc.) and the preceding line ends with a hard break
    (`  \n`). Editing `9.` → `10.` re-aligns every continuation by
    +1 space; the inverse for narrowing.
  - Soft breaks within an item promote to hard break + indent so
    the chat renderer's soft-break-as-space rule doesn't collapse
    multi-line item content onto one line.
  - The trailing `\n` of a list is the boundary with the next
    block: the soft-break rule promotes `- item\n# heading` (and
    similar pairings) to `- item\n\n# heading`.
  - Empty-item Enter and Backspace at the start of an item content
    both *decrease the item's nesting depth by one* (analog of
    blockquote outdent). For a top-level list item this drops the
    marker (item becomes a paragraph); for a list inside a
    blockquote it leaves the BQ scope intact while ending the
    list. This subsumes the typical "double-Enter exits a list" UX
    and the "Backspace at start of list item joins it" UX without
    a dedicated state flag.
- Soft-wrap.
- Cursor + selection geometry, mouse hit-test, basic keyboard navigation
  (arrows / home / end / doc start / doc end), basic editing (insert text,
  backspace / delete, newline / line break), select-all.

Explicitly *out* of this first phase: setext-heading normalization, ordered
list renumbering on edit, nested-inside-list lists (lists inside list
items), inline code, links, images, thematic rules, tables, HTML,
IME marked-text, word / line-aware delete, and tab-trapped focus
traversal. Each will land as a follow-up.

### Container chain (composability invariant)

Every `RenderBlock` carries a `containers: Vec<Container>` chain
(outermost first). A leaf inside `> > para` carries `[BlockQuote,
BlockQuote]`. When lists land they'll add `Container::ListItem { … }`
entries to the same chain, and the element layer will read indent /
decoration off the chain in the same loop. This keeps the renderer
flat: `inject_empty_paragraphs` doesn't have to know about nesting,
and the element layer doesn't special-case "blockquote inside list"
vs. "list inside blockquote" — it just iterates `containers` in
order. Adding a new container kind is one new variant + one new
`match` arm in `containers_left_indent` / `paint`'s decoration loop.

Blockquote-internal whitespace and editing are the depth-D
generalization of the top-level pairs invariant. `\n\n` at top level
is the structural paragraph-break unit; inside a blockquote at depth
D, the corresponding unit is `\n[prefix]\n[prefix]` where `[prefix] =
"> " × D` (length `2 + 4D` bytes). The two halves are:

- The first `[prefix]` line is the marker-only "middle" of the pair.
  It collapses to one paragraph_gap visually — *no rendered row*.
  Cursor positions strictly inside the pair are forbidden and snap to
  the nearest boundary, the same way the byte between a top-level
  `\n\n` is forbidden today.
- The second `[prefix]` is the start of the new paragraph. When a
  parsed paragraph follows, the leaf claims it; when nothing follows
  yet (the post-Enter transient), `render_blockquote` emits a
  synthetic empty leaf so the cursor has a row to land on.

The same rules drop out across the editor:

- **Soft-break promotion is depth-aware.** A stray `\n` is exempt
  from promotion only if it's part of a complete pair (the new
  `is_paragraph_break_interior` recognizes the alternation `\n
  [prefix] \n [prefix]…` and forbids interior bytes accordingly). Any
  other lone mid-content `\n` — soft breaks across BQ lines, lazy
  continuations — is promoted: `enforce_invariants` inserts
  `[prefix(D)]\n[prefix(D - existing)]` after the offending `\n` so
  the result is a complete depth-D pair. Lazy continuations with
  hard breaks are normalized the same way (the missing prefix is
  inserted on the continuation line). The chat renderer's
  CommonMark soft-break-as-soft-break rendering does diverge from
  the editor's promote-everything rule on paste — that's the only
  pixel-fidelity cost of the simpler invariant.
- **Atomic pair delete.** Backspace at the *end* of a depth-D pair
  removes all `2 + 4D` bytes in one keystroke; Delete-forward at the
  *start* does the symmetric delete. Both subsume the old
  top-level `\n\n` delete and the (now removed) blockquote-pop logic
  under one rule. Inside fenced code-block content `\n`s are
  literal — the pair detector is bypassed there, falling through to
  grapheme delete.
- **Blockquote-aware Enter and Shift+Enter.** `editor::enter` parses,
  finds the deepest blockquote at the cursor, and inserts `\n` +
  `"> " × D` + `\n` + `"> " × D`. `editor::shift_enter` inserts
  `  \n` + `"> " × D` so the hard-break continuation line carries
  the marker — and `render_blockquote` extends the previous paragraph
  leaf forward to swallow the trailing marker line so the cursor has
  a visible continuation row even before the user types content.
- **Prefix normalization.** `enforce_invariants` rewrites every
  blockquote `>` to `> ` (inserting the trailing space if missing),
  unless the cursor is sitting on the byte right after that specific
  `>` — the user may be about to type the space themselves. Code
  content is exempt, same gate as soft-break promotion.

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
