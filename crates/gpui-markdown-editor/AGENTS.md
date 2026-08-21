# gpui-markdown-editor — Agent Development Guide

A WYSIWYG markdown editor as a `gpui-component`-style widget. Targets `crates/eidola-gui` (the chat composer) but is intentionally generic so other gpui applications can drop it in.

## Foundational Goals

1. **Valid, compliant markdown.** The buffer is always valid CommonMark (one exception: two consecutive newlines are preserved as user-visible paragraph separation rather than collapsed). The editor *may* normalize on input (e.g. setext → ATX) but never invents non-spec syntax.
2. **A single editable document.** Selections cross spans and blocks; the user thinks in markdown.
3. **Block composability.** Lists / blockquotes nest arbitrarily; leaf blocks (code, math) are inert islands.

## Target Behavior

The user edits markdown source but sees rich formatting — *except around their cursor*. Delimiters hide when the cursor is outside the construct and reveal (dimmed) when the cursor or an active selection enters it. Formatting applies to content, never to delimiters; the underlying markdown is never modified to achieve a visual effect; copy/paste always produces raw markdown.

## Pixel-fidelity goal with chat rendering

The chat in `crates/eidola-gui` renders posts through this editor's read-only mode and composes through its editable mode — **the two must match pixel-for-pixel**: what the user types is what they see in the transcript after they send. The editor lifts the same `TextViewStyle`-equivalent inputs (paragraph_gap, heading sizes, highlight theme, code-block styling) so callers configure both sides identically. If a future change forces a fork between this editor and any other renderer of the same content, fork forward in both rather than letting the surfaces drift.

## Architecture

A pure transformation pipeline:

```text
EditorState + EditorEvent  →  update()  →  new EditorState
                                                 ↓
                                            parse()  →  SyntaxTree
                                                 ↓
                                            render(state, tree)  →  RenderSpec
                                                 ↓
                                            BlockElement (gpui Element, one per block)
```

- **Per-block gpui `Element`s** rather than one attributed string — each block is its own painter, making full-width code backgrounds and blockquote borders per-block decorations.
- **`display_to_source` per shaped line.** Display strings can be shorter than their source range (delimiters genuinely removed); gpui's `WrappedLine` returns display-byte positions, translated back at hit-test / cursor-paint time via a per-line map.
- **Keyboard input through gpui actions**; IME / dead-key composition via `EntityInputHandler`.
- **Shift-arrow extension state on `Selection::Range::anchor`.**

### Widget shape — state/element split (mirrors `gpui_component::Input`)

A retained **state entity** plus an ephemeral **render element**, not one `Render` entity.

- **`MarkdownEditorState`** (the Entity) holds `state: EditorState`, `focus_handle`, `disabled`, and every cross-frame layout cache (`last_blocks`, `intended_x`, per-block scroll, `marked_range`). Implements `Focusable`, `EntityInputHandler` (IME — why the state half must be an entity), `EventEmitter<MarkdownEditorEvent>`; **not** `Render`. The host owns one, mutates through `set_value`/`clear` (fields are `pub(crate)` — no field-poking), reads via `value()`/`selection()`/`is_empty()`, and subscribes to events.
- **`MarkdownEditor`** (the `#[derive(IntoElement)]` element) is built each frame: `MarkdownEditor::new(&state).style(..).disabled(..)`. Its `render` derives theme colors over caller overrides, syncs `disabled`, registers key/IME handlers via `window.listener_for(&state, ..)`, builds the `BlockElement`s. When `disabled`, only the *mutating* handlers are skipped and IME `replace_*` early-returns; selection/navigation/copy/select-all stay registered and the caret isn't painted — a read-only editor still selects and copies. **A disabled editor's dispatch routes through `update::update_readonly`, never `update::update`**: selection/navigation events apply verbatim (no `enforce_invariants` canonicalization) and any document-mutating event is refused wholesale, so a read-only buffer stays byte-identical to what the host seeded. Load-bearing: model output routinely contains shapes the editable pipeline canonicalizes on the first event, and when a click's `SetSelection` rewrote a read-only post, the host's `sync_bodies` saw the buffer diverge and re-seeded it every frame, resetting the selection — pointer selection was impossible. Regression: `tests/readonly.rs`.
- **Notes-editor fill — `.min_height(px)` + `content_height()`.** `min_height(h)` reserves at least `h` so the editor fills a slot taller than its text; the blank tail is part of the editor's own click surface, and `offset_for_position` resolves a click below the last line to **document end** (a click in the text stays a normal caret placement; an inter-block gap snaps to the nearest line) — the "notes editor" click-to-end feel with no host overlay listener. `content_height()` reports the *natural* laid-out text height (union of painted block bounds, independent of the `min_height` floor, so a host sizing from it has no feedback loop). Paint-derived — one frame behind on first layout; read it in a later-sibling `canvas`'s paint phase for the current frame's value.
- **Host-driven scroll-into-view — `caret_content_y()` / `content_y_for_offset(offset)`.** The editor has **no internal vertical scroll** — it lays out to full `content_height()` and the host scrolls it (the only internal scroll is per-code-block *horizontal*). Both return `Option<(top, bottom)>`: the vertical span relative to the laid-out content top (content-local, unlike the window-absolute IME `bounds_for_range` they derive from). `caret_content_y` answers for the caret, with its wrap-boundary affinity; `content_y_for_offset` answers for **any** source offset — a host revealing something the caret is not at (a search match, a cited passage) — and resolves a boundary **downstream**, since an arbitrary offset carries no affinity. Paint-derived like `content_height()`; independent of the runway. An offset past every laid-out range (document end after a trailing newline) falls back to the last line; `None` means no layout yet, so a host over virtualized content must estimate first and correct once it answers.
- **IME composition — `is_composing()`.** True while `marked_range` holds uncommitted preedit text. This editor emits `Change` on **every** preedit keystroke (the marked text really is in the buffer, and a host sizing to content must follow it) — unlike `gpui_component::InputState`, whose preedit path only notifies. So a host that acts on the buffer's *meaning* on `Change` (searching it, parsing it, sending it) must ask this first and wait: the reader has not chosen those characters yet. The `EntityInputHandler::marked_text_range` half of the IME contract needs a `Window` and is not a host-facing query.
- **Outward events — `MarkdownEditorEvent`**: `Change` (text mutated), `SelectionChanged` (caret/selection moved with **no** buffer change — keyboard navigation), `PressEnter { secondary, shift }` (a submit-intent chord — the editor reports *intent*, the host decides *meaning*), `Focus`, `Blur`. Plain Enter inserts a newline and emits nothing. An edit emits `Change` only; a pure navigation move emits `SelectionChanged` only — a host scrolling the caret into view listens to **both** so keyboard navigation off-screen follows the caret too.
- **Self-contained keymap.** `gpui_markdown_editor::init(cx)` installs the default keymap scoped to the `MarkdownEditor` key context — called once at startup like `gpui_component::init`; hosts don't hand-roll bindings. Chords use gpui's `secondary-` alias (⌘/Ctrl) for clipboard, select-all, undo/redo, semantic bold/italic toggles, and the submit chords; motion/deletion are per-platform tables (macOS ⌥=word / ⌘=line-or-document; Linux Ctrl+arrows=word, Ctrl+Home/End=document, Ctrl+⌫/Del=word-delete) — see the `init` doc comment for why the alias alone would be wrong for motion. The element also maps `gpui_component::input::{Undo,Redo,Cut,Copy,Paste,SelectAll}` action types onto its handlers, so a host's macOS Edit-menu items reach the editor when focused.
- **Test seam.** `MarkdownEditorState::apply_event_for_test(event, cx)` (`#[doc(hidden)]`) drives the internal `update` pipeline from integration tests; tests needing a `Render` root wrap the state in a tiny `EditorHarness`.

## Minimum viable scope (current)

- ATX headings (`# `..`###### `): size + weight, dimmed prefix when the cursor is on the line.
- Bold / italic / strikethrough: trait + dim/hide of delimiters. **⌘B / ⌘I (Ctrl on non-macOS) are semantic toggles**, not delimiter insertion: `formatting.rs` derives independent inline islands from the parser (paragraph/heading, direct tight-list content, and each table cell), expands a collapsed caret to its Unicode word, and applies or removes the target style across explicit selections. A mixed styled/plain selection applies throughout; an entirely styled selection removes only the selected portion, splitting/merging the touched target component while preserving the opposite style. Applying beside an existing target span removes the intervening target delimiter pair and emits one merged span rather than materializing adjacent delimiter bytes. **Inline ancestry is a boundary too**: target delimiters are emitted independently for each strong/emphasis, strikethrough, and link-text context (including nested context stacks), so a selection leaving `**strong**` or `[link](url)` closes the new style before that construct's chrome and reopens it in the surrounding context rather than crossing delimiters. Block chrome and inert islands (list/BQ prefixes, paragraph separators, table pipes/delimiter rows, fences, display math, thematic breaks, mapped embeds) are never wrapped. Resolved escapes/entities are source-complete atomic formatting units (including a leading backslash pulldown omits from its `Text` range); inline images are inert boundaries because bold/italic cannot affect a bitmap and must not demote a standalone image block. Inert images do not vote on removing an existing outer target style, while their non-target inline ancestry remains protected. Every candidate is run through editable canonicalization and reparsed; if canonicalization would further rewrite it, block/task/embed/image classification changes, a protected construct or non-target inline style changes, or the requested style does not cover exactly the semantic selection (including unselected fragments of touched parser atoms), the command is a no-op rather than committing valid Markdown with changed meaning. One event = one undo step; disabled post readers register no mutating handler.
- Fenced code blocks: mono font, full-width rounded background, no soft-wrap (horizontal scroll), dim/hide of fences + info string.
- Blockquotes (`> `): per-level left border bar + cumulative indent via the `containers` chain (any leaf inside N nested blockquotes carries N `Container::BlockQuote` entries; the element applies `N * blockquote_indent` padding and paints N bars; code backgrounds inset inside the indent so the bar stays visible).
- **Lists** (unordered `-`/`*`/`+`, ordered; nested in blockquotes and in list items). Each leaf carries a `Container::ListItem` chain entry per enclosing item, recording the item's `marker_byte_len` and the list's `list_max_marker_text` (the widest marker — so every item aligns at one content edge). Marker bytes are always hidden from the shaped line; the marker glyph paints as a `MarkerOverlay` right-aligned in the indent strip. Continuation lines hide the cumulative ancestor indent so wrapped text shapes from the first line's column. **Tab** nests the cursor's item under the previous sibling (no-op without one); **Shift+Tab** dedents, falling through to "drop the marker" at depth 0. Unordered overlay glyph is `•` outside / the raw bullet char inside; ordered digits stay visible always and **renumber automatically** (`start + index` regardless of what was typed).
  - **CommonMark interaction: an ordered list can't open mid-item unless it starts at 1** (pulldown follows the spec's interrupt rule), so Tab-nesting an ordered item rewrites its marker to `1. `; renumbering then fixes siblings. General principle: structural edits must produce source pulldown actually parses as the intended structure — the canonicalization passes operate only on what pulldown sees.
  - **Marker-space injection is `-`/`+` only** (`update::inject_unordered_marker_space`): typing `-foo` at a fresh top-level line salvages the missing space CommonMark requires, so the user gets a bullet. **`*` is excluded** — it is an emphasis delimiter as much as a bullet, and the pass sees only the prefix: `*I` is the live prefix of both `*Italic*` and a would-be `* Italic` bullet, and injection fires on the *second* keystroke, before any closing `*` exists, so no lookahead can disambiguate. Emphasis opening a paragraph wins; a `*` bullet is typed with its space. Pinned by the `typing_*_marker_space` / `*_italic_at_paragraph_start_*` family in `tests/behavior.rs`.
  - **Whitespace rules `enforce_invariants` enforces in lists:** lists are rendered tight *between* items (`\n\n+` between items collapses to one `\n`); *inside* an item `\n\n` is a paragraph break (multi-paragraph items are first-class; the loose-list spacing divergence from the chat renderer is the documented cost). Two consecutive hard breaks collapse to a paragraph break in the same scope (what makes Shift+Enter twice the "paragraph break inside this item" gesture). No lazy continuations: continuation lines carry exactly the item's cumulative indent and the preceding line ends with a hard break; editing `9.` → `10.` re-aligns continuations. Soft breaks within an item promote to hard break + indent. Empty-item Enter and Backspace at item-content start both *decrease nesting depth by one* (top level: drop the marker) — subsuming "double-Enter exits" and "Backspace joins" without a state flag.
- Soft-wrap; cursor + selection geometry; mouse hit-test; basic navigation and editing; select-all.
- **Pointer selection — click granularity + drag.** `click_count`: single places a caret (Shift extends **keeping the original anchor**), double selects the word, triple the line; the granularity is recorded as a `SelectMode` so a drag extends by the same unit, growing symmetrically around the clicked unit (native macOS behavior). Word bounds via `word_range_at` (Unicode segments); lines `\n`-delimited. Works read-only too. **The drag is window-global**: gpui fires the div's `on_mouse_move` only inside the element's *visible* bounds, so a drag on a post taller than the window froze at the fold — the element registers window-global `MouseMove`/`MouseUp` listeners via a hitbox-free `canvas`, **unconditionally every frame** (the `is_selecting()` guards live inside the handlers): gpui dispatches against the last-painted frame, so a gated registration would miss a fast first move that outran the mouse-down's repaint. Host autoscroll is the host's job: `is_selecting()` + `drag_extend_to(window_pos, cx)` are the read/drive pair. Test seams: `begin_selection_for_test` / `extend_selection_for_test`; `word_range_at`/`line_range_at` unit-tested; the hit-test + global routing exercised by the eidola-gui driver.
- **Word-granular motion/extension** (`MoveWordLeft/Right` + `Extend*`, Unicode word boundaries).
- **Line/word-aware deletion** (`DeleteWordBackward/Forward`, `DeleteToLineStart/End`); inside a structural chain the target clamps to the line's chain-prefix end so `> ` / `- ` / continuation-indent bytes survive — deletion affects content, not structure.
- **Visual-position-aware vertical navigation.** Up/Down (+Shift) route through `vertical_move`, which consults the previous frame's `LaidOutBlock` layout to step one wrap-row in display coordinates: navigation respects soft-wrap rows, and a long → short → long round trip lands at the original visual column via an `intended_x` anchor that survives the streak (cleared on any non-vertical event). Headless tests with no layout fall back to source-byte `MoveUp`/`MoveDown`.
- **Caret wrap-boundary affinity (`WrapAffinity`).** A byte offset on a soft-wrap boundary is both the end of one row and the start of the next; gpui always resolves it to the upper row's end, which mis-rendered boundary carets and stalled repeated Down. `wrap_affinity` (`Downstream` — lower row's start, the default, set by Down/Home/Right/edits/IME — vs `Upstream`, set by End) is a transient signal read by the caret renderer, `caret_content_y`, `visual_move_caret`, and `visual_line_bound`; `visual_move_caret` sets it from where a move landed. Only matters at a boundary.
- **Display-line-aware Home/End** (+Shift, and the `cmd-left/right` aliases): read the caret's wrap-row from the layout and map that row's left/right edge to a source offset — landing on the *display* line's edges, not the `\n`-delimited source line. On a blockquote/list line the shaped text begins past the hidden chain prefix, so Home lands on the visible content edge; results route through `SetSelection`, whose snap preserves chain-prefix skipping. Home/End clear `intended_x` and set affinity (Home → `Downstream`, End → `Upstream`). Headless fallback: source-line `MoveLineStart/End`. Pure navigation — no undo step.
- Inline code: mono family + chip background, dim/hide of backticks; multi-backtick spans work. **No per-run font size** (gpui's `TextRun` has none — a line shapes at one size), so the lever is `MarkdownStyle::inline_code_font_family` — hosts pair a low-x-height serif body with an x-height-compatible mono (the app: Newsreader body, Courier New inline, Menlo fences). **The chip needs the explicit background pass**: `WrappedLine::paint` draws glyphs only; `element.rs` paints `TextRun::background_color` in a separate pass before selection quads. Defaults to `theme.accent` (the same chip `TextView` paints).
- Inline links: link color + underline, dim/hide of `[`/`](url)`; nested styling inside link text composes.
- Thematic breaks: a thin rule painted as a block decoration; bytes hide/dim per cursor.
- GFM task list items: parser sets `task: Option<bool>` on `ListItem` from `Event::TaskListMarker`; `marker_range` stays `- ` (indent math unchanged); the renderer hides the `[ ] `/`[x] ` bytes and the overlay paints `☐ `/`☑ `.
- CommonMark §2.4 backslash escapes and §2.5 entity references: each occurrence becomes a `Substitution` (cursor outside — resolved literal) or a dimmed `InlineRun` (cursor inside — raw bytes). Driven by `escapes.rs`, a source-byte scanner (pulldown's `Event::Text` is lossy for these), skipping verbatim contexts.
- **LaTeX math** — inline `$x$` and display `$$..$$`. Inline, cursor outside: bytes hide, a `MathOverlay` is typeset (`math::typeset`), measured, and substituted with a width-matched NBSP run so text shapes around it; paint places the math baseline-aligned. Cursor inside: dim-delimiter mono fallback. Display math becomes `BlockKind::DisplayMath { content_range, edit_mode }` via two parser paths feeding one helper: block-level `$$\n..\n$$` (the pulldown fork's `parse_display_math_block`, terminated by a `$$` line or EOF) and **sole-paragraph promotion** (a paragraph whose only content is one inline `$$x$$`). Two modes: display (cursor outside — typeset, block height = math size) and edit (cursor inside — `$$` dims, LaTeX in mono; the block reserves `max(natural edit, natural display)` so toggling doesn't shift content). Editing inside follows fenced-code rules (`is_in_verbatim_region`): literal `\n` + chain prefix on Enter, literal Tab, blank lines pass through; the first Enter in an unterminated `$$` runs `auto_close_math_edit` (injects the closer). Boundary cursors count as inside (inclusive overlap) — the "click to edit" feel. KaTeX fonts auto-register on first paint (`register_katex_fonts`; hosts may call it at init).
- **Images** (`![alt](url)`) — structurally like math. Cursor outside: an `ImageOverlay` — load via `image::load`, cap height to `INLINE_HEIGHT_FACTOR × line_height`, NBSP-substitute, paint centered on the row. Cursor inside: dim-delimiter + visible alt. A sole-Image paragraph promotes to `BlockKind::Image` (same promotion + inclusive-overlap rules as DisplayMath); display mode scales to content width and paints via `window.paint_image`, edit mode reserves `max(natural edit, natural display)`. Loading is async: `Loading` reserves a placeholder and invalidates on resolve; `Failed` falls back to the inline-run pair so the user sees the broken URL. `http(s)`/`file`/absolute paths via gpui's image cache; relative/embedded need a host `AssetSource`.
- GFM pipe tables — see [Tables](#tables--gfm-pipe-tables).
- Embed blocks (`{{ embed N }}`) — see [Embed blocks](#embed-blocks--the-block-plugin-mechanism).

Explicitly *out*: setext normalization, HTML, IME marked-text, reference-style images, image titles/data-URIs/links, rich-content paste from HTML/RTF (needs a gpui clipboard-mime extension — see [Clipboard pipeline](#clipboard-pipeline)).

### Tables — GFM pipe tables

The **aligned-source hybrid**: the buffer stays *minimal canonical GFM* at all times (`| a | b |` single-space cells, compact `| --- | :-: |` delimiter row — never physical space padding), and column layout is **display-only** — the element lays each row out as side-by-side cell boxes over the same source bytes. (A structured-grid widget would invert the crate's one-selectable-document foundation; flip-to-raw is visually violent — the hybrid keeps a table a table in both modes with every byte honest markdown.)

**The model** (`src/table.rs`): a table is a **structural leaf block** like a fenced code block — its internal single `\n`s are row separators, exempt from soft-break promotion (`promote_soft_breaks` consults `table_ranges_in_tree`; the final newline still promotes at the block boundary). `TableGeometry` (rows × cells, line ranges, trimmed/untrimmed ranges, alignments) is computed **from source bytes** by an unescaped-pipe line scanner — not pulldown's cell events, which omit the delimiter row, synthesize degenerate ranges, and report untrimmed content — while pulldown stays the authority on *whether* a table parses (the scanner only runs inside a parsed `NodeKind::Table`). V1 is **top-level tables only**: nested tables parse (newlines protected) but render as raw source and take no table editing rules.

**Box-model layout** (`element.rs::layout_table` — one path for both modes, fitting and overflowing): every cell is its own shaped `LaidOutLine`, wrapped at its column's width; a table row stays **one physical GFM line in the buffer, always** — wrapping is purely visual. Column sizing is HTML-auto-inspired: per column min-content (widest token, measured with real styles) and max-content; if Σmax fits, no wrapping; else columns shrink proportionally to their `(max − min)` excess, flooring at min; if Σmin overflows, the table takes the code block's horizontal-scroll treatment. Row height = tallest wrapped cell; alignment colons shift non-wrapped cells (wrapped cells left-align).

**Two render modes** (`BlockKind::Table { geometry, edit_mode }`, the display-math click-to-edit pattern):

- **Display** (cursor outside; also the read-only path): chrome hidden; cell boxes with a fixed gutter; header at `table_header_weight` (Medium); hairline rules at row-band boundaries (full color under the header, 0.45× between body rows) — booktabs: no boxes, no vertical rules, no zebra.
- **Edit** (cursor inside): chrome bytes shape as real dimmed single-row fragments in the gutters between boxes, so every byte keeps a true caret position; the delimiter row renders only here, dash cells stretched to column width (display-only) except on the caret's own row, where raw bytes stay visible for colon editing.

**Geometry consumers are x-aware** for the multiple-boxes-per-y-band shape: `offset_for_position` picks among y-hits by horizontal proximity; `visual_move_caret` tie-breaks equal vertical distance by x (Down from column 2 lands in column 2). Within a row, cells precede chrome fragments and claim one byte past their content end, so the caret at a cell's typing position paints at the text end, not the box edge. Regression: `tests/table_wrapped.rs`.

**Canonicalizer** (`normalize_tables`, after `normalize_lists`): every row **not hosting the caret** rewrites to canonical form — single-space padding, short rows padded, the delimiter row regenerated from its alignments at the header's width. The caret's row is left alone and snaps canonical when the caret leaves. Idempotent; `update_readonly` never runs it.

**Allowed caret positions**: between-cell chrome is forbidden (`is_table_chrome_position` in `analysis::is_forbidden_position`). Allowed = cell content plus the strict interior of a cell's untrimmed segment (a just-typed trailing space keeps the caret until normalize trims it). This is what makes arrows hop cell-to-cell and clicks land in cells.

**Recoverability — a table broken mid-edit never loses its line structure while being repaired.** The rule: *the recoverability authority is parser history, never shape heuristics* — nothing anywhere answers "does this look like a table?" outside the real parser (and the scanner, which runs only inside a parsed table). Three layers: (1) **`update::TableGuard`** (session-scoped, threaded through every editable dispatch): the breaking edit itself is protected because the *pre-event* parse had a table at the selection — suppressing soft-break promotion in the caret's block for that event, so a damaged delimiter row degrades to one multi-line text block instead of exploding into paragraphs; the guard then arms (pre-table ∧ post-not), persists by block-range overlap while the caret stays, and clears on re-parse-as-table or when the caret moves on (from there the recovery is **undo** — every structural op is one step). Deliberately transient — a document saved broken reloads as ordinary text. (2) **Targeted hardening**: the delimiter dash floor (deletes that would remove a delimiter cell's *last* dash are refused; colons stay freely deletable; dissolving a table is an explicit act — delete the delimiter row's line), plus the structural-`|` family. (3) A table that doesn't parse takes no table rules at all — honest plain text until repaired.

**Keystroke map** (all in `update.rs`, edits computed in `src/table.rs`; every op is one undo step):

| Keys | In a table |
|------|-----------|
| `\| a \| b \|` + **Enter** | The scaffold: a pipe-shaped paragraph line (starts+ends with `\|`, ≥1 non-empty cell, top-level, non-verbatim) grows the delimiter row + one empty body row; caret in the first body cell |
| **Tab** / **Shift-Tab** | Next / previous cell (delimiter row skipped) — selects the target's content (typing replaces); Tab past the last cell appends a row |
| **Enter** | New empty row below (header ⇒ first body position); on the last body row with all cells empty ⇒ exit into a paragraph |
| **`\|`** | Insert a column boundary at the caret, **table-wide** (works over a Tab-selected cell). On the delimiter row the dash cell splits into two valid cells distributing its colons. Escape is backslash-run parity: `\\` + `\|` types a literal pipe |
| **Backspace** at cell start (k > 0) | Remove that column boundary table-wide (merged delimiter cell keeps the left alignment) — the exact inverse of `\|` |
| **Backspace** in an all-empty body row | Delete the row (caret to previous row's last cell) |
| **Backspace** at a body row's first cell start | Hop to the previous row's last cell, no deletion (the header's first cell falls through so the table can merge into the preceding block) |
| **Delete-forward** at cell end | Hop to the next cell's start |
| **`:`** on the delimiter row | Alignment is the colons — dash cells are editable content; normalize re-canonicalizes when the caret leaves |
| **←/→** | Hop across chrome cell-to-cell (forbidden-position snapping) |
| **⌥⌫ / ⌥Del** | Word-delete clamped to the caret's cell |
| **Paste** | Cell-safe splice: newlines → spaces, unescaped pipes → `\|` |

A collapse to plain text is always honest: any edit that breaks the GFM shape stops parsing as a table and renders as the raw paragraphs it now is; re-completing the shape restores it.

**Deliberately out**: tables in blockquotes/lists (raw-source fallback); a column-*move* command; multi-line cells via `<br>` (no HTML); pasting a table into a cell as cells; alignment inside *wrapped* cell boxes.

Tests: `tests/table.rs` (the keystroke gate), unit tests in `src/table.rs`, `tests/readonly.rs` (`editable_pipeline_no_longer_rewrites_tables`), table visual cases; the eidola-gui driver scene `markdown_table` is the interactive fixture.

### Embed blocks — the block-plugin mechanism

The editor's one **opaque block plugin** (`src/embed.rs`): the host supplies an `EmbedMap` of ordinals → markdown content, and a top-level paragraph whose entire content is the marker `{{ embed N }}` — with `N` mapped — renders as a single **atomic** block: the mapped markdown shown read-only inside a quiet quote-like container. The editor never learns what the content *means* — the ordinal is the shared key between buffer text, map, and host bookkeeping (in Eidola: reference ordinals; this crate carries no such symbols).

**Lexical rules** (`embed::parse_embed_text`, unit-pinned): the paragraph's whole content, modulo leading/trailing spaces/tabs, must match `"{{" WS* "embed" WS+ DIGITS WS* "}}"` (`WS` = space/tab; `DIGITS` = non-negative decimal `u64`, leading zeros allowed). Canonical spelling `{{ embed N }}` (`embed_marker(n)`). **The recognition rule is deliberately duplicated in `eidola-common`** (`embed::{parse_embed_marker, embed_marker_spans}` — this crate can't depend on eidola-common, and app-core can't depend on gpui). Lockstep is held by the corpus test `crates/eidola-gui/tests/embed_lockstep.rs` (extend the corpus with any case that motivates a change) plus identical lexical cases in all three crates. One parser fact the corpus pinned: this crate's parser has **indented code blocks disabled**, so an indented marker line is an ordinary paragraph and still promotes.

**Everything else is literal text** — which is the escaping story: an unmapped ordinal is plain editable text (also how a marker looks before its reference exists); an inline occurrence is literal; a marker inside BQ/LI/fence is literal (v1 embeds are top-level, like tables); the literal text of a *mapped* marker is typed by breaking the pattern (`\{{ embed 1 }}` — the matcher reads raw source bytes). **The marker's whole line is protected**: a splice landing on either edge (typed char, IME commit, paste) is padded with exactly the paragraph separator it is missing, so the bytes open a fresh paragraph beside the block rather than dissolving it (`update::protect_embed_line`); newlines the incoming text carries count toward the separator, so a host call that pads its own insertion is never padded twice. Deleting the block still deletes the marker string.

**Round-trip:** the buffer always contains the plain marker text — the map is render-time state only (`set_value` preserves it; `value()`/copy see clean markdown). Typing a mapped marker re-materializes the block.

**Atomicity** (the hidden-chrome machinery from tables/list indents): positions strictly inside a mapped marker are forbidden — `analysis::is_forbidden_position_with` and the `next/prev/nearest_allowed_position_with` variants consult `EditorState.embeds` (**editable-pipeline code must call the `_with` variants with `state.embeds`**; the map-less wrappers serve tests and embed-free contexts). Arrows hop the block in one step, clicks/selection endpoints snap to its edges, Backspace at the trailing edge / Delete-forward at the leading edge / word-deletes remove the whole marker in one step. A selection spanning the block deletes normally. The canonicalizer never touches a marker line.

**Render + element:** `render::promote_embeds` (a post-pass after the recursive walk) converts matching top-level paragraphs into `BlockKind::Embed { ordinal }` with one hide over the whole range — no edit mode, in both `render` and `render_readonly`. The element layer (`layout_embed`, shared by measure + prepaint like `layout_table`) parses + readonly-renders the mapped content and stacks sub-blocks with the editor's own shaping/spacing helpers, clipped to the block bounds. **Chrome that isn't shaped text is re-emitted explicitly** (only the *shaping* path is shared): list markers via `emit_embed_marker_overlays` (right-aligned at the level's content edge — also what makes checkboxes paint), a rounded panel per fenced code block, table/thematic-break rules, nested blockquote bars, inline **and** display math (`place_embed_inline_math`; rows carrying tall math reserve the editor's own overshoot). Blockquote `>` markers deliberately not — embed content renders through `render_readonly`, where they never appear. Fidelity notes: an embedded table takes no horizontal scroll (clips at the mask); a code block gets one flat panel (the fence rows the frame needs are collapsed); **images render as raw markdown source** (`strip_embed_image_overlays` un-hides their bytes so nothing silently vanishes — resolving one needs `Window::use_asset`, unavailable in the measure closure). **Caret affinity at the edges:** the marker's shaped line is fully hidden (zero width), so both edges would paint carets in the same place; the element keeps the leading edge at the container's top-left and repositions a caret at `source_range.end` to the container's **bottom-right** — typing there opens the paragraph below the embed, which is where it renders.

**Host API:** `set_embeds(entries, cx)` (replaces the map; buffer untouched, no `Change` — the view re-renders) + `embeds()`. `set_embeds` also **re-snaps the selection against the new map** — a caret legally parked inside a *literal* marker becomes forbidden-interior the moment its ordinal is mapped, and without the snap the next insertion would splice into hidden marker bytes. `MarkdownEditor::on_embed_click(|ordinal, window, app| …)` — a click anywhere on a rendered embed fires with the ordinal (the click-to-navigate seam); the hit-test reads the ordinal the render recorded on the painted block (`LaidOutBlock::embed_ordinal`), never re-deriving ranges per click. Both editable and read-only editors render embeds and fire the callback.

**Placing and un-placing a marker** are a symmetric host-call pair, so a host never splices buffer bytes itself: `insert_embed_marker(ordinal, cx)` writes the canonical marker as its own top-level paragraph at the caret — padding with exactly the missing blank lines, swallowing adjacent spaces/tabs, replacing an active selection; `remove_embed_marker(ordinal, cx)` deletes the **recognized** block plus one paragraph separator so the surrounding prose rejoins cleanly. Both route through the normal update pipeline — one undo step, one `Change`. Removal reads the *live* map, so a host clearing its own bookkeeping must call `remove_embed_marker` **before** dropping the ordinal; an unmapped or fence-defused ordinal is not a recognized block and removal is a no-op (the literal text stays). Insertion inside a verbatim region lands as literal text — the documented honest degradation.

**Test seam:** `LaidOutBlock::embed_content` records what the container painted each frame (text pieces with origins — marker glyphs included, since they appear in no shaped line — plus code panels, math origins, rules, bars); read via `debug_embed_content(ordinal)`. Tests: `tests/embed.rs` (the keystroke gate — promotion, degradation, line protection, atomic navigation/deletion, map lifecycle, readonly, the real-layout click, and the embedded-markdown fidelity set), the paired control-vs-embedded visual corpus (`EMBED_AUDIT_CORPORA`; control rendered read-only so the cursor doesn't flip it to edit mode; `VISUAL_FILTER=embed_audit`), lexical unit tests, the cross-crate lockstep corpus, and the insert/remove pair pinned in `tests/highlight.rs`.

### Highlight ranges — the second opaque plugin

The other host plugin (`src/highlight.rs`), deliberately much smaller: the host supplies `(source-byte range, opaque u64 key)` pairs via `set_highlights_in(layer, entries, cx)`, and the editor paints a quiet wash behind the covered text (in Eidola the base layer's key indexes the incoming-reference list).

- **Highlights are inert decorations** — the whole design constraint: not document content (`value()`/copy never see them), no forbidden positions, never touch the update pipeline or canonicalizer, setting them emits no `Change`. They live on the entity (`MarkdownEditorState`), **not** `EditorState`: the pure pipeline never consults them; only paint and the click hit-test do. This is why the plugin needed none of the embed atomicity machinery.
- **Layers.** One `HighlightSet` per `HighlightLayer` (`HighlightLayers`), so unrelated kinds of decoration (a quoted passage; a phrase being searched for) do not merge into one wash, one color and one click target. An enum rather than an index, so every layer has a color by construction and no out-of-range layer can be named. Layers paint bottom to top in `HighlightLayer::ALL` order — `Base`, `Overlay`, `Accent` — each in its own color, so an upper layer's wash sits *on top of* a lower one. `set_highlights_in` replaces one layer and leaves the rest; `highlights_in(layer)` is the read half a host needs for the compare-before-set guard (setting notifies unconditionally). `set_highlights` / `highlights()` are the `Base` shorthands.
- **Overlaps merge visually, within a layer**: `HighlightSet::merged_ranges` coalesces (adjacent ranges join), and `build_highlight_quads` paints one wash per merged range with the same per-line geometry as a selection — overlapping highlights never stack alpha. Empty/inverted ranges are dropped at construction. Quads paint **before** selection quads (a selection over highlighted text reads on top); in code blocks they follow the delimiter/content mask split.
- **Click routing is `Base` only**: `MarkdownEditor::on_highlight_click(|keys, ..| …)` fires with the keys of **every** base-layer range containing the clicked offset (`keys_at`, insertion order). Upper layers are inert paint and can never fire it — a decoration is not a target. Only a *plain* click fires: the press arms `highlight_press` on a single unmodified click landing on base-layer highlighted text, and mouse-up fires only if the selection is still collapsed — **a drag across a highlight selects normally and never navigates**. Registered read-only too.
- **Styling**: `MarkdownStyle::highlight_layer_color(layer)` — total by construction over `highlight_color` (base), `highlight_overlay_color`, `highlight_accent_color`, with low-alpha warm defaults per theme mode ramping from quiet to prominent. Keep the base fainter than `selection_color`.

Tests: `tests/highlight.rs` (buffer/selection untouched, typing/selecting over highlights, real-layout click routing, layer independence, an upper layer routing no click); merge/keys/layer math unit-tested in-module; the `highlight_wash_*` visual cases are the only pixel coverage of the wash itself.

### Container chain (composability invariant)

Every `RenderBlock` carries `containers: Vec<Container>` (outermost first) — a leaf inside `> > para` carries `[BlockQuote, BlockQuote]`; list items add `Container::ListItem` entries to the same chain. The element layer reads indent/decoration off the chain in one loop, so nothing special-cases "blockquote inside list" vs the reverse. Adding a container kind is one variant + one arm in `containers_left_indent` / the decoration loop.

Blockquote-internal whitespace is the depth-D generalization of the top-level `\n\n` pair: inside a blockquote at depth D the paragraph-break unit is `\n[prefix]\n[prefix]` (`[prefix] = "> " × D`). The first prefix line is the marker-only middle — collapses to one paragraph_gap, no rendered row, interior positions forbidden (snap to boundary); the second starts the new paragraph (a synthetic empty leaf is emitted post-Enter so the cursor has a row). The same rules drop out across the editor:

- **Soft-break promotion is chain-aware**: a stray `\n` is exempt only as part of a complete pair (`chain_pair_shape`); any other lone mid-content `\n` is promoted — `enforce_invariants` inserts the missing prefix bytes per `chain_continuation_prefix`. (The chat renderer's soft-break-as-space rendering diverges on paste — the one pixel-fidelity cost of the simpler invariant.)
- **Atomic pair delete**: Backspace at a pair's end / Delete-forward at its start removes the whole pair in one keystroke, per `chain_pair_shape(chain)`. Inside fenced code, `\n`s are literal — the detector is bypassed.
- **Blockquote-aware Enter/Shift+Enter**: Enter inserts `\n` + prefix + `\n` + prefix at the deepest blockquote; Shift+Enter inserts `  \n` + prefix (and `render_blockquote` extends the previous leaf to swallow the trailing marker line so there's a visible continuation row before content is typed).
- **Prefix normalization**: every `>` rewrites to `> ` unless the cursor sits right after that `>` (the user may be typing the space). Code content exempt.
- **Unterminated-fence-aware classification**: `is_in_fenced_code` treats the EOF position of an unterminated fence as inside. Every cursor-driven query funnels through it.
- **Auto-close-fence on Enter**: `auto_close_fence_edit` fires before regular Enter routing inside an unterminated fence — injects a matching closer below (fence char/length, chain prefix per line), lands the cursor on a body row. After it, rules read off `is_in_fenced_code` without unterminated ambiguity.
- **In-fence Enter emits chain prefix** (`\n` + `chain_continuation_prefix`, not bare `\n`) so the new code row stays in its enclosing scope.
- **Empty-BQ Enter outdents** (mirror of empty-LI): `empty_bq_paragraph_exit_edit` drops the innermost BQ scope; wired before `enter_insertion`.

### Chain-aware invariants (the helper family)

Every byte sequence that introduces a continuation line is built by walking the chain outermost-first, emitting each container's prefix in order (`[LI(2), BQ, LI(2), BQ]` → `"  >   > "`). `analysis.rs` exposes the canonical helper family — **use these; don't compute prefixes locally** (raw `\n` boundaries or hand-built `"> "` strings in chain-aware context are a bug class we've repeatedly fixed by migrating here):

| Helper | Use when… |
|--------|----------|
| `chain_continuation_prefix(chain)` | You need the bytes introducing a continuation line (Enter/Shift+Enter inserts, soft-break promotion, render's hide pass) |
| `chain_continuation_prefix_bytes(chain)` | Same length without allocating |
| `chain_outer_prefix_bytes(chain)` | The bytes contributed by containers *above* the innermost — where to insert/strip indent without disturbing outer markers (Tab/Shift+Tab) |
| `chain_pair_shape(chain) -> (blank, content)` | Emitting or recognizing a structural pair (`\n{blank}\n{content}`; ends-in-BQ → symmetric, BQ-then-LIs → asymmetric, no BQ → `("", full)`) |

These power `enter_insertion`, `line_break_insertion`, soft-break promotion, list indent/dedent edits, atomic pair-delete, forbidden-position detection, and the render side's `chain_for_position` / `hide_chain_continuation_prefix` / `merge_hard_break_continuations`. A new shape variant belongs here, same naming pattern. `analysis::enclosing_containers_at` is the single source of truth for "what containers enclose byte X"; `chain_for_position` delegates to it so the two analyses can never disagree.

### Render walker pipeline

`render::render` is a pipeline, not a tree walk; the post-pass order is load-bearing:

1. Recursive walk (`render_node` → per-construct renderers) → flat `Vec<RenderBlock>`.
2. `inject_empty_paragraphs` — synth empty Paragraph leaves for trailing positions and inter-block breaks pulldown didn't claim (chains from `chain_for_position`).
3. `merge_hard_break_continuations` — merge the two blocks pulldown splits a `  \n` + prefix-only trailing line into, matching the with-content case.
4. `hide_chain_continuation_prefix` (per block) — final chain-driven hide catching alternating-chain prefix bytes the per-container hides miss.
5. `merge_hidden_ranges` (per block) — normalize `hidden_ranges` to sorted, non-overlapping.

The doc comment on `render::render` carries this same list; keep both in sync.

## Clipboard pipeline

Three event variants, one `update.rs:paste` router (which drops any active selection first, so chain/verbatim analysis runs on the post-deletion buffer):

| Event | Trigger | Behavior |
|-------|---------|----------|
| `InsertText(String)` | IME commit, programmatic | Splice raw at cursor; no paste transforms |
| `Paste { text, internal }` | Cmd+V | Markdown-aware; `internal: true` when the clipboard metadata matches `CLIPBOARD_SENTINEL` (set on every copy/cut) |
| `PastePlain { text }` | Cmd+Shift+V | Plain splice — bypasses markdown parse and soft-break collapse |

Routing: `Paste` → `verbatim_paste` in a verbatim region, else `markdown_paste`; `PastePlain` → `verbatim_paste` in a verbatim region, else `insert_text`.

- **Verbatim paste** (cursor in a fence or `$$..$$`): chain-prefix injection after every embedded `\n`, plus **fence widening** — if the pasted bytes contain a run matching the enclosing fence's closer, opener and closer both widen to `max_run + 1` in lockstep so the paste can't close the construct (`analysis::fence_with_delimiters_at`; `$$` has no longer form — lands verbatim). Widening + splice compose into one `SourceEditList` so cursor remapping is a single pass.
- **Markdown-canonicalize paste** (elsewhere), three transforms in order: (1) **canonicalize** — parse the text and replace each `NodeKind::SoftBreak`'s `\n` + following chain prefix with a space (pulldown emits SoftBreak *only* for genuine in-paragraph soft breaks — the parser-blessed enumeration byte-scanning can't reproduce, since a ListItem's range swallows its trailing `\n`); skipped when `internal` (already canonical); (2) **block-boundary padding** — if the first/last top-level node is non-Paragraph, pad with `\n\n` (or `\n`) so the construct lands on its own line; (3) **chain-prefix injection**. The deliberate divergence: a hard-wrapped paragraph from any source collapses to one paragraph (CommonMark's soft-break rendering); plaintext where breaks are meaningful uses Plain paste.
- **Plain paste**: bytes splice raw; each `\n` becomes a paragraph break post-splice (via `promote_soft_breaks`). Sentinel ignored — the user chose plain semantics. Markdown markers are *not* pre-escaped (a "paste as literal text" mode that escapes them is a follow-up).
- **CRLF normalization**: `normalize_line_endings` collapses CRLF/CR to LF before bytes reach `update`, for both paste variants.
- **Sentinel**: `copy`/`cut` tag clipboard writes with `ClipboardItem::new_string_with_metadata(text, CLIPBOARD_SENTINEL)`; the sentinel is crate-namespaced (no Eidola symbols).
- **Deferred — rich-content paste**: real HTML/RTF needs a gpui `ClipboardEntry::Html`/mime variant that doesn't exist (gpui flattens to one plaintext flavor); then an HTML → Markdown walker feeding the canonicalize path. Heuristic markdown detection on plaintext was rejected (surprising false positives; Plain paste is the better answer).

### Host-driven commands + the context-menu gesture

Small additive seams so a host builds its own menu without duplicating any of the above:

- **`MarkdownEditorState::perform(EditorCommand, window, cx)`** runs Cut/Copy/Paste/SelectAll programmatically — the same code the keymap actions reach, minus the responder chain a read-only editor is not on. Cut/Paste are refused while `disabled`.
- **`can_perform(EditorCommand, cx)`** is `perform`'s **enablement twin** — a host that re-derives conditions itself will eventually advertise a verb `perform` then declines. Cut/Paste need editable; Cut/Copy a non-empty selection; Paste **text on the clipboard** — the condition a host cannot see, and the one that bit (an unconditional "Paste" row on an empty clipboard did nothing). SelectAll has no precondition.
- **`append_at_end(text, cx)`** places the caret at document end and inserts — the seam for "the host decided these keystrokes belong here" (the app's type-to-compose). Normal update pipeline (one undo step, one `Change`); the caret move **collapses** any selection rather than replacing it (the intent is *append*). Refused while `disabled`. Pinned by `append_at_end_lands_after_the_text_with_the_caret_behind_it`.
- **`MarkdownEditor::on_context_menu(cb)`** reports a right mouse-down with the window position a menu should open at. The editor opens nothing itself but places the caret first, on the platform convention: a press inside the selection leaves it (that's what Cut/Copy act on); outside collapses to the clicked offset (what makes Paste land where pointed). Registered read-only too. **The callback fires from inside the editor's own `update`** — a host must not read the editor entity synchronously in it (re-entry panic); defer by a turn (the app's `space_view::context_menu` does).

## Module map

| File | Purpose |
|------|---------|
| `state.rs` | `EditorState` (markdown + selection), `Selection` |
| `event.rs` | `EditorEvent` — every user action |
| `formatting.rs` | Parser-driven semantic bold/italic toggle planning, candidate reparse/verification |
| `update.rs` | Pure `update(state, event) -> state` |
| `parser.rs` | pulldown-cmark walker → `Vec<SyntaxNode>` |
| `syntax.rs` | `SyntaxNode`, `NodeKind` (only what we render) |
| `render.rs` | Pure `render(state, tree, style) -> RenderSpec` |
| `render_spec.rs` | `RenderSpec`, `RenderBlock`, `InlineRun`, `InlineStyle` |
| `style.rs` | `MarkdownStyle` — derived from `gpui_component::Theme` |
| `table.rs` | Table model: `TableGeometry` scanner, structural edit computations, chrome-position detection |
| `embed.rs` | Embed plugin: `EmbedMap`, the lexical rule (lockstep-duplicated in eidola-common), `embed_blocks` scanner + atomicity queries |
| `highlight.rs` | Highlight plugin: `HighlightLayer`, `HighlightLayers`, `HighlightSet`, `keys_at`, `merged_ranges` |
| `element.rs` | `BlockElement` — paints one block, owns a `display_to_source` map per shaped line |
| `editor.rs` | `MarkdownEditorState` + `MarkdownEditor` + `init` (keymap) |
| `escapes.rs` | §2.4/§2.5 source-byte scanner → `ResolvedSpan`s |
| `math.rs` | RaTeX adapter: `register_katex_fonts`, `typeset(latex, mode) -> MathLayout`, `MathLayout::paint` (native gpui paint ops) |
| `image.rs` | Image-cache adapter: `load` (async `Loading`/`Failed`), `inline_size`/`block_size` caps, `paint` |
| `bin/demo.rs` | Standalone demo window |

## Theme integration

The editor carries **no color palette of its own**. `MarkdownStyle::from_theme` derives every color from `gpui_component::Theme`, and the element's `render` re-derives **every theme-sourced color each frame** so a mode flip can't leave any stale — color fields are therefore *not* caller-overridable across frames; the typography knobs (font size, families, heading callback, paragraph gap, `list_item_gap_factor`, `inline_code_font_family`) are.

## Vertical rhythm

Inter-block spacing is symmetric: each block reserves half its factor above and below, so two stacked paragraphs are one `paragraph_gap` apart. Blocks inside a `Container::ListItem` tighten by `list_item_gap_factor` (default 0.35 — items read as lines, not paragraphs); the full rhythm reappears at the list ↔ neighbor boundary (the non-list neighbor contributes its untightened half plus the `container_boundary_gap` both sides add when chains differ). Headings keep their own larger factors even inside items.

## Testing — two tiers (mirrors `crates/eidola-gui`)

- **Behavior tests (`tests/behavior.rs`) — the regression gate.** `gpui::TestAppContext`, worker-thread cheap. Construct an `Entity<MarkdownEditorState>` (wrapped in `EditorHarness` for a `Render` root), drive through the `focus_handle` or `apply_event_for_test`, assert on `value()`/`selection()`/`RenderSpec`. State transitions + pure render decisions; never geometry.
- **Visual snapshots (`tests/visual.rs`) — local debug aid.** `VisualTestAppContext`, `harness = false` (main-thread AppKit), rendered Day + Night to gitignored PNGs (platform-bound pixels; not a CI gate). Missing → written; mismatch → `.new.png` + fail; `UPDATE_SNAPSHOTS=1` overwrites; `VISUAL_FILTER=<substring>` narrows a run (the full set is minutes). Build alongside `eidola-gui` (`cargo build -p gpui-markdown-editor -p eidola-gui --tests`) — that crate enables gpui's `runtime_shaders`, without which the Metal shader build needs Apple's `metal` toolchain; run the `target/debug/deps/visual-*` binary directly.
- **Required cursor-position coverage.** Every construct's snapshot suite must exercise the cursor: (1) inside the construct, (2) just outside on either side, (3) on a separate line, (4) with a selection overlapping it. The Kitchen Sink case combines all constructs and varies the cursor — the safety net for cross-feature interaction bugs.

## Build & run

```bash
cargo build -p gpui-markdown-editor
cargo test -p gpui-markdown-editor                                   # behavior tests (the gate)
EIDOLA_RUN_VISUAL_TESTS=1 cargo test -p gpui-markdown-editor --test visual
UPDATE_SNAPSHOTS=1 cargo test -p gpui-markdown-editor --test visual
cargo run -p gpui-markdown-editor --bin demo
```

## Process for adding a markdown feature

1. **Discover.** Build against the demo, *read every snapshot*, identify deviations. Think like a user.
2. **Articulate tests.** Behavior tests (state/render-spec) + visual cases (cursor at the positions above).
3. **Fix.** Iterate.

### Where to make changes

| What | Where |
|------|-------|
| Keyboard behavior (Enter, Backspace, Tab) | `update.rs` |
| Table structure / editing rules | `table.rs` (pure edits) + `update.rs` (event arms) |
| New events | `event.rs` + `update.rs` + `editor.rs` wiring |
| New construct (parsing) | `parser.rs` + `syntax.rs` |
| Cursor-aware delimiter visibility | `render.rs` |
| Visual styling | `style.rs` (derived from theme) |
| Glyph substitution / hidden ranges | `render.rs` (RenderBlock fields) + `element.rs` (shape time) |
| Full-width decorations | `render.rs::Decoration` + `element.rs::paint_decoration` |
| Vertical navigation geometry | `editor.rs::visual_move_caret` + `LaidOutBlock` in `element.rs` |
| Paste / clipboard transforms | `update.rs::{paste, plain_paste, verbatim_paste, markdown_paste}` + `editor.rs` handlers |

## Known design notes

- **The `gpui-component` spec is shared with `crates/eidola-gui`** — both track the `eidola` branch of `eidola-ai/gpui-component` (same spec so cargo unifies; `Cargo.lock` holds the resolved rev). Move them in lockstep; carried-patch inventory and update practice live in `crates/eidola-gui/AGENTS.md` → gpui / gpui-component pinning. `pulldown-cmark` tracks the `eidola` branch of `eidola-ai/pulldown-cmark` (empty-nested-lists opt-in + block display math; back to crates.io when the upstream PRs land).
- **No Eidola-specific symbols.** Deps: `gpui`, `gpui-component`, `gpui-component-assets`, `pulldown-cmark`, `unicode-segmentation`, `smallvec`. Other gpui apps can use it without the rest of the workspace.
- **No async / no I/O.** Everything synchronous and pure except the gpui paint hooks. No tokio, no spawned tasks.
