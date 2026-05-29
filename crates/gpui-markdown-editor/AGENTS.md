# gpui-markdown-editor — Agent Development Guide

A WYSIWYG markdown editor as a `gpui-component`-style widget. Targets
`crates/eidola-gui` (the chat composer) but is intentionally generic so other gpui
applications can drop it in.

## Foundational Goals

1. **Valid, compliant markdown.** The buffer is always valid CommonMark
   (with one exception: two consecutive newlines are preserved as user-visible
   paragraph separation rather than collapsed). The editor *may* normalize on
   input (e.g. setext → ATX) but never invents non-spec syntax.
2. **A single editable document.** Selections cross spans and blocks; the
   user thinks in markdown.
3. **Block composability.** Lists / blockquotes nest arbitrarily; leaf blocks
   (code, math) are inert islands.

## Target Behavior

The user edits markdown source but sees rich formatting — *except around their
cursor*. Delimiters hide when the cursor is outside the construct and reveal
(dimmed) when the cursor or an active selection enters it. Formatting applies
to content, never to delimiters; the underlying markdown is never modified to
achieve a visual effect; copy/paste always produces raw markdown.

## Pixel-fidelity goal with chat rendering

The chat in `crates/eidola-gui` already uses `gpui-component`'s `TextView::markdown`
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

## Architecture

The editor is structured as a pure transformation pipeline:

```
EditorState + EditorEvent  →  update()  →  new EditorState
                                                 ↓
                                            parse()  →  SyntaxTree
                                                 ↓
                                            render(state, tree)  →  RenderSpec
                                                 ↓
                                            BlockElement (gpui Element, one per block)
```

Architectural decisions:

- **Per-block GPUI `Element`s** rather than a single attributed string. Each
  block is its own painter, which makes full-width code-block backgrounds and
  blockquote borders trivial — they're per-block decorations.
- **`display_to_source` per shaped line.** We can shape display strings
  shorter than their source range (delimiters genuinely removed from the
  shaped line) because gpui's `WrappedLine` returns positions in display-byte
  coordinates. We translate back at hit-test / cursor-paint time via a
  per-line map, mirroring the pattern from `gpui::Editor`.
- **Keyboard input through gpui actions.** IME / dead-key composition uses
  `EntityInputHandler` (the gpui input-handler trait) — see the
  `examples/input.rs` upstream pattern.
- **Shift-arrow extension state on `Selection::Range::anchor`**, which gpui
  updates from a cursor-state member directly.

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
- Lists (top-level, nested inside blockquotes, **and nested inside
  list items**): unordered (`-` / `*` / `+`) and ordered (`1.`,
  `2.`, …). Each item's children are walked in source order; each
  inline run (or `Paragraph` child) becomes one leaf, and each
  block-level child (nested `List`, `BlockQuote`, `CodeBlock`,
  `Heading`) recurses to emit its own leaves. Every leaf carries
  the same `Container::ListItem` chain entry for the item it sits
  in — nested items pick up an additional entry per level, so a
  triple-nested item's leaf carries three. Each entry records this
  item's `marker_byte_len` and the parent list's
  `list_max_marker_text` (the widest marker text by digit count —
  for `1..11` items that's `"11. "`, for unordered lists
  canonicalized to `"- "`). The element layer shapes
  `list_max_marker_text` in the body font and uses
  `list_indent + marker_pixel_width` as that level's left padding,
  so every item in the list aligns at the same content edge
  regardless of its own marker's width. The marker bytes
  themselves are always hidden from the shaped line (analogous to
  the blockquote `>` overlay treatment) and the marker glyph
  paints as a `MarkerOverlay` right-aligned inside the item's
  indent strip. The renderer additionally hides the cumulative
  ancestor-indent on every continuation line of every leaf so
  nested content and wraparound text shape from the same column
  as the first line. **Tab** nests the cursor's item under the
  previous sibling at its depth (no-op if there's no previous
  sibling); **Shift+Tab** dedents the item by one level, falling
  through to "drop the marker" at depth 0. For unordered items
  the overlay glyph is `• ` when the cursor is outside the item
  and the raw bullet char (`- `, `* `, `+ `) when inside, so the
  user has visual feedback while editing the marker scope.
  Ordered items keep their digits visible always (the numbers
  carry meaning); they renumber automatically when items are
  inserted, removed, or reordered: every item in an ordered list
  gets `start + index` regardless of what the user typed.

  **CommonMark interaction note: ordered lists can't open mid-item
  unless they start at 1.** Pulldown follows the spec rule that
  "an ordered list with start > 1 cannot interrupt a paragraph"
  — and the same restriction applies inside another list item's
  content. So when Tab nests an ordered item, the marker is
  rewritten to `1. ` regardless of what number it had at the
  outer level (otherwise the post-Tab source parses as
  continuation text, not a nested list). The renumbering pass
  then handles any subsequent siblings — joining an existing
  nested list with prior items at numbers 1, 2, 3 simply
  rewrites the new arrival from `1. ` back to `n+1. `. The
  general principle: editing operations that change list
  structure must produce a source that pulldown actually parses
  as the intended structure; the renumbering / canonicalization
  passes only operate on what pulldown sees.

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
  - No lazy continuations: continuation lines carry exactly the
    item's *cumulative* indent (sum of every enclosing list-item's
    marker width — 2 for top-level `- `, 4 for an item nested
    once inside another `- ` item, etc.) and the preceding line
    ends with a hard break (`  \n`). Editing `9.` → `10.`
    re-aligns every continuation by +1 space; the inverse for
    narrowing. Nested items inherit ancestor indent so a deep
    triple-nest reads at the right column.
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
- **Word-granular motion and selection-extension.** `MoveWordLeft` /
  `MoveWordRight` (and the `Extend*` variants) walk Unicode word
  boundaries via `unicode-segmentation`, skipping whitespace and
  punctuation. Default macOS bindings: Option+Arrow and
  Option+Shift+Arrow.
- **Line- and word-aware deletion.** `DeleteWordBackward` /
  `DeleteWordForward` consume a word in the cursor's direction;
  `DeleteToLineStart` / `DeleteToLineEnd` consume to the line's
  visible content edge / line end. Inside a structural chain (BQ / LI)
  the target is clamped to the line's chain-prefix end so the `> ` /
  `- ` / continuation-indent bytes survive — deletion only affects
  content, not structure.
- **Visual-position-aware vertical navigation.** Up / Down / Shift+Up
  / Shift+Down route through `MarkdownEditor::vertical_move`, which
  consults the previous frame's `LaidOutBlock` layout to step exactly
  one wrap-row in display coordinates (not source bytes). Two
  consequences: (a) navigation respects soft-wrap rows inside a single
  logical line, and (b) a long line → wrapped short row → long line
  round-trip lands back at the original visual column via an
  `intended_x` anchor that survives the streak (cleared on any
  non-vertical event by `dispatch_reset_intended_x_unless_vertical`).
  Headless tests with no laid-out blocks fall back to the source-byte
  `MoveUp` / `MoveDown` event so behavior tests still move
  predictably.
- Inline code (`` `code` ``): mono font + faint background fill on the
  span content, hide / dim of the backtick delimiters per the cursor
  rule. Multi-backtick spans (`` ``a`b`` ``) work — the parser
  detects the opener / closer run lengths from the source bytes.
- Inline links (`[text](url)`): link-colored, single-underline text;
  hide / dim of the `[`, `](url)` delimiter pair per the cursor
  rule. Nested styling inside the link text composes (a `**bold**`
  inside a link picks up both the link color and the bold weight).
- Thematic breaks (`---` / `***` / `___`): a `BlockKind::ThematicBreak`
  block kind with a thin horizontal rule painted as a per-block
  decoration centered on the row. The source bytes hide when the
  cursor is outside, dim when inside.
- GFM task list items (`- [ ] todo` / `- [x] done`): the parser sets
  a `task: Option<bool>` field on `NodeKind::ListItem` whenever
  pulldown emits `Event::TaskListMarker(checked)` inside an item.
  The bullet's `marker_range` stays at `- ` (2 bytes) so the
  continuation-indent math is unchanged; the renderer adds a
  separate hide for the `[ ] ` / `[x] ` task bytes that follow,
  and the marker overlay paints `☐ ` / `☑ ` in place of the
  bullet glyph.
- CommonMark §2.4 backslash escapes (`\*`, `\\`, …) and §2.5
  entity references (`&amp;`, `&copy;`, `&#x2014;`). Each
  occurrence becomes a `Substitution` (cursor outside) that
  replaces the source bytes with the resolved literal, or a
  dimmed `InlineRun` (cursor inside) that reveals the raw bytes.
  The scan happens in a render post-pass driven by `escapes.rs`
  and skips verbatim contexts (inline code, fenced code, link
  destinations, math). Pulldown's `Event::Text` content is lossy
  for these constructs (the backslash byte is uncovered, entities
  decode to their resolved value), so the editor scans source
  bytes itself rather than trusting the event stream — see
  `escapes.rs` module docs.
- LaTeX math: inline `$x^2$` and display `$$ ... $$`. Pulldown's
  `Event::InlineMath` / `Event::DisplayMath` produce
  `NodeKind::InlineMath` / `NodeKind::DisplayMath`. Inline math
  has two paths driven by the cursor:
  - **Cursor outside** the construct: the render layer hides
    every byte and emits a `MathOverlay` record. The element
    layer's `augment_block_with_math` pre-pass typesets each
    overlay via `crate::math::typeset`, measures the result,
    and substitutes a width-matched run of NBSPs (U+00A0) in
    place of the source bytes. The shaped line reserves
    horizontal space for the math; surrounding text shapes
    around it. After painting the line text, the paint phase
    locates each substitution's display offset and paints the
    typeset math at that x with baseline aligned to the line's
    text baseline (≈0.78 of row height from row top).
  - **Cursor inside** the construct: fall back to the
    dim-delimiter / mono-content path so the user can read and
    edit the raw LaTeX directly.
  Display math becomes a `BlockKind::DisplayMath { content_range,
  edit_mode }` via two parser paths that both feed the same
  `render::emit_display_math_block` helper:
  - **Block-level `$$\n...\n$$`** — the `pulldown-cmark` fork's
    `parse_display_math_block` (mirrors `parse_fenced_code_block`)
    emits these as a top-level `Event::DisplayMath` with no
    wrapping paragraph. Content can span blank lines; the
    construct is terminated by a `$$`-only line or EOF
    (unterminated, same shape as an unterminated fenced code
    block). `analysis::display_math_blocks` projects this to a
    `DisplayMathBlock { range, terminated }` for cursor queries.
  - **Sole-paragraph promotion** — a paragraph whose only
    content-bearing child is a single inline `$$x$$` (e.g.
    `$$x^2$$` on its own line) is still promoted via
    `render_paragraph::sole_display_math_child` into the same
    block kind, so single-line and multi-line constructs render
    identically.

  The block has two render modes that swap based on cursor
  position:
  - **Display mode** (`edit_mode == false`, cursor outside): the
    element layer typesets via `math::typeset(latex, MathMode::Display)`,
    allocates the block's height to the math layout's pixel
    size, and paints the typeset math via `MathLayout::paint`.
  - **Edit mode** (`edit_mode == true`, cursor inside): falls
    back to text shaping — `$$` delimiters dim, inner LaTeX
    shapes in the mono font, multi-line content shapes
    naturally. The block reserves at least the natural
    display-mode height (`max(natural edit, natural display)`)
    so toggling edit mode doesn't shift surrounding content.

  Editing semantics inside a block-level `$$..$$` follow the same
  rules as fenced code (collected under [`is_in_verbatim_region`]):
  Enter inserts a literal `\n` plus the chain continuation prefix
  (no soft-break-to-pair promotion), Tab inserts a literal `\t`,
  Backspace at structural boundaries falls through to grapheme
  delete, blank lines pass through as content. The first Enter
  inside an unterminated `$$` triggers `analysis::auto_close_math_edit`
  — analog of `auto_close_fence_edit`, injecting a matching `$$`
  closer below the cursor. After auto-close, every other rule reads
  off `is_in_display_math_block` (or `is_in_verbatim_region`) without
  the unterminated-state ambiguity.

  The inclusive-overlap test (boundary cursors *do* count as
  inside for display-vs-edit) gives the "click to edit" feel:
  the cursor can park at the math's leading or trailing edge
  and immediately flip to edit. KaTeX fonts auto-register on
  first display-math paint via `math::register_katex_fonts`;
  hosts may also call it at app init alongside their own font
  loads.
- Images (`![alt](url)`): structurally identical to math — pulldown
  emits `Tag::Image` as a container with the alt text as inline
  children. `NodeKind::Image { delimiter_ranges, alt_range,
  dest_url }` records the `![` / `](url)` delimiter pair and the
  alt span between them. Two cursor-driven paths mirror inline
  math:
  - **Cursor outside**: render emits an `ImageOverlay`. The
    element layer's `augment_block_with_images` pre-pass calls
    `crate::image::load`, measures the natural size, caps height
    to `INLINE_HEIGHT_FACTOR * line_height`, and substitutes a
    width-matched NBSP run so surrounding text reserves space.
    `paint_inline_image_overlays` paints each loaded image
    vertically centered on the row that hosts its substitution.
  - **Cursor inside**: dim-delimiter / visible-alt-text fallback
    so the user can edit the raw markdown.
  A paragraph whose sole content is an Image promotes to
  `BlockKind::Image { alt_range, dest_url, edit_mode }` — same
  promotion rule and inclusive-overlap edit-mode test as
  `DisplayMath`. In display mode the element layer scales the
  image to fit the available content width (`crate::image::block_size`)
  and paints directly via `window.paint_image`. Edit mode reserves
  at least the natural display-mode height (the same
  `max(natural edit, natural display)` rule as `BlockKind::DisplayMath`)
  so toggling edit mode doesn't shift surrounding content. Image
  loading is asynchronous: the cache returns `Loading` until the
  asset resolves, `Failed` on error. `Loading` reserves a
  placeholder (square at the inline height cap; ~8em-tall banner
  for block images) and invalidates the view when the load
  completes; `Failed` falls back to the dim-delimiter + alt-text
  inline run pair so the user sees the construct and the broken
  URL. Loaders for `http://`, `https://`, `file://`, and absolute
  paths come for free via gpui's image cache; relative-path /
  embedded resolution depends on hosts registering an `AssetSource`.

Explicitly *out* of this phase: setext-heading normalization, tables,
HTML, IME marked-text, reference-style images (`![alt][label]`), the
`title` attribute on images, image data URIs, image links
(`[![alt](img)](url)`), rich-content paste from HTML / RTF (depends
on a gpui-side clipboard-mime extension that doesn't exist yet — see
[Clipboard pipeline](#clipboard-pipeline) for the deferred path).

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

- **Soft-break promotion is chain-aware.** A stray `\n` is exempt
  from promotion only if it's part of a complete pair (the
  forbidden-position detector recognizes the chain-aware pair shape
  `\n{blank_prefix}\n{content_prefix}` from `chain_pair_shape` and
  forbids interior bytes accordingly). Any other lone mid-content
  `\n` — soft breaks across BQ lines, lazy continuations — is
  promoted: `enforce_invariants` inserts the missing prefix bytes
  per `chain_continuation_prefix` so the result is a complete pair
  for the cursor's chain. The chat renderer's CommonMark
  soft-break-as-soft-break rendering does diverge from the editor's
  promote-everything rule on paste — that's the only pixel-fidelity
  cost of the simpler invariant.
- **Atomic pair delete.** Backspace at the *end* of a chain-aware
  pair removes the whole pair in one keystroke; Delete-forward at
  the *start* does the symmetric delete. The pair shape is whatever
  `chain_pair_shape(chain)` produces for the cursor's chain — symmetric
  `\n{prefix}\n{prefix}` for chains ending in BQ, asymmetric
  `\n{blank}\n{content}` for chains with BQ trailed by LIs, or
  `\n\n{indent}` for chains with no BQ. Inside fenced code-block
  content `\n`s are literal — the pair detector is bypassed there,
  falling through to grapheme delete.
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
- **Unterminated-fence-aware code classification.** `is_in_fenced_code`
  treats the EOF position of an unterminated fence as inside the
  construct (an unterminated fence's range ends at `bytes.len()`, so
  there's no closer to sit "after"). Every cursor-driven query —
  Enter routing, atomic pair-delete bypass, soft-break exemption,
  forbidden-position predicate — funnels through this predicate.
  See `analysis::FencedCodeBlock` and
  `fenced_code_content_ranges_with_state`.
- **Auto-close-fence on Enter.** `analysis::auto_close_fence_edit`
  fires before regular Enter routing when the cursor sits inside an
  unterminated fence. The edit injects a matching closer below the
  cursor (matching fence char and length, with the chain continuation
  prefix on each new line) and lands the cursor on a body row in
  between. Once this fires the construct is terminated; subsequent
  rules read off `is_in_fenced_code` without the unterminated-state
  ambiguity that produced the original cascade
  (`bugs.md::enter_at_end_of_unterminated_fenced_code_inserts_paragraph_break_pair`).
- **In-fence Enter emits chain prefix.** Inside a (now-terminated)
  fence, `enter_insertion` returns `\n` + `chain_continuation_prefix(chain)`,
  *not* a bare `\n`. The new code-body row keeps the BQ / LI
  continuation bytes so the construct stays inside its enclosing
  scope.
- **Empty-BQ Enter outdents** (mirror of empty-LI Enter outdent).
  `analysis::empty_bq_paragraph_exit_edit` returns the chain-aware
  pair-replacement edit when the user presses Enter at the end of a
  trailing BQ-pair shape; the innermost BQ scope drops, the trailing
  shape becomes the reduced-chain pair. Wired before `enter_insertion`
  in `update::insert_newline`.

### Chain-aware invariants (the helper family)

The depth-D-pair invariants above are special cases of a more general
rule: every byte sequence that "introduces" a continuation line in a
nested chain is built by walking the chain outermost-first, emitting
the per-container prefix (LI indent, BQ marker) in chain order. So a
chain `[LI(2), BQ, LI(2), BQ]` produces `"  >   > "` — outer-LI indent,
outer BQ marker, inner-LI indent, inner BQ marker.

`analysis.rs` exposes a small canonical helper family. **Use these —
don't compute prefixes locally.** Reaching for raw `\n` boundaries or
hand-built `"> "` strings in a chain-aware context is a bug; we've
fixed several of those by migrating to these helpers.

| Helper | Use when… |
|--------|----------|
| `chain_continuation_prefix(chain)` | You need the bytes that introduce a continuation line for the cursor's chain (Enter inserts, Shift+Enter inserts, soft-break promotion, render's chain-aware hide pass). |
| `chain_continuation_prefix_bytes(chain)` | Same byte-length without allocating. |
| `chain_outer_prefix_bytes(chain)` | The byte count contributed by every container *above* the innermost — the offset to insert / strip indent at without disturbing outer markers (Tab indent insertion, Shift+Tab dedent strip). |
| `chain_pair_shape(chain) -> (blank, content)` | You're emitting or recognizing a structural pair. The pair shape is always `\n{blank}\n{content}`; three branches collapse into one tuple: chain ends in BQ → symmetric, BQ trailed by LIs → asymmetric, no BQ → `("", full)`. |

These helpers power: `enter_insertion`, `line_break_insertion`, soft-break
promotion in `update.rs`, `list_item_indent_edits`, `list_item_dedent_edits`,
`build_depth_decrease_edit`, atomic pair-delete (`pair_at_end_for_chain`),
forbidden-position detection (`is_chain_pair_interior`), and on the render
side `chain_for_position`, `hide_chain_continuation_prefix`, and
`merge_hard_break_continuations`. New chain-aware code should funnel
through them; if a future call site needs a *new* shape variant, add it
here with the same naming pattern.

The cursor walker `analysis::enclosing_containers_at` is similarly the
single source of truth for "what containers enclose byte X". The render
walker's `chain_for_position` delegates to it so the two analyses can
never disagree.

### Render walker pipeline

`render::render` is structured as a pipeline, not a tree walk. After the
recursive `render_node` walk produces a flat `Vec<RenderBlock>`, several
post-passes run in a specific order to refine the spec. The order is
load-bearing; reorder only with care.

```text
1. recursive walk (render_node → render_paragraph / render_blockquote /
   render_list / render_list_item / render_code_block / render_heading)
2. inject_empty_paragraphs   — synth empty Paragraph leaves for trailing
                                positions and inter-block paragraph
                                breaks pulldown didn't claim. Each
                                synth's chain comes from
                                `chain_for_position`.
3. merge_hard_break_continuations  — when pulldown splits a `  \n`
                                      hard break followed by a trailing
                                      line of pure chain-continuation
                                      prefix into two blocks, merge
                                      them so the visual matches the
                                      with-content case.
4. hide_chain_continuation_prefix (per block) — final chain-driven
                                                 hide pass that catches
                                                 alternating-chain
                                                 prefix bytes the
                                                 per-container hides
                                                 miss.
5. merge_hidden_ranges (per block) — normalize the per-block
                                      `hidden_ranges` into a sorted,
                                      non-overlapping list.
```

New passes that fix follow-on bugs slot into this list with a clear
rationale. The doc comment on `render::render` carries this same list
in code; keep both in sync.

## Clipboard pipeline

Three event variants cross the clipboard boundary, each with its own
update-pipeline branch:

| Event | Triggered by | Behavior |
|-------|-------------|----------|
| `EditorEvent::InsertText(String)` | IME composition, programmatic insertion | Splice raw at cursor. Used by the input-handler trait — IME's commit step routes here. No paste-specific transforms. |
| `EditorEvent::Paste { text, internal }` | Cmd+V | Markdown-aware splice. `internal: true` when the clipboard's `metadata()` matches the [`CLIPBOARD_SENTINEL`] (set on every `copy` / `cut`). |
| `EditorEvent::PastePlain { text }` | Cmd+Shift+V | Plain-text splice — bypasses markdown parse and soft-break collapse. |

[`CLIPBOARD_SENTINEL`]: src/editor.rs

All three flow through the same `update.rs:paste` router, which drops
any active selection first (so chain / verbatim analysis runs on the
post-deletion buffer) and then dispatches:

```text
Paste { text, internal } ─┬─ in_verbatim_region? → verbatim_paste
                          └─ otherwise           → markdown_paste

PastePlain { text } ──────┬─ in_verbatim_region? → verbatim_paste
                          └─ otherwise           → insert_text
```

### Verbatim paste (`verbatim_paste`, Phase 1)

Cursor sits inside a fenced code block or block-level `$$..$$` math.
Two transforms apply:

- **Chain-prefix injection.** Every embedded `\n` is followed by
  [`chain_continuation_prefix`] so continuation lines stay inside any
  enclosing BQ / LI scope.
- **Fence widening.** If the pasted bytes contain a run of the
  enclosing fence's char (`` ` `` or `~`) that matches the
  closing-fence pattern after chain-prefix stripping, the opener and
  closer are both widened to `max_run + 1` in lockstep so the pasted
  content can't accidentally close the construct. Reads opener and
  closer ranges via [`analysis::fence_with_delimiters_at`]. Display
  math `$$` has no longer form — the splice lands verbatim and the
  user fixes any break.

The widening edits and the splice are composed into one
`SourceEditList` so `apply_edits` handles cursor remapping in a single
pass.

### Markdown-canonicalize paste (`markdown_paste`, Phase 2)

Cursor sits outside any verbatim region. Three transforms apply in
order:

1. **Canonicalize.** Parse `text`; walk the tree for every
   `NodeKind::SoftBreak` leaf (pulldown emits `Event::SoftBreak`
   *only* for genuine in-paragraph soft breaks — never between sibling
   list items, never between blocks, never inside verbatim regions,
   never for hard breaks). For each one, replace the `\n` byte plus
   the chain continuation prefix that follows it with a single space.
   This turns `foo\n> bar` into `foo bar` inside a BQ paragraph (not
   `foo > bar`, which would leave a literal `>` mid-content).
   Skipped when `internal: true` — internal pastes are already
   canonical markdown.
2. **Block-boundary padding.** Re-parse the canonical paste; if its
   first / last top-level node is non-Paragraph (heading, list, fence,
   BQ, thematic break, display math), prepend / append `\n\n` (or
   `\n` if the cursor is one short of a blank-line boundary) so the
   construct lands on its own line. Cursor's surrounding bytes
   determine whether padding is needed at all.
3. **Chain-prefix injection.** Same helper as `verbatim_paste` —
   every `\n` in the post-padding bytes is followed by
   `chain_continuation_prefix(chain_at_cursor)`.

The Phase-2 transform deliberately diverges from the editor's "every
`\n` is structural" invariant *on paste*: a hard-wrapped paragraph
from any source (markdown, terminal output, browser plaintext)
collapses to one paragraph, matching CommonMark's rendering rule for
soft breaks. The trade-off: plaintext where line breaks are meaningful
(code, poetry, address blocks) also collapses — the user-triggered
[Plain paste](#plain-paste-plain_paste-phase-3a) path is the escape
hatch.

**Why walk `NodeKind::SoftBreak` instead of scanning for `\n`s inside
Paragraph ranges?** Byte-level scanning can't distinguish "`\n`
between two content lines of one paragraph" from "`\n` between two
sibling list items" — the trailing `\n` is included in the first
ListItem's range, and adjacency to other `\n`s doesn't help.
Pulldown's SoftBreak event is the exact parser-blessed enumeration we
want, and the parser already lifts it into a `NodeKind::SoftBreak`
leaf.

### Plain paste (`plain_paste`, Phase 3a)

User-triggered Cmd+Shift+V. Bytes splice raw — no markdown parse, no
soft-break collapse, no block-boundary padding. Each `\n` becomes a
paragraph break post-splice (the chain-aware `promote_soft_breaks`
pass in `enforce_invariants` lifts each lone `\n` into the depth-D
pair shape for the cursor's chain). Sentinel metadata is ignored —
the user explicitly chose plain semantics, overriding any "this came
from our editor" signal.

Markdown markers (`#`, `*`, `` ` ``, etc.) are *not* pre-escaped. The
contract is "splice raw"; downstream interpretation depends on what
the pasted bytes happen to parse as. A separate "paste as literal
text" mode that escapes every CommonMark marker is a follow-up.

### CRLF normalization

`editor::normalize_line_endings` collapses CRLF (Windows) and bare CR
(legacy macOS) to LF before the bytes ever reach `update::update`, so
the chain-prefix and parser passes only have to reason about `\n`.
Applies to both `Paste` and `PastePlain`.

### Sentinel metadata

`copy` / `cut` tag every clipboard write with
`ClipboardItem::new_string_with_metadata(text, CLIPBOARD_SENTINEL)`.
`paste` reads the metadata back to set `internal: true`. The sentinel
literal is intentionally crate-namespaced (`gpui-markdown-editor`),
not app-namespaced — this crate carries no Eidola-specific symbols.

### Deferred: rich-content paste (Phase 3b/3c)

Real HTML / RTF passthrough depends on a gpui-side
`ClipboardEntry::Html(String)` (or a generic
`ClipboardEntry::MimeData { mime, bytes }`) variant that doesn't
exist yet — gpui's clipboard layer flattens to a single plaintext
flavor. Once the gpui change lands, the natural next step is an
HTML → Markdown converter (lean walker over a tag-soup parser,
covering the subset browsers actually emit: `<p>`, `<h1..h6>`,
`<a>`, `<strong>`, `<em>`, `<code>`, `<pre>`, `<ul>/<ol>/<li>`,
`<blockquote>`, `<img>`). Its output feeds into the Phase 2
pipeline. Heuristic markdown detection on plaintext was considered
and rejected — too many surprising false positives, and the
user-triggered Plain paste action is a better UX answer.

[`chain_continuation_prefix`]: src/analysis.rs
[`analysis::fence_with_delimiters_at`]: src/analysis.rs

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
| `escapes.rs` | CommonMark §2.4 / §2.5 source-byte scanner. Returns one `ResolvedSpan` per `\X` or `&entity;` occurrence; the render post-pass turns each into a `Substitution` (cursor outside) or a dimmed `InlineRun` (cursor inside). |
| `math.rs` | RaTeX adapter. `register_katex_fonts(text_system)` loads the bundled KaTeX TTFs; `typeset(latex, mode) -> MathLayout` parses + lays out a LaTeX expression; `MathLayout::paint(...)` walks RaTeX's `DisplayList` and emits native gpui paint ops (`paint_quad` for fraction bars and rects, `paint_path` for radicals, shaped glyph runs for letters / operators). |
| `image.rs` | Image-cache adapter. `load(dest_url, window, cx) -> LoadedImage` routes a URL through gpui's image cache (`http`/`https`/`file`/absolute paths supported out of the box, embedded paths via the host's `AssetSource`). `inline_size` / `block_size` apply the height cap / width fit so inline and block images each scale predictably. `paint(image, bounds, window)` is the thin `window.paint_image` wrapper. Asynchronous: `LoadedImage::Loading` while in flight, `Failed` on error — the asset cache invalidates the view when a load resolves so the next frame sees the new state. |
| `bin/demo.rs` | Standalone demo window |

## Theme integration

The editor does **not** carry its own color palette. `MarkdownStyle::from_theme`
derives every color (text, secondary, delimiter, background) from
`gpui_component::Theme`. Day / Night just work because they're the theme's
job. Callers can override individual fields after construction (font size,
heading callback, paragraph gap) the same way `eidola_gui::chat::markdown_style`
overrides `TextViewStyle`.

## Testing — two tiers (mirrors `crates/eidola-gui`)

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

Mirrors `crates/eidola-gui/tests/visual.rs` exactly:

- Missing PNG → write it and report `written`.
- Mismatch → save `<name>-<mode>.new.png` for review and fail.
- `UPDATE_SNAPSHOTS=1` overwrites.

The PNGs are **gitignored** because they're platform-bound. They're a debug
aid for the developer making a UI change, not a CI gate.

### Required cursor-position coverage

Every visual test must exercise the cursor at many positions, not just
where typing left it. Each construct's snapshot suite must include cursor:

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

## Process for adding a markdown feature

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
| Vertical navigation behavior (Up / Down geometry) | `editor.rs::visual_move_caret` + `LaidOutBlock` layout in `element.rs` |
| Paste / clipboard transforms | `update.rs::{paste, plain_paste, verbatim_paste, markdown_paste}` + `editor.rs` clipboard handlers — see [Clipboard pipeline](#clipboard-pipeline) |

## Known design notes

- **`gpui-component` pin is shared with `crates/eidola-gui`.** Same git rev. If we
  bump `crates/eidola-gui`, we bump here in lockstep so cargo unifies.
- **The crate has no Eidola-specific symbols.** It only depends on `gpui`,
  `gpui-component`, `gpui-component-assets`, `pulldown-cmark`,
  `unicode-segmentation`, and `smallvec`. Other gpui apps can use it
  without pulling in the rest of the workspace.
- **No async / no I/O.** Everything is synchronous and pure except for the
  gpui paint hooks themselves. No tokio, no spawned tasks.
