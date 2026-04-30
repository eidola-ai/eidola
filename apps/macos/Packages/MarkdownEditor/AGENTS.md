# MarkdownEditor — Agent Development Guide

## Foundational Goals

These are the load-bearing commitments of the editor. Every behavior, rule, and design choice below should be evaluated against these. When in doubt, return here.

### 1. Valid, compliant markdown

- Everything produced by the editor MUST be valid markdown, and MUST be rendered correctly according to the CommonMark spec unless explicitly noted in an exception.
- The editor MAY choose to normalize the markdown representation of features. Normalizations SHOULD be applied to the content regardless of input type (manual editing, paste, etc.) unless otherwise noted in an exception.
- **Exceptions:**
  - The CommonMark spec collapses any number of consecutive newlines into a single paragraph break. The editor MUST treat every two newlines as a new "empty" paragraph to allow user-controlled visual separation while editing.
  - The editor MAY opt into rendering behavior that is not compliant or normalized **only when** doing so is necessary for a smooth editing experience **and** the non-compliant behavior is conditional on cursor position during active editing. For example, a `-` at the start of a new line might display literally while followed by the cursor, rather than immediately being normalized from a setext heading into an ATX heading.

### 2. A single editable document

- The editing flow MUST treat the editable value as a single document where selections can cross spans and blocks.
- A user should be able to "think" in markdown and interact with it accordingly:
  - Invisible characters MUST be handled thoughtfully based on cursor position.
  - Selected content SHOULD expose its raw markdown when appropriate.
  - Jitters and changes in line height MUST be minimized.

### 3. Block composability

- Non-leaf block components (lists of all types, blockquotes) MUST be nestable. It MUST be possible to nest arbitrary depths and combinations of these.
- Leaf block components (code blocks, math) MUST avoid all markdown normalization and custom editor behavior within their contents.

## Target Behavior

This editor aims to match the inline WYSIWYG behavior of **Obsidian** and **Milkdown**. The core UX principle:

**The user edits markdown source, but sees rich formatting — except around their cursor.**

### Visual Rules

1. **Delimiters hide when the cursor is outside the construct.** For example, `# ` before a heading disappears — the user sees only the large-font heading text. `**` around bold text disappears — the user sees bold text without asterisks.

2. **Delimiters reveal when the cursor — or an active selection — enters the construct.** When the cursor moves into (or a selection range overlaps) a heading line, the `# ` prefix reappears (dimmed) so the user can see, edit, delete, or copy the raw markdown source. Same for `**`, `` ` ``, `[](url)`, etc.

3. **Formatting applies to content, not delimiters (except paragraph style).** The heading text is large/bold. The `# ` prefix, when visible, is shown in a dimmed color but at the heading's font size — it shares the heading line's paragraph style so there is no jarring size mismatch within the line.

4. **The underlying text is always valid markdown.** The editor never modifies the markdown to achieve visual effects — it only changes how it's _displayed_. Copy/paste always produces the raw markdown.

5. **Paragraph spacing and indentation match the construct.** Headings have extra spacing above. List items are indented. Block quotes are indented with secondary label color. These are visual-only and don't change the markdown.

### What "Correct" Looks Like (per construct)

When reviewing test snapshots, check against these expectations:

**Block constructs:**
- **Heading (cursor outside):** `# ` is hidden. Text renders in larger/bolder font. Extra spacing above.
- **Heading (cursor inside):** `# ` is visible but dimmed, in heading font size.
- **Body text:** Normal font, normal spacing. No hidden characters.
- **Code block (cursor outside):** Opening/closing ` ``` ` fences hidden. Content in monospace font. Full-width background color forms a uniform box.
- **Code block (cursor inside):** Fences visible and dimmed. Content still monospace with background.
- **Blockquote (cursor outside):** `> ` prefix on each line hidden. Content in secondary label color, indented.
- **Blockquote (cursor inside):** `> ` prefixes visible and dimmed on ALL lines of the blockquote.
- **Horizontal rule (cursor outside):** `---`/`***`/`___` text is transparent with thick strikethrough in separator color.
- **Horizontal rule (cursor inside):** Raw `---`/`***`/`___` visible and dimmed.
- **Setext headings:** Normalized to ATX (`# `) format when cursor leaves the underline. Single `-` underline suppressed while cursor is on it (ambiguous with list start).

**List constructs:**
- **Unordered list (cursor outside):** `- ` hidden, replaced by bullet glyph `•` with space. Content indented. Wrapped text aligns with content start after bullet.
- **Unordered list (cursor inside):** `- ` visible, dimmed. Content indented.
- **Ordered list (cursor outside):** Number marker (`1. `, `2. `) always visible. Content indented. All items in same list use widest marker width for consistent alignment. Shorter markers padded via kern.
- **Ordered list (cursor inside):** Same as outside (markers always visible).
- **Checkbox list (cursor outside):** `- [ ] ` / `- [x] ` hidden, replaced by ☐/☒ glyph with space. Content indented.
- **Checkbox list (cursor inside):** Full `- [ ] ` / `- [x] ` visible and dimmed.
- **Nested lists:** Progressive indentation. Leading whitespace always hidden (paragraph style controls indent). Continuation lines have whitespace hidden too.
- **Multi-line list items:** Shift+Return creates continuation lines. Wrapped and continuation text aligns with content start after marker.

**Inline constructs:**
- **Bold (cursor outside):** `**` hidden on both sides. Text renders bold.
- **Bold (cursor inside):** `**` visible but dimmed. Text renders bold.
- **Italic (cursor outside):** `*` hidden. Text renders italic.
- **Italic (cursor inside):** `*` visible but dimmed. Text renders italic.
- **Bold italic (cursor outside):** `***` hidden. Text renders bold and italic.
- **Bold italic (cursor inside):** `***` visible but dimmed.
- **Strikethrough (cursor outside):** `~~` hidden. Text renders with strikethrough line.
- **Strikethrough (cursor inside):** `~~` visible but dimmed. Strikethrough still applied.
- **Inline code (cursor outside):** Backticks hidden. Text in monospace with subtle background.
- **Inline code (cursor inside):** Backticks visible, dimmed. Text in monospace with background.
- **Link (cursor outside):** `[` and `](url)` hidden. Link text in blue with underline.
- **Link (cursor inside):** Full `[text](url)` visible, URL portion dimmed.
- **Image (cursor outside):** `![` and `](url)` hidden. Alt text in secondary color, italic.
- **Image (cursor inside):** Full `![alt](url)` visible, delimiters dimmed.

### Common Visual Bugs to Watch For

- Delimiter styling bleeding into content (e.g., `# ` causing heading font on the next line)
- Delimiters not hiding when they should (cursor is clearly outside the construct)
- Delimiters hiding when they shouldn't (cursor is inside the construct)
- Font/size suddenly changing mid-word because a construct boundary falls there
- Hidden characters leaving blank gaps or causing text to jump when cursor moves in/out
- Heading delimiter `# ` rendered at body-text font size instead of the heading's font size
- Paragraph spacing shrinking when delimiters at paragraph start are hidden
- List items losing indentation when leading whitespace is visible alongside paragraph style indent
- Wrapped/continuation text not aligning with content start after marker

## Architecture

The editor follows the Elm architecture. All state transitions and rendering are pure functions.

```
EditorState + EditorEvent  →  EditorUpdate.update()  →  new EditorState
                                                              ↓
                                                    MarkdownRenderer.render()  →  RenderSpec
                                                              ↓
                                                    TextKit2RenderApplicator.apply()  →  NSTextView
```

### Core Types

| Type | File | Purpose |
|------|------|---------|
| `EditorState` | `EditorState.swift` | Model: markdown string + selection |
| `EditorEvent` | `EditorEvent.swift` | User actions: insertText, deleteBackward, indent, etc. |
| `EditorUpdate` | `EditorUpdate.swift` | Pure function: `update(state, event) -> state` + post-processing (ordinal renumbering, setext normalization) |
| `MarkdownRenderer` | `MarkdownRenderer.swift` | Pure function: `render(state) -> RenderSpec` |
| `RenderSpec` | `RenderSpec.swift` | Rendering instructions (attributes, hidden glyphs, bullet/checkbox indexes, code block ranges) |
| `TextKit2RenderApplicator` | `TextKit2RenderApplicator.swift` | Imperative shell: applies RenderSpec to NSTextView via the TextKit 2 stack |

### Supporting Types

| Type | File | Purpose |
|------|------|---------|
| `MarkdownParser` | `MarkdownParser.swift` | Walks swift-markdown AST → `[SyntaxNode]` |
| `SyntaxNode` | `SyntaxNode.swift` | Parsed markdown construct with ranges and type |
| `SourceRangeConverter` | `SourceRangeConverter.swift` | UTF-8 ↔ UTF-16 offset conversion |
| `MarkdownStyle` | `MarkdownStyle.swift` | Theme: fonts, colors, paragraph styles for all constructs |
| `TextKit2ContentStorageDelegate` | `TextKit2ContentStorageDelegate.swift` | NSTextContentStorageDelegate: vends length-matching display paragraphs (ZWSP for hidden chars, • for bullet markers, ☐/☒ for checkboxes) so display length equals source length |
| `TextKit2LayoutManagerDelegate` | `TextKit2LayoutManagerDelegate.swift` | NSTextLayoutManagerDelegate: vends `TextKit2LayoutFragment` per paragraph with code-block / blockquote decoration data |
| `TextKit2LayoutFragment` | `TextKit2LayoutFragment.swift` | NSTextLayoutFragment subclass: draws full-width code-block backgrounds and blockquote left borders |
| `TextKit2MarkdownTextView` | `TextKit2MarkdownTextView.swift` | NSTextView subclass: arrow-key overrides that skip over hidden source positions during navigation |
| `MarkdownEditor` | `MarkdownEditor.swift` | SwiftUI NSViewRepresentable shell (thin) |

### Key Principles

- **`EditorUpdate.update()` is the only place state transitions happen.** All markdown-aware keyboard behavior belongs here.
- **Post-processing runs after every text mutation:** ordered list renumbering and setext heading normalization.
- **Leading whitespace in nested list items is always hidden** — paragraph style controls indentation, not source spaces.
- **Continuation line whitespace is always hidden** — same reason.
- **The `.controlCharacter` glyph property** (not `.null`) is used for the first hidden character at paragraph boundaries to preserve paragraph spacing calculations.

### Architecture Lessons (don't relearn the hard way)

- **TK2's `NSTextLocation` model requires `display.length == source.length`; preserve this invariant via ZWSP substitution rather than character removal.** TK2's hit-test, character navigation, rendering attributes, and the cursor's visual position all silently assume the displayed paragraph has the same UTF-16 length as the source range it covers. When the content delegate vended *shorter* display paragraphs (hidden chars removed), TK2 computed cursor positions in display coordinates and reported them as source coordinates — clicks landed on the wrong glyph, arrow keys jumped past hidden runs to arbitrary offsets, and the cursor's visual position drifted relative to its source position when delimiter visibility flipped (the `right-arrow from 7 jumps to 12 in "A **B** C D E F G"` repro). The fix is to substitute, not remove: hidden chars become `U+200B ZERO WIDTH SPACE` (length 1, zero visual width, valid cursor landing), bullet markers become `•`, and multi-char glyph substitutions (checkboxes) emit the visible glyph followed by ZWSP padding so total length still matches. With the invariant restored, the per-paragraph display→source translation maps, the hit-test override (`translateHitTestIndex`), the word-level navigation translation, and the Coordinator's drift-correction guard all became dead code and were removed. The single exception to the invariant is paragraphs whose entire source range is in `hiddenIndexes` — those are skipped via `NSTextContentManagerDelegate.shouldEnumerate` (the element isn't vended at all). Character-level move overrides still skip-over-hidden as a UX choice (cursor never lands on a hidden delimiter), but that's now the *only* translation layer required.
- **Trust TK2's element model; manipulate spacing, not display merging.** TK2's cursor navigation, hit-testing, and selection enumeration all operate on `NSTextElement` ranges, which are 1:1 with source-`\n`-bounded paragraphs. Any approach that tries to coalesce multiple source paragraphs into one displayed paragraph (e.g., substituting `\n` with `U+2028 LINE SEPARATOR` and returning a single merged `NSTextParagraph`) breaks navigation: absorbed elements become "non-represented", the cursor can't enter their content, and right-arrow jumps over their characters entirely. The fix for soft / hard line breaks is per-paragraph spacing (each source paragraph keeps its own element with `paragraphSpacing = 0` between soft-break-coupled segments). The U+2028 trick looks great on paper — and the discovery / articulation phases recommended it — but burned ~70 minutes of agent iteration before the fundamental clash with TK2's element model surfaced. Don't revisit unless TK2 grows an API for source-vs-display element coalescing.
- **Soft / hard breaks are an `NSAttributedString.paragraphStyle` problem, not an `NSTextContentStorageDelegate` substitution problem.** The renderer's `applyParagraphStyle` (and the list-item style in `renderListItem`) split styled ranges at each soft-break `\n` and apply `paragraphSpacing = 0` to non-final segments and `paragraphSpacingBefore = 0` to non-first segments. The content delegate stays simple (just hide / bullet / checkbox substitution) and ignores `lineBreakIndexes` entirely — it's a renderer-side concern.
- **`NSTextContentStorage.delegate` is `NSTextContentStorageDelegate?`, but its inherited `NSTextContentManager` has its own delegate slot of type `NSTextContentManagerDelegate?`.** They are sibling protocols, not inheritance. To use both delegate hooks (e.g., `textParagraphWith:` AND `shouldEnumerate:`), the same object must conform to both protocols and you must set both delegate properties — or rely on the storage subclass dispatching both hooks to its single `.delegate` (which it does in practice for our usage; verified empirically).
- **(Historical, kept for the lesson.) Before the length-matching invariant, TK2's hit-test and character-level selection navigation reported display offsets dressed as source offsets.** When the content delegate vended an `NSTextParagraph` whose display string was shorter than its source range (hidden delimiters / glyph substitutions reduced length), TK2's `characterIndexForInsertion(at:)` and `NSTextSelectionNavigation.destinationSelection(...)` walked the *display* string and reported results as source offsets by adding the display offset to `paragraph.elementRange.location`. A click on display char 4 of "A B C D E F G" came back as source 4 even though visible 'C' was at source 8. Right-arrow could jump arbitrarily across hidden runs for the same reason. The original mitigation was a hit-test override that mapped through a per-paragraph `displayToSourceMap` plus a Coordinator-level drift guard that re-set the cursor when `apply` re-snapped it. The structural fix (preserve `display.length == source.length` via ZWSP substitution) made all of that unnecessary — TK2 now reports real source offsets and the move overrides only need to skip over hidden source positions as a UX concern. The per-paragraph map, the hit-test override, the word-level translation, and the drift guard have all been removed.

## Testing Harness

### `EditorTestHarness` (in Tests/)

The harness accepts an initial `EditorState` and a sequence of `EditorEvent`s. After each event, it:

1. Runs `EditorUpdate.update()` to get the new state
2. Renders the state via `MarkdownRenderer` + `TextKit2RenderApplicator`
3. Captures a bitmap snapshot (PNG)
4. Saves the image to `test-artifacts/<testName>/`
5. Writes a `manifest.md` with state at each step

#### Usage

```swift
// Character-by-character typing
let results = EditorTestHarness.runTyping(
    name: "my-test",
    characters: "# Hello World\n")

// Custom event sequences
let results = EditorTestHarness.run(
    name: "my-test",
    initial: EditorState(markdown: "existing text", selection: .cursor(5)),
    events: [
        .insertText("new "),
        .setSelection(.range(anchor: 0, head: 9)),
        .deleteBackward,
    ])
```

### Critical: Test Cursor at Many Positions

Most bugs only appear when the cursor is NOT at the position where the user just finished typing. Typing tests alone are insufficient.

**Every visual test must include cursor placement at varied positions:**

1. **Inside the construct** — cursor in the middle of content (delimiters should be visible/dimmed)
2. **At the start boundary** — cursor at the first character of the construct
3. **At the end boundary** — cursor at the last character of the construct
4. **Just outside** — cursor one position before or after the construct (delimiters should be hidden)
5. **On a completely unrelated line** — cursor far from the construct
6. **At the end of a line followed by `\n`** — known tricky boundary
7. **At the end of the document** (no trailing `\n`) — another known boundary

For inline constructs, also test cursor just before/after delimiters and between adjacent constructs.

**Kitchen Sink Test:** A combined test with ALL supported constructs in various combinations, moving the cursor to many "interesting" positions. This catches interaction bugs between features.

### Test Categories

#### 1. State Tests (unit, fast)
Test `EditorUpdate.update()` directly — no rendering needed.

#### 2. Visual Tests (integration, with rendering)
Use the harness to capture images and verify visual properties.

#### 3. Determinism Tests
Verify that incremental rendering matches fresh rendering.

## Process for Adding/Fixing a Markdown Feature

Each feature goes through three phases. The agent should complete all three in one pass.

### Phase 1: Discover
Run the test harness, **read every image**, identify deviations from expected behavior. Think like a user.

### Phase 2: Articulate Tests
Turn discovered issues into functional tests and visual regression tests with clear pass/fail criteria.

### Phase 3: Fix
Update implementation, run tests, re-read images, iterate until correct.

### Quick Reference: Where to Make Changes

| What | Where |
|------|-------|
| Keyboard behavior (Enter, Backspace, Tab, Shift+Return) | `EditorUpdate.swift` |
| New events (indent, outdent, line break, delete variants) | `EditorEvent.swift` + `EditorUpdate.swift` + `MarkdownEditor.swift` |
| Markdown parsing (new construct types) | `MarkdownParser.swift` + `SyntaxNode.swift` |
| Visual styling (fonts, colors, spacing) | `MarkdownStyle.swift` |
| Delimiter hiding/revealing logic | `MarkdownRenderer.swift` |
| Glyph suppression/substitution | `TextKit2ContentStorageDelegate.swift` |
| Full-width background drawing | `TextKit2LayoutFragment.swift` (vended by `TextKit2LayoutManagerDelegate.swift`) |
| Attribute application to NSTextView | `TextKit2RenderApplicator.swift` |
| Post-processing (renumbering, normalization) | `EditorUpdate.swift` |

### Build & Test

```bash
cd apps/macos/Packages/MarkdownEditor
swift build
swift test
swift run MarkdownEditorDemo
swift run MarkdownEditorScript Scripts/markdown-bug-bold-arrow-walk.json
```

### Scripted runner (`MarkdownEditorScript`)

`MarkdownEditorScript` hosts the SwiftUI `MarkdownEditor` view in a real
`NSWindow` against a real `NSApplication` and replays a JSON event script
against the production code path (Coordinator wired as `NSTextView`
delegate, full `apply` pipeline runs synchronously per keypress). Useful
for blind agent iteration: every script produces a `trace.jsonl` (JSONL
of state transitions — `selection.before/after`, `apply.start/end`,
`move.start/end`, `hitTest.start/end`, `selection.drift_corrected`),
PNG snapshots per `snapshot` event, and a `manifest.md` summary.

Script schema (see `Scripts/markdown-bug-bold-arrow-walk.json` for a
template):

```json
{
  "markdown": "A **B** C D E F G",
  "size": [800, 200],
  "outDir": "/tmp/out",
  "injection": "sendEvent",
  "stepDelayMs": 60,
  "events": [
    { "type": "set_cursor", "position": 5 },
    { "type": "key", "name": "right" },
    { "type": "snapshot", "name": "after-right" }
  ]
}
```

Event types: `snapshot`, `set_cursor`, `select_range`, `key`, `type`,
`click`, `wait`. Key names: `right`, `left`, `up`, `down`, `shift+right`,
`shift+left`, `home`, `end`, `enter`, `backspace`, `delete`.

Injection mechanisms (for diagnosing event-routing-path sensitivity):
- `direct` (default fallback) — calls action selectors (`moveRight`)
  directly. Same path the existing test rig uses.
- `sendEvent` — synthesizes `NSEvent.keyEvent(...)` and dispatches via
  `NSApp.sendEvent`. Goes through the real responder chain.
- `cgEvent` — posts low-level `CGEvent` keypresses. Requires accessibility
  permissions; without them the events are silently dropped (visible in
  the trace as a missing `move.start`).

`DebugTrace` (in `Sources/MarkdownEditor/DebugTrace.swift`) is the
production-side instrumentation gated by `MARKDOWN_EDITOR_TRACE=1` or
the runner. Per-call cost is one branch when off; never enable in shipped
code paths.

### Cursor-jump-after-bold-arrow bug — fixed (2026-04)

User-reported bug: source `"A **B** C D E F G"`, cursor at source 5,
press right-arrow seven times. Expected walk 5→6→7→8→9→10→11; pre-fix
live demo walk reportedly 5→6→7→12→…, with delimiter dimming and italic
font traits also appearing shifted by the same offset.

Root cause: the content delegate vended display paragraphs *shorter* than
their source ranges (hidden chars were dropped). TK2 then computed cursor
positions, hit-test results, and rendering-attribute ranges in display
coordinates while the rest of the pipeline assumed source coordinates —
so visible glyphs, attribute applications, and cursor-visual-position
all drifted relative to source positions whenever the display length
diverged.

Fix: the content delegate now substitutes hidden chars with `U+200B`
(zero-width space), bullet markers with `•`, and checkboxes with the
visible glyph plus ZWSP padding so display length always equals source
length. With the invariant restored, the runner walks 5→6→7→8→9→10→11→12
cleanly, no drift correction is needed, and the per-paragraph
display→source maps were removed from the hit-test and word-navigation
paths. See `TextKit2ContentStorageDelegate` for the substitution rules.

## Supported Constructs

| Construct | Parser | Renderer | Keyboard | Tests |
|-----------|--------|----------|----------|-------|
| Headings (ATX) | `visitHeading` | delimiter hide/reveal + heading font | — | Yes |
| Setext headings | `visitHeading` (detected by delimiterLength==0) | suppressed for single `-` near cursor | normalized to ATX on cursor move | Yes |
| Bold (`**`) | `visitStrong` | delimiter hide/reveal + bold trait | — | Yes |
| Italic (`*`) | `visitEmphasis` | delimiter hide/reveal + italic trait | — | Yes |
| Bold italic (`***`) | nested Strong+Emphasis | combined traits | — | Yes |
| Strikethrough (`~~`) | `visitStrikethrough` | delimiter hide/reveal + strikethrough | — | Yes |
| Inline code (`` ` ``) | `visitInlineCode` | delimiter hide/reveal + monospace + background | — | Yes |
| Links (`[text](url)`) | `visitLink` | delimiter hide/reveal + blue/underline + `.link` URL | — | Yes |
| Images (`![alt](url)`) | `visitImage` | delimiter hide/reveal + secondary color + italic | — | Yes |
| Unordered lists (`- `) | `visitListItem` | bullet substitution + indent + wrap align | Enter/Backspace/Tab/Shift+Tab/Shift+Return | Yes |
| Ordered lists (`1. `) | `visitListItem` | marker always visible + indent + kern padding | Enter/Backspace/Tab/Shift+Tab + renumbering | Yes |
| Checkbox lists (`- [ ]`) | `visitListItem` | checkbox substitution + indent | Enter/Backspace | Yes |
| Code blocks (` ``` `) | `visitCodeBlock` | fence hide/reveal + monospace + full-width background | — | Yes |
| Blockquotes (`> `) | `visitBlockQuote` | `> ` hide/reveal + secondary color + indent | Enter/Backspace/Shift+Return | Yes |
| Horizontal rules (`---`) | `visitThematicBreak` | transparent text + strikethrough | — | Yes |

## File Layout

```
Sources/MarkdownEditor/
├── EditorState.swift                      # State model (markdown + selection)
├── EditorEvent.swift                      # Event enum (all user actions)
├── EditorUpdate.swift                     # Pure state transitions + post-processing
├── MarkdownRenderer.swift                 # Pure render function (state → RenderSpec)
├── RenderSpec.swift                       # Rendering specification
├── TextKit2RenderApplicator.swift         # Applies spec to NSTextView (TextKit 2 stack)
├── MarkdownParser.swift                   # AST walker → [SyntaxNode]
├── SyntaxNode.swift                       # Parsed construct with ranges
├── SourceRangeConverter.swift             # UTF-8 ↔ UTF-16
├── MarkdownStyle.swift                    # Theme (fonts, colors, paragraph styles)
├── TextKit2ContentStorageDelegate.swift   # Display-paragraph substitution (ZWSP / bullet / checkbox), length-matching invariant
├── TextKit2LayoutManagerDelegate.swift    # Vends TextKit2LayoutFragment per paragraph
├── TextKit2LayoutFragment.swift           # Code-block backgrounds + blockquote borders
├── TextKit2MarkdownTextView.swift         # Arrow-key overrides that skip hidden source positions during navigation
└── MarkdownEditor.swift                   # SwiftUI NSViewRepresentable shell

Sources/MarkdownEditorDemo/
└── DemoApp.swift                          # Demo app with split view

Tests/MarkdownEditorTests/
├── EditorTestHarness.swift                # Test harness (state + events → snapshots)
├── EditorUpdateTests.swift                # State transition tests
├── EditorVisualTests.swift                # Visual integration tests
├── KitchenSinkVisualTests.swift           # Combined all-features test
├── *RenderTests.swift                     # Per-feature render spec tests
├── *VisualTests.swift                     # Per-feature visual tests
├── *UpdateTests.swift                     # Per-feature keyboard behavior tests
└── VisualRegression/
    ├── BitmapComparator.swift             # Pixel comparison
    ├── MarkdownTextViewFactory.swift      # Test NSTextView creation
    └── SnapshotCapture.swift              # Bitmap capture
```
