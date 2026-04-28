# MarkdownEditor — Agent Development Guide

## Target Behavior

This editor aims to match the inline WYSIWYG behavior of **Obsidian** and **Milkdown**. The core UX principle:

**The user edits markdown source, but sees rich formatting — except around their cursor.**

### Visual Rules

1. **Delimiters hide when the cursor is outside the construct.** For example, `# ` before a heading disappears — the user sees only the large-font heading text. `**` around bold text disappears — the user sees bold text without asterisks.

2. **Delimiters reveal when the cursor enters the construct.** When the cursor moves into a heading line, the `# ` prefix reappears (dimmed) so the user can edit or delete it. Same for `**`, `` ` ``, `[](url)`, etc.

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
                                                    RenderApplicator.apply()  →  NSTextView
```

### Core Types

| Type | File | Purpose |
|------|------|---------|
| `EditorState` | `EditorState.swift` | Model: markdown string + selection |
| `EditorEvent` | `EditorEvent.swift` | User actions: insertText, deleteBackward, indent, etc. |
| `EditorUpdate` | `EditorUpdate.swift` | Pure function: `update(state, event) -> state` + post-processing (ordinal renumbering, setext normalization) |
| `MarkdownRenderer` | `MarkdownRenderer.swift` | Pure function: `render(state) -> RenderSpec` |
| `RenderSpec` | `RenderSpec.swift` | Rendering instructions (attributes, hidden glyphs, bullet/checkbox indexes, code block ranges) |
| `RenderApplicator` | `RenderApplicator.swift` | Imperative shell: applies RenderSpec to NSTextView |

### Supporting Types

| Type | File | Purpose |
|------|------|---------|
| `MarkdownParser` | `MarkdownParser.swift` | Walks swift-markdown AST → `[SyntaxNode]` |
| `SyntaxNode` | `SyntaxNode.swift` | Parsed markdown construct with ranges and type |
| `SourceRangeConverter` | `SourceRangeConverter.swift` | UTF-8 ↔ UTF-16 offset conversion |
| `MarkdownStyle` | `MarkdownStyle.swift` | Theme: fonts, colors, paragraph styles for all constructs |
| `GlyphHidingLayoutManagerDelegate` | `GlyphHidingLayoutManagerDelegate.swift` | NSLayoutManager delegate: glyph suppression (`.null`/`.controlCharacter`), bullet/checkbox substitution |
| `CodeBlockBackgroundLayoutManager` | `CodeBlockBackgroundLayoutManager.swift` | NSLayoutManager subclass: draws full-width code block backgrounds |
| `MarkdownEditor` | `MarkdownEditor.swift` | SwiftUI NSViewRepresentable shell (thin) |

### Key Principles

- **`EditorUpdate.update()` is the only place state transitions happen.** All markdown-aware keyboard behavior belongs here.
- **Post-processing runs after every text mutation:** ordered list renumbering and setext heading normalization.
- **Leading whitespace in nested list items is always hidden** — paragraph style controls indentation, not source spaces.
- **Continuation line whitespace is always hidden** — same reason.
- **The `.controlCharacter` glyph property** (not `.null`) is used for the first hidden character at paragraph boundaries to preserve paragraph spacing calculations.

## Testing Harness

### `EditorTestHarness` (in Tests/)

The harness accepts an initial `EditorState` and a sequence of `EditorEvent`s. After each event, it:

1. Runs `EditorUpdate.update()` to get the new state
2. Renders the state via `MarkdownRenderer` + `RenderApplicator`
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
| Glyph suppression/substitution | `GlyphHidingLayoutManagerDelegate.swift` |
| Full-width background drawing | `CodeBlockBackgroundLayoutManager.swift` |
| Attribute application to NSTextView | `RenderApplicator.swift` |
| Post-processing (renumbering, normalization) | `EditorUpdate.swift` |

### Build & Test

```bash
cd apps/macos/Packages/MarkdownEditor
swift build
swift test
swift run MarkdownEditorDemo
```

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
├── RenderApplicator.swift                 # Applies spec to NSTextView
├── MarkdownParser.swift                   # AST walker → [SyntaxNode]
├── SyntaxNode.swift                       # Parsed construct with ranges
├── SourceRangeConverter.swift             # UTF-8 ↔ UTF-16
├── MarkdownStyle.swift                    # Theme (fonts, colors, paragraph styles)
├── GlyphHidingLayoutManagerDelegate.swift # Glyph suppression + bullet/checkbox substitution
├── CodeBlockBackgroundLayoutManager.swift # Full-width code block backgrounds
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
