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

- **Heading (cursor outside):** `# ` is hidden. Text renders in larger/bolder font. Extra spacing above.
- **Heading (cursor inside):** `# ` is visible but dimmed. Text renders in larger/bolder font. The `# ` itself should render in the same large font (it's part of the heading line's paragraph style).
- **Body text:** Normal font, normal spacing. No hidden characters.
- **Bold (cursor outside):** `**` hidden on both sides. Text renders bold.
- **Bold (cursor inside):** `**` visible but dimmed. Text renders bold.
- **Italic (cursor outside):** `*` hidden. Text renders italic.
- **Unordered list (cursor outside):** `- ` hidden, replaced by bullet glyph `•`. Content indented.
- **Unordered list (cursor inside):** `- ` visible, dimmed. Content indented.
- **Inline code (cursor outside):** Backticks hidden. Text in monospace with subtle background.
- **Inline code (cursor inside):** Backticks visible, dimmed. Text in monospace with subtle background.
- **Link (cursor outside):** `[` and `](url)` hidden. Link text shown in blue with underline.
- **Link (cursor inside):** Full `[text](url)` visible, URL portion dimmed.

### Common Visual Bugs to Watch For

- Delimiter styling bleeding into content (e.g., `# ` causing heading font on the next line)
- Delimiters not hiding when they should (cursor is clearly outside the construct)
- Delimiters hiding when they shouldn't (cursor is inside the construct)
- Font/size suddenly changing mid-word because a construct boundary falls there
- Hidden characters leaving blank gaps or causing text to jump when cursor moves in/out
- Heading delimiter `# ` rendered at body-text font size instead of the heading's font size

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
| `EditorEvent` | `EditorEvent.swift` | User actions: insertText, deleteBackward, insertNewline, etc. |
| `EditorUpdate` | `EditorUpdate.swift` | Pure function: `update(state, event) -> state` |
| `MarkdownRenderer` | `MarkdownRenderer.swift` | Pure function: `render(state) -> RenderSpec` |
| `RenderSpec` | `RenderSpec.swift` | Rendering instructions (attributes, hidden glyphs, etc.) |
| `RenderApplicator` | `RenderApplicator.swift` | Imperative shell: applies RenderSpec to NSTextView |

### Supporting Types

| Type | File | Purpose |
|------|------|---------|
| `MarkdownParser` | `MarkdownParser.swift` | Walks swift-markdown AST → `[SyntaxNode]` |
| `SyntaxNode` | `SyntaxNode.swift` | Parsed markdown construct with ranges |
| `SourceRangeConverter` | `SourceRangeConverter.swift` | UTF-8 ↔ UTF-16 offset conversion |
| `MarkdownStyle` | `MarkdownStyle.swift` | Theme: fonts, colors, paragraph styles |
| `GlyphHidingLayoutManagerDelegate` | `GlyphHidingLayoutManagerDelegate.swift` | NSLayoutManager glyph suppression/substitution |
| `MarkdownEditor` | `MarkdownEditor.swift` | SwiftUI NSViewRepresentable shell (thin) |

### Key Principle

**`EditorUpdate.update()` is the only place state transitions happen.** All markdown-aware keyboard behavior (list continuation, heading creation, etc.) belongs here. The `MarkdownEditor.swift` view is a thin adapter that converts NSTextView delegate calls into `EditorEvent` values and feeds them through the update loop.

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

#### StepResult

Each step returns:
- `event` — the event that was processed
- `state` — the resulting `EditorState` (markdown + selection)
- `imagePath` — path to the PNG snapshot
- `bitmapHash` — hash of the bitmap data (for change detection)

### Test Categories

#### 1. State Tests (unit, fast)

Test `EditorUpdate.update()` directly — no rendering needed:

```swift
@Test("Insert character at cursor position")
func insertCharAtCursor() {
    let state = EditorState(markdown: "hllo", selection: .cursor(1))
    let result = EditorUpdate.update(state, event: .insertText("e"))
    #expect(result.markdown == "hello")
    #expect(result.selection == .cursor(2))
}
```

#### 2. Visual Tests (integration, with rendering)

Use the harness to capture images and verify visual properties:

```swift
@Test("Cursor inside heading reveals delimiters")
func headingDelimiterVisibility() {
    let initial = EditorState(markdown: "# Hello\n\nBody", selection: .cursor(3))
    let results = EditorTestHarness.run(
        name: "heading-visibility",
        initial: initial,
        events: [.setSelection(.cursor(10))])  // Move to body

    // Visuals should differ (delimiters visible vs hidden)
    #expect(results[0].bitmapHash != results[1].bitmapHash)
}
```

#### 3. Determinism Tests

Verify that incremental rendering matches fresh rendering:

```swift
let freshBitmap = SnapshotCapture.capture(
    text: finalState.markdown,
    cursorPosition: finalState.selection.head)
let comparison = BitmapComparator.compare(freshBitmap, incrementalBitmap)
#expect(comparison.isMatch)
```

## Process for Adding/Fixing a Markdown Feature

Each feature goes through three phases. The agent should complete all three in one pass.

### Phase 1: Discover

Run the test harness to produce visual artifacts, then **read every image** to identify deviations from the expected Obsidian/Milkdown behavior described in "Target Behavior" above.

Think like a user: Would this feel right? Would there be a jarring jump? Would a delimiter appearing/disappearing cause confusion? Does the styling feel consistent?

### Phase 2: Articulate Tests

Turn discovered issues into test cases. There are two kinds:

**Functional tests** — programmatic assertions on state or rendering properties:

```swift
@Test("Heading delimiter uses heading font when cursor is inside")
func headingDelimiterFont() {
    // Set up state, render, check that the delimiter range has heading-sized font
}
```

**Visual regression tests** — produce artifacts with natural-language descriptions of expected appearance. The test itself just ensures images are generated; a reviewer (human or LLM) checks correctness:

```swift
@Test("Typing '# Hello World' character by character")
func typingHeading() {
    let results = EditorTestHarness.runTyping(
        name: "heading-typing",
        characters: "# Hello World")
    // ... assertions on state ...
}
```

The harness writes a `manifest.md` for each test. Add a `## Expected Behavior` section to the manifest (via the harness or test code) describing what a reviewer should see at key steps. This builds the "well-articulated spec" iteratively.

### Phase 3: Fix

Update the implementation to make tests pass. After fixing:

1. Run `swift test` — all tests must pass
2. **Read the generated images again** to verify the fix looks correct
3. If images still look wrong, iterate (back to Phase 1)

### Quick Reference: Where to Make Changes

| What | Where |
|------|-------|
| Keyboard behavior (Enter, Backspace with markdown awareness) | `EditorUpdate.swift` |
| Markdown parsing (new construct types) | `MarkdownParser.swift` + `SyntaxNode.swift` |
| Visual styling (fonts, colors, spacing) | `MarkdownStyle.swift` |
| Delimiter hiding/revealing logic | `MarkdownRenderer.swift` |
| Glyph suppression mechanics | `GlyphHidingLayoutManagerDelegate.swift` |
| Attribute application to NSTextView | `RenderApplicator.swift` |

### Build & Test

```bash
cd apps/macos/Packages/MarkdownEditor
swift build
swift test
```

## Build & Run

```bash
# Build
cd apps/macos/Packages/MarkdownEditor
swift build

# Run tests
swift test

# Run demo app
swift run MarkdownEditorDemo
```

## File Layout

```
Sources/MarkdownEditor/
├── EditorState.swift          # State model
├── EditorEvent.swift          # Event enum
├── EditorUpdate.swift         # Pure state transitions
├── MarkdownRenderer.swift     # Pure render function
├── RenderSpec.swift           # Rendering specification
├── RenderApplicator.swift     # Applies spec to NSTextView
├── MarkdownParser.swift       # AST → SyntaxNode
├── SyntaxNode.swift           # Parsed markdown construct
├── SourceRangeConverter.swift # UTF-8 ↔ UTF-16
├── MarkdownStyle.swift        # Theme/fonts/colors
├── GlyphHidingLayoutManagerDelegate.swift  # Glyph suppression
└── MarkdownEditor.swift       # SwiftUI shell (thin)

Sources/MarkdownEditorDemo/
└── DemoApp.swift              # Demo app

Tests/MarkdownEditorTests/
├── EditorTestHarness.swift    # Test harness
├── EditorUpdateTests.swift    # State transition tests
├── EditorVisualTests.swift    # Visual integration tests
└── VisualRegression/
    ├── BitmapComparator.swift       # Pixel comparison
    ├── MarkdownTextViewFactory.swift # Test NSTextView creation
    └── SnapshotCapture.swift        # Bitmap capture
```
