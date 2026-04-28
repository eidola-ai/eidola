import AppKit
import Foundation
import Testing

@testable import MarkdownEditor

/// Visual tests for checkbox list wrap/continuation behavior and deeply nested mixed lists.
///
/// ## Gap 1: Checkbox wrap/continuation
///
/// Checkbox list items (`- [ ] ` and `- [x] `) should have the same hanging-indent behavior
/// as bullet and ordered list items: wrapped text aligns after the checkbox glyph, and
/// Shift+Return continuation lines align correctly.
///
/// ## Gap 2: Mixed nested lists (bullet + checkbox + ordered)
///
/// A deeply nested list mixing all three marker types should render with correct indentation,
/// wrap alignment, and glyph hiding/revealing at every nesting level.
@Suite("Mixed Nested List Tests")
@MainActor
struct MixedNestedListTests {

  // MARK: - Gap 1: Checkbox wrap alignment

  static let longText =
    "This is a very long checkbox item that wraps around so that we are able to see its indentation behavior with wrapped text on a narrow view"

  /// Unchecked checkbox item with long wrapping text, cursor outside.
  ///
  /// **PASS:** Wrapped lines align with the content start (after the checkbox glyph),
  /// NOT with the left margin. The checkbox glyph (empty square) is visible. The text
  /// forms a clean hanging indent just like bullet items.
  ///
  /// **FAIL:** Wrapped text starts at the left margin or at a different indent than
  /// the first line's content start.
  @Test("Unchecked checkbox wrap alignment -- cursor outside")
  func uncheckedCheckboxWrapOutside() {
    let markdown = "- [ ] \(Self.longText)\n- Short item"
    let results = EditorTestHarness.run(
      name: "checkbox-wrap-align/unchecked-outside",
      initial: EditorState(markdown: markdown, selection: .cursor(markdown.count - 2)),
      events: [],
      size: NSSize(width: 400, height: 200))

    let fm = FileManager.default
    #expect(fm.fileExists(atPath: results[0].imagePath))
  }

  /// Unchecked checkbox item with long wrapping text, cursor inside.
  ///
  /// **PASS:** The `- [ ] ` delimiter is visible (dimmed). Wrapped lines still align
  /// with content start after the delimiter. The hanging indent is consistent.
  ///
  /// **FAIL:** Delimiter is hidden when cursor is inside, or wrapped lines shift
  /// position when delimiter visibility changes.
  @Test("Unchecked checkbox wrap alignment -- cursor inside")
  func uncheckedCheckboxWrapInside() {
    let markdown = "- [ ] \(Self.longText)\n- Short item"
    let results = EditorTestHarness.run(
      name: "checkbox-wrap-align/unchecked-inside",
      initial: EditorState(markdown: markdown, selection: .cursor(10)),
      events: [],
      size: NSSize(width: 400, height: 200))

    let fm = FileManager.default
    #expect(fm.fileExists(atPath: results[0].imagePath))
  }

  /// Checked checkbox item with long wrapping text, cursor outside.
  ///
  /// **PASS:** Wrapped lines align with content start after the checked checkbox glyph
  /// (filled/checked square). Same hanging indent behavior as unchecked.
  ///
  /// **FAIL:** Wrap alignment differs between checked and unchecked items.
  @Test("Checked checkbox wrap alignment -- cursor outside")
  func checkedCheckboxWrapOutside() {
    let markdown = "- [x] \(Self.longText)\n- Short item"
    let results = EditorTestHarness.run(
      name: "checkbox-wrap-align/checked-outside",
      initial: EditorState(markdown: markdown, selection: .cursor(markdown.count - 2)),
      events: [],
      size: NSSize(width: 400, height: 200))

    let fm = FileManager.default
    #expect(fm.fileExists(atPath: results[0].imagePath))
  }

  /// Checked checkbox item with long wrapping text, cursor inside.
  ///
  /// **PASS:** The `- [x] ` delimiter is visible (dimmed). Wrapped lines align consistently.
  ///
  /// **FAIL:** Delimiter hidden, or wrap alignment shifts when cursor enters the item.
  @Test("Checked checkbox wrap alignment -- cursor inside")
  func checkedCheckboxWrapInside() {
    let markdown = "- [x] \(Self.longText)\n- Short item"
    let results = EditorTestHarness.run(
      name: "checkbox-wrap-align/checked-inside",
      initial: EditorState(markdown: markdown, selection: .cursor(10)),
      events: [],
      size: NSSize(width: 400, height: 200))

    let fm = FileManager.default
    #expect(fm.fileExists(atPath: results[0].imagePath))
  }

  // MARK: - Gap 1: Checkbox Shift+Return continuation

  /// Shift+Return on an unchecked checkbox item produces a correctly aligned continuation.
  ///
  /// **PASS:** After Shift+Return, the continuation line text aligns with the content
  /// start of the checkbox item (after the glyph). The continuation line does NOT get
  /// its own checkbox marker. The leading whitespace on the continuation line is hidden.
  ///
  /// **FAIL:** Continuation line starts at the left margin, or gets its own `- [ ] `
  /// marker, or the leading whitespace is visible as extra indentation.
  @Test("Shift+Return continuation on unchecked checkbox item")
  func shiftReturnUncheckedCheckbox() {
    var events: [EditorEvent] = []
    for c in "- [ ] First line" { events.append(.insertText(String(c))) }
    events.append(.insertLineBreak)
    for c in "second line" { events.append(.insertText(String(c))) }

    let results = EditorTestHarness.run(
      name: "checkbox-wrap-align/shiftreturn-unchecked",
      initial: EditorState(),
      events: events,
      size: NSSize(width: 600, height: 300))

    let finalState = results.last!.state
    // Continuation should be indented with 6 spaces (matching "- [ ] " width)
    #expect(finalState.markdown == "- [ ] First line\n      second line")

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }
  }

  /// Shift+Return on a checked checkbox item produces a correctly aligned continuation.
  ///
  /// **PASS:** Same alignment behavior as unchecked. Continuation line aligns with content.
  ///
  /// **FAIL:** Continuation misaligns, or checked state affects continuation behavior.
  @Test("Shift+Return continuation on checked checkbox item")
  func shiftReturnCheckedCheckbox() {
    var events: [EditorEvent] = []
    for c in "- [x] First line" { events.append(.insertText(String(c))) }
    events.append(.insertLineBreak)
    for c in "second line" { events.append(.insertText(String(c))) }

    let results = EditorTestHarness.run(
      name: "checkbox-wrap-align/shiftreturn-checked",
      initial: EditorState(),
      events: events,
      size: NSSize(width: 600, height: 300))

    let finalState = results.last!.state
    #expect(finalState.markdown == "- [x] First line\n      second line")

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }
  }

  // MARK: - Gap 2: Mixed nested lists (bullet + checkbox + ordered)

  static let mixedNestedMarkdown = """
    - Bullet top level
        - [ ] Nested checkbox unchecked with long wrapping text that should align properly after the checkbox glyph
        - [x] Nested checkbox checked
            1. Deeply nested ordered item one with long wrapping text that verifies alignment
            2. Second ordered item
        - Another nested bullet
    - [ ] Top level checkbox
        1. Nested ordered under checkbox
            - [ ] Deeply nested checkbox under ordered
    - Back to bullet
    """

  /// Mixed nested list with cursor at end of document (outside all items).
  /// All glyphs should be in their hidden/substituted state.
  ///
  /// **PASS:**
  /// - Top-level bullets show bullet glyphs (no `- ` visible)
  /// - Top-level checkboxes show checkbox glyphs (no `- [ ] ` or `- [x] ` visible)
  /// - Nested items at each level are progressively indented
  /// - Wrapped text at each level aligns with content start, not with the marker
  /// - Ordered items show numbers without raw `1. ` delimiter
  /// - Each nesting level has distinct, increasing left indentation
  ///
  /// **FAIL:**
  /// - Any raw delimiter text visible (`- `, `- [ ] `, `1. `, etc.)
  /// - Flat indentation (all items at same level)
  /// - Wrapped text not aligned with first-line content
  /// - Missing glyphs (no bullet/checkbox/number shown)
  @Test("Mixed nested list -- cursor outside all items (glyphs hidden)")
  func mixedNestedOutside() {
    let markdown = Self.mixedNestedMarkdown
    let results = EditorTestHarness.run(
      name: "mixed-nested/outside-all",
      initial: EditorState(markdown: markdown, selection: .cursor(markdown.count)),
      events: [],
      size: NSSize(width: 500, height: 500))

    let fm = FileManager.default
    #expect(fm.fileExists(atPath: results[0].imagePath))
  }

  /// Mixed nested list with cursor inside a top-level bullet item.
  /// The `- ` delimiter on that line should be visible/dimmed; all other items hidden.
  ///
  /// **PASS:**
  /// - "Bullet top level" line shows dimmed `- ` delimiter
  /// - All other items still show substituted glyphs (bullets, checkboxes, numbers)
  /// - Indentation is consistent
  ///
  /// **FAIL:**
  /// - Delimiter not visible on the cursor's line
  /// - Other items' delimiters also revealed
  @Test("Mixed nested list -- cursor inside top-level bullet")
  func mixedNestedInsideBullet() {
    let markdown = Self.mixedNestedMarkdown
    // "- Bullet top level" -- cursor at offset 5, inside "Bullet"
    let results = EditorTestHarness.run(
      name: "mixed-nested/inside-top-bullet",
      initial: EditorState(markdown: markdown, selection: .cursor(5)),
      events: [],
      size: NSSize(width: 500, height: 500))

    let fm = FileManager.default
    #expect(fm.fileExists(atPath: results[0].imagePath))
  }

  /// Mixed nested list with cursor inside a nested unchecked checkbox.
  ///
  /// **PASS:**
  /// - The nested `- [ ] Nested checkbox unchecked...` line shows dimmed `- [ ] ` delimiter
  /// - The checkbox glyph is replaced by the raw delimiter text
  /// - Other items remain in their hidden state
  /// - Wrapped text on this item aligns correctly even with delimiter visible
  ///
  /// **FAIL:**
  /// - Delimiter not visible
  /// - Other items' delimiters also revealed
  /// - Wrap alignment breaks when delimiter becomes visible
  @Test("Mixed nested list -- cursor inside nested unchecked checkbox")
  func mixedNestedInsideCheckbox() {
    let markdown = Self.mixedNestedMarkdown
    // "    - [ ] Nested checkbox unchecked..." starts after "- Bullet top level\n"
    // That's 19 chars + 4 spaces indent + 6 for "- [ ] " = offset ~29, place cursor at ~35
    let lines = markdown.components(separatedBy: "\n")
    var offset = lines[0].count + 1  // skip first line + newline
    offset += 10  // into the nested checkbox content
    let results = EditorTestHarness.run(
      name: "mixed-nested/inside-nested-checkbox",
      initial: EditorState(markdown: markdown, selection: .cursor(offset)),
      events: [],
      size: NSSize(width: 500, height: 500))

    let fm = FileManager.default
    #expect(fm.fileExists(atPath: results[0].imagePath))
  }

  /// Mixed nested list with cursor inside a deeply nested ordered item.
  ///
  /// **PASS:**
  /// - The `1. Deeply nested ordered item...` line shows dimmed `1. ` delimiter
  /// - Other items remain hidden
  /// - The ordered item's wrapped text aligns after the number
  /// - Three distinct indentation levels visible (bullet > checkbox > ordered)
  ///
  /// **FAIL:**
  /// - Delimiter not visible on the ordered line
  /// - Indentation levels not clearly distinct
  /// - Wrap alignment incorrect at the deeply nested level
  @Test("Mixed nested list -- cursor inside deeply nested ordered item")
  func mixedNestedInsideOrderedItem() {
    let markdown = Self.mixedNestedMarkdown
    let lines = markdown.components(separatedBy: "\n")
    // Lines: 0=bullet, 1=checkbox unchecked, 2=checkbox checked,
    //        3=ordered 1, 4=ordered 2, 5=another bullet, 6=top checkbox,
    //        7=nested ordered, 8=deeply nested checkbox, 9=back to bullet
    var offset = 0
    for i in 0..<3 {
      offset += lines[i].count + 1
    }
    offset += 15  // into the ordered item content
    let results = EditorTestHarness.run(
      name: "mixed-nested/inside-deep-ordered",
      initial: EditorState(markdown: markdown, selection: .cursor(offset)),
      events: [],
      size: NSSize(width: 500, height: 500))

    let fm = FileManager.default
    #expect(fm.fileExists(atPath: results[0].imagePath))
  }

  /// Mixed nested list with cursor inside a deeply nested checkbox under ordered.
  ///
  /// **PASS:**
  /// - The deeply nested `- [ ] Deeply nested checkbox under ordered` shows dimmed delimiter
  /// - This is at the deepest nesting level (bullet > checkbox > ordered > checkbox)
  /// - The indentation clearly reflects the nesting depth
  ///
  /// **FAIL:**
  /// - Delimiter not visible
  /// - Nesting depth not reflected in indentation
  @Test("Mixed nested list -- cursor inside deeply nested checkbox under ordered")
  func mixedNestedInsideDeeplyNestedCheckbox() {
    let markdown = Self.mixedNestedMarkdown
    let lines = markdown.components(separatedBy: "\n")
    // Line 8: "            - [ ] Deeply nested checkbox under ordered"
    var offset = 0
    for i in 0..<8 {
      offset += lines[i].count + 1
    }
    offset += 20  // into the deeply nested checkbox content
    let results = EditorTestHarness.run(
      name: "mixed-nested/inside-deepest-checkbox",
      initial: EditorState(markdown: markdown, selection: .cursor(offset)),
      events: [],
      size: NSSize(width: 500, height: 500))

    let fm = FileManager.default
    #expect(fm.fileExists(atPath: results[0].imagePath))
  }

  /// Multiple cursor positions in one run to verify delimiter toggling across nesting levels.
  /// Captures a snapshot at each position for side-by-side comparison.
  ///
  /// **PASS:**
  /// - Each step shows exactly one item with visible delimiter (the one under the cursor)
  /// - All other items show substituted glyphs
  /// - Moving the cursor between nesting levels does not break indentation
  /// - Bitmap hashes differ between positions (different delimiters revealed)
  ///
  /// **FAIL:**
  /// - Multiple items show raw delimiters simultaneously
  /// - Some positions produce identical bitmaps when they shouldn't
  /// - Indentation shifts when cursor moves between items
  @Test("Mixed nested list -- cursor sweep across all nesting levels")
  func mixedNestedCursorSweep() {
    let markdown = Self.mixedNestedMarkdown
    let lines = markdown.components(separatedBy: "\n")

    // Compute offsets to interesting positions
    func offsetInLine(_ lineIndex: Int, _ charsIn: Int) -> Int {
      var off = 0
      for i in 0..<lineIndex {
        off += lines[i].count + 1
      }
      return off + charsIn
    }

    let positions: [(String, Int)] = [
      ("top-bullet", offsetInLine(0, 5)),
      ("nested-unchecked-checkbox", offsetInLine(1, 15)),
      ("nested-checked-checkbox", offsetInLine(2, 12)),
      ("deep-ordered-1", offsetInLine(3, 18)),
      ("deep-ordered-2", offsetInLine(4, 15)),
      ("nested-bullet", offsetInLine(5, 10)),
      ("top-checkbox", offsetInLine(6, 12)),
      ("ordered-under-checkbox", offsetInLine(7, 12)),
      ("deepest-checkbox", offsetInLine(8, 20)),
      ("back-to-bullet", offsetInLine(9, 5)),
      ("end-of-document", markdown.count),
    ]

    var events: [EditorEvent] = []
    for (_, offset) in positions.dropFirst() {
      events.append(.setSelection(.cursor(offset)))
    }

    let initial = EditorState(
      markdown: markdown, selection: .cursor(positions[0].1))

    let results = EditorTestHarness.run(
      name: "mixed-nested/cursor-sweep",
      initial: initial,
      events: events,
      size: NSSize(width: 500, height: 500))

    #expect(results.count == positions.count)

    // At least several distinct visual states (different items highlighted)
    var uniqueHashes = Set<Int>()
    for r in results {
      uniqueHashes.insert(r.bitmapHash)
    }
    #expect(
      uniqueHashes.count >= 4,
      "Expected at least 4 visually distinct states across \(positions.count) positions, got \(uniqueHashes.count)")

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }
  }
}
