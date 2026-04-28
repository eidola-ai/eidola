import AppKit
import Testing

@testable import MarkdownEditor

/// Tests that wrapped list item text aligns with the content start, not the bullet/number.
///
/// The visual requirement: when a list item's text wraps to the next line, the wrapped
/// text should begin at the same horizontal position as where the content starts after
/// the bullet or number on the first line. This creates a clean hanging indent.
@Suite("List Wrap Alignment Tests")
@MainActor
struct ListWrapAlignmentTests {

  static let longText =
    "This is a very long item that wraps around so that we are able to see its indentation behavior with wrapped text"

  @Test("Unordered list wrap alignment — cursor outside")
  func unorderedWrapOutside() {
    let markdown = "- \(Self.longText)\n- Short item"
    // Cursor on "Short item" — outside the long item, so bullet shows
    let results = EditorTestHarness.run(
      name: "wrap-align-unordered-outside",
      initial: EditorState(markdown: markdown, selection: .cursor(markdown.count - 2)),
      events: [],
      size: NSSize(width: 400, height: 200))

    let fm = FileManager.default
    #expect(fm.fileExists(atPath: results[0].imagePath))
    // Agent review: wrapped text should start at same position as content after bullet
  }

  @Test("Unordered list wrap alignment — cursor inside")
  func unorderedWrapInside() {
    let markdown = "- \(Self.longText)\n- Short item"
    // Cursor inside the long item — "- " should be dimmed and visible
    let results = EditorTestHarness.run(
      name: "wrap-align-unordered-inside",
      initial: EditorState(markdown: markdown, selection: .cursor(5)),
      events: [],
      size: NSSize(width: 400, height: 200))

    let fm = FileManager.default
    #expect(fm.fileExists(atPath: results[0].imagePath))
  }

  @Test("Ordered list wrap alignment")
  func orderedWrap() {
    let markdown = "1. \(Self.longText)\n2. Short item"
    let results = EditorTestHarness.run(
      name: "wrap-align-ordered",
      initial: EditorState(markdown: markdown, selection: .cursor(markdown.count - 2)),
      events: [],
      size: NSSize(width: 400, height: 200))

    let fm = FileManager.default
    #expect(fm.fileExists(atPath: results[0].imagePath))
  }

  @Test("Nested unordered list wrap alignment")
  func nestedUnorderedWrap() {
    let markdown = """
      - Short top item
      - \(Self.longText)
        - \(Self.longText)
          - \(Self.longText)
      - Short bottom item
      """
    // Cursor far from the items
    let results = EditorTestHarness.run(
      name: "wrap-align-nested-unordered",
      initial: EditorState(markdown: markdown, selection: .cursor(markdown.count)),
      events: [],
      size: NSSize(width: 500, height: 400))

    let fm = FileManager.default
    #expect(fm.fileExists(atPath: results[0].imagePath))
  }

  @Test("Nested ordered list wrap alignment")
  func nestedOrderedWrap() {
    let markdown = """
      1. Short top item
      2. \(Self.longText)
        1. \(Self.longText)
          1. \(Self.longText)
      3. Short bottom item
      """
    let results = EditorTestHarness.run(
      name: "wrap-align-nested-ordered",
      initial: EditorState(markdown: markdown, selection: .cursor(markdown.count)),
      events: [],
      size: NSSize(width: 500, height: 400))

    let fm = FileManager.default
    #expect(fm.fileExists(atPath: results[0].imagePath))
  }

  // MARK: - Ordered list marker padding (single → double digit)

  static let repeatingL = "LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL"

  /// When an ordered list crosses from single-digit (1-9) to double-digit (10+),
  /// shorter markers like "1. " must be padded so their content starts at the
  /// same horizontal position as "10. " content. The "L" characters make
  /// alignment easy to verify visually — all "L" columns should be vertically
  /// aligned across every item.
  ///
  /// **Pass criteria:**
  /// - The "L" in items 1-9 starts at the SAME horizontal position as the "L" in item 10
  /// - Wrapped lines in ALL items align with the first-line content start
  /// - There is NO visible jagged left edge in the content column
  ///
  /// **Fail criteria:**
  /// - Items 1-9 have their "L" further left than item 10 (no padding applied)
  /// - Wrapped lines start at a different position than first-line content
  @Test("Ordered list single-to-double digit marker padding alignment")
  func orderedListMarkerPadding() {
    var lines: [String] = []
    for i in 1...10 {
      lines.append("\(i). \(Self.repeatingL)")
    }
    let markdown = lines.joined(separator: "\n")

    let results = EditorTestHarness.run(
      name: "ordered-marker-padding",
      initial: EditorState(markdown: markdown, selection: .cursor(markdown.count)),
      events: [],
      size: NSSize(width: 600, height: 800))

    let fm = FileManager.default
    #expect(fm.fileExists(atPath: results[0].imagePath))
  }

  // MARK: - Nested wrap regression (headings + 3 levels of both list types)

  static let nestedWrapMarkdown = """
    ### Unordered Lists
    - This is a very long item that wraps around so that we are able to see its indentation. This is a very long item that wraps around so that we are able to see its indentation.\u{0020}
        - This is a very long item that wraps around so that we are able to see its indentation. This is a very long item that wraps around so that we are able to see its indentation.\u{0020}
            - This is a very long item that wraps around so that we are able to see its indentation. This is a very long item that wraps around so that we are able to see its indentation.\u{0020}

    ### Ordered Lists
    1. This is a very long item that wraps around so that we are able to see its indentation. This is a very long item that wraps around so that we are able to see its indentation.\u{0020}
        1. This is a very long item that wraps around so that we are able to see its indentation. This is a very long item that wraps around so that we are able to see its indentation.\u{0020}
            1. This is a very long item that wraps around so that we are able to see its indentation. This is a very long item that wraps around so that we are able to see its indentation.\u{0020}
    """

  @Test("Nested wrap regression — cursor outside all constructs")
  func nestedWrapRegressionOutside() {
    let markdown = Self.nestedWrapMarkdown
    // Place cursor at the very end — outside all constructs so delimiters are hidden
    let results = EditorTestHarness.run(
      name: "nested-wrap-regression/outside",
      initial: EditorState(markdown: markdown, selection: .cursor(markdown.count)),
      events: [],
      size: NSSize(width: 600, height: 800))

    let fm = FileManager.default
    #expect(fm.fileExists(atPath: results[0].imagePath))
  }

  @Test("Nested wrap regression — cursor inside nested item")
  func nestedWrapRegressionInside() {
    let markdown = Self.nestedWrapMarkdown
    // Find the second-level unordered item and place cursor inside it
    // "    - This is a very long..." — place cursor ~10 chars into this line
    let lines = markdown.components(separatedBy: "\n")
    // Line 0: "### Unordered Lists"
    // Line 1: "- This is..."  (level 1)
    // Line 2: "    - This is..."  (level 2)
    // We want cursor inside line 2
    var offset = 0
    for i in 0..<2 {
      offset += lines[i].count + 1  // +1 for newline
    }
    offset += 10  // ~10 chars into line 2 (inside the content)

    let results = EditorTestHarness.run(
      name: "nested-wrap-regression/inside-nested",
      initial: EditorState(markdown: markdown, selection: .cursor(offset)),
      events: [],
      size: NSSize(width: 600, height: 800))

    let fm = FileManager.default
    #expect(fm.fileExists(atPath: results[0].imagePath))
  }
}
