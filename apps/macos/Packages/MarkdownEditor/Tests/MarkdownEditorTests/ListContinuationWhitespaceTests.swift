import AppKit
import Testing

@testable import MarkdownEditor

/// Tests that continuation line leading whitespace in list items is hidden.
///
/// Multi-line list items have continuation lines indented with spaces to tell the
/// parser they belong to the list item. These spaces should be hidden because the
/// paragraph style's `headIndent` already handles the visual indentation.
@Suite("List Continuation Whitespace Tests")
@MainActor
struct ListContinuationWhitespaceTests {

  // MARK: - Unordered list continuation whitespace

  @Test("Unordered list continuation line whitespace is hidden when cursor outside")
  func unorderedContinuationHiddenOutside() {
    // "- First\n  cont\n\nBody"
    //  0123456 7 89...
    let text = "- First\n  cont\n\nBody"
    let cursorRange = NSRange(location: 18, length: 0)  // in "Body"
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Positions 8 and 9 are the leading "  " on the continuation line
    #expect(
      spec.hiddenIndexes.contains(8),
      "First space of continuation indent should be hidden")
    #expect(
      spec.hiddenIndexes.contains(9),
      "Second space of continuation indent should be hidden")
    // "cont" should not be hidden
    #expect(
      !spec.hiddenIndexes.contains(10),
      "Continuation content should not be hidden")
  }

  @Test("Unordered list continuation line whitespace is hidden when cursor inside")
  func unorderedContinuationHiddenInside() {
    // "- First\n  cont"
    //  0123456 7 89...
    let text = "- First\n  cont"
    let cursorRange = NSRange(location: 5, length: 0)  // inside "First"
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Continuation whitespace should ALWAYS be hidden, regardless of cursor position
    #expect(
      spec.hiddenIndexes.contains(8),
      "Continuation indent should be hidden even with cursor inside item")
    #expect(
      spec.hiddenIndexes.contains(9),
      "Continuation indent should be hidden even with cursor inside item")
  }

  // MARK: - Ordered list continuation whitespace

  @Test("Ordered list continuation line whitespace is hidden when cursor outside")
  func orderedContinuationHiddenOutside() {
    // "1. First\n   cont\n\nBody"
    //  012345678 9...
    let text = "1. First\n   cont\n\nBody"
    let cursorRange = NSRange(location: 20, length: 0)  // in "Body"
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Positions 9, 10, 11 are the leading "   " on the continuation line
    #expect(
      spec.hiddenIndexes.contains(9),
      "First space of ordered continuation should be hidden")
    #expect(
      spec.hiddenIndexes.contains(10),
      "Second space of ordered continuation should be hidden")
    #expect(
      spec.hiddenIndexes.contains(11),
      "Third space of ordered continuation should be hidden")
    // "cont" should not be hidden
    #expect(
      !spec.hiddenIndexes.contains(12),
      "Continuation content should not be hidden")
  }

  @Test("Ordered list continuation line whitespace is hidden when cursor inside")
  func orderedContinuationHiddenInside() {
    // "1. First\n   cont"
    let text = "1. First\n   cont"
    let cursorRange = NSRange(location: 5, length: 0)  // inside "First"
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(
      spec.hiddenIndexes.contains(9),
      "Ordered continuation indent should be hidden even with cursor inside")
    #expect(
      spec.hiddenIndexes.contains(10),
      "Ordered continuation indent should be hidden even with cursor inside")
    #expect(
      spec.hiddenIndexes.contains(11),
      "Ordered continuation indent should be hidden even with cursor inside")
  }

  // MARK: - Multiple continuation lines

  @Test("Multiple continuation lines all have whitespace hidden")
  func multipleContinuationLines() {
    // "- First\n  line2\n  line3\n\nBody"
    let text = "- First\n  line2\n  line3\n\nBody"
    let cursorRange = NSRange(location: 27, length: 0)  // in "Body"
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // First continuation: positions 8, 9
    #expect(spec.hiddenIndexes.contains(8), "First cont line space 1 should be hidden")
    #expect(spec.hiddenIndexes.contains(9), "First cont line space 2 should be hidden")
    #expect(!spec.hiddenIndexes.contains(10), "First cont line content should not be hidden")

    // Second continuation: positions 16, 17
    #expect(spec.hiddenIndexes.contains(16), "Second cont line space 1 should be hidden")
    #expect(spec.hiddenIndexes.contains(17), "Second cont line space 2 should be hidden")
    #expect(!spec.hiddenIndexes.contains(18), "Second cont line content should not be hidden")
  }

  // MARK: - Visual tests

  @Test("Continuation line aligns with first line content visually")
  func continuationAlignmentVisual() {
    let markdown = "- First line content\n  continuation content\n\nBody text"
    // Test with cursor outside
    let results = EditorTestHarness.run(
      name: "continuation-whitespace/unordered-outside",
      initial: EditorState(markdown: markdown, selection: .cursor(markdown.count)),
      events: [],
      size: NSSize(width: 400, height: 200))

    let fm = FileManager.default
    #expect(fm.fileExists(atPath: results[0].imagePath))
  }

  @Test("Continuation line aligns with first line content visually — cursor inside")
  func continuationAlignmentVisualInside() {
    let markdown = "- First line content\n  continuation content\n\nBody text"
    // Cursor inside the list item
    let results = EditorTestHarness.run(
      name: "continuation-whitespace/unordered-inside",
      initial: EditorState(markdown: markdown, selection: .cursor(5)),
      events: [],
      size: NSSize(width: 400, height: 200))

    let fm = FileManager.default
    #expect(fm.fileExists(atPath: results[0].imagePath))
  }

  @Test("Ordered list continuation aligns correctly")
  func orderedContinuationVisual() {
    let markdown = "1. First line content\n   continuation content\n\nBody text"
    let results = EditorTestHarness.run(
      name: "continuation-whitespace/ordered-outside",
      initial: EditorState(markdown: markdown, selection: .cursor(markdown.count)),
      events: [],
      size: NSSize(width: 400, height: 200))

    let fm = FileManager.default
    #expect(fm.fileExists(atPath: results[0].imagePath))
  }

  @Test("Ordered list continuation aligns correctly — cursor inside")
  func orderedContinuationVisualInside() {
    let markdown = "1. First line content\n   continuation content\n\nBody text"
    let results = EditorTestHarness.run(
      name: "continuation-whitespace/ordered-inside",
      initial: EditorState(markdown: markdown, selection: .cursor(5)),
      events: [],
      size: NSSize(width: 400, height: 200))

    let fm = FileManager.default
    #expect(fm.fileExists(atPath: results[0].imagePath))
  }
}
