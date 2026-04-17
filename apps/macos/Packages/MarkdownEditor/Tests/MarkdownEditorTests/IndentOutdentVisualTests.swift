import AppKit
import Foundation
import Testing

@testable import MarkdownEditor

/// Visual integration tests for indent/outdent (Tab/Shift+Tab) behavior.
@Suite("Indent/Outdent Visual Tests")
@MainActor
struct IndentOutdentVisualTests {

  // MARK: - Indent visual tests

  @Test("Indent unordered list item and verify nested rendering")
  func indentUnorderedVisual() {
    let initial = EditorState(markdown: "- First\n- Second\n- Third", selection: .cursor(16))

    let events: [EditorEvent] = [
      .indent,  // indent "- Second" to "    - Second"
      .setSelection(.cursor(0)),  // move cursor away to see bullet rendering
    ]

    let results = EditorTestHarness.run(
      name: "indent-unordered-visual",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 3)

    // After indent, "- Second" should be nested
    #expect(results[1].state.markdown == "- First\n    - Second\n- Third")

    // Visual should change between before and after indent
    #expect(
      results[0].bitmapHash != results[1].bitmapHash,
      "Indenting should produce a visual change")

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath))
    }
  }

  @Test("Outdent nested unordered list item and verify it returns to top level")
  func outdentUnorderedVisual() {
    let initial = EditorState(
      markdown: "- First\n    - Second\n- Third", selection: .cursor(20))

    let events: [EditorEvent] = [
      .outdent,  // outdent "    - Second" back to "- Second"
      .setSelection(.cursor(0)),  // move cursor away
    ]

    let results = EditorTestHarness.run(
      name: "outdent-unordered-visual",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 3)
    #expect(results[1].state.markdown == "- First\n- Second\n- Third")

    // Visual should change
    #expect(
      results[0].bitmapHash != results[1].bitmapHash,
      "Outdenting should produce a visual change")

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath))
    }
  }

  // MARK: - Ordered list indent/outdent with renumbering

  @Test("Indent ordered list item and verify renumbering")
  func indentOrderedWithRenumbering() {
    let initial = EditorState(
      markdown: "1. First\n2. Second\n3. Third", selection: .cursor(14))

    let events: [EditorEvent] = [
      .indent,  // indent "2. Second" -- should renumber to "1. Second" as sub-list
      .setSelection(.cursor(0)),  // move cursor away
    ]

    let results = EditorTestHarness.run(
      name: "indent-ordered-renumber",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 3)

    // After indent, "2. Second" should be indented and renumbered
    let markdown = results[1].state.markdown
    #expect(markdown.contains("    1. Second"))
    // "3. Third" should be renumbered to "2. Third"
    #expect(markdown.contains("2. Third"))

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath))
    }
  }

  @Test("Outdent ordered list item and verify renumbering")
  func outdentOrderedWithRenumbering() {
    let initial = EditorState(
      markdown: "1. First\n    1. Second\n2. Third", selection: .cursor(21))

    let events: [EditorEvent] = [
      .outdent,  // outdent "    1. Second" -- should renumber
      .setSelection(.cursor(0)),
    ]

    let results = EditorTestHarness.run(
      name: "outdent-ordered-renumber",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 3)

    let markdown = results[1].state.markdown
    // "1. Second" should be at top level now, renumbered as "2. Second"
    #expect(markdown.contains("2. Second"))
    // "2. Third" should become "3. Third"
    #expect(markdown.contains("3. Third"))

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath))
    }
  }

  // MARK: - Tab outside list

  @Test("Tab outside list inserts spaces visually")
  func tabOutsideListVisual() {
    let initial = EditorState(markdown: "Hello", selection: .cursor(5))

    let events: [EditorEvent] = [
      .indent,  // should insert 4 spaces at cursor
    ]

    let results = EditorTestHarness.run(
      name: "tab-outside-list",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 200))

    #expect(results.count == 2)
    #expect(results[1].state.markdown == "Hello    ")
    #expect(results[1].state.selection == .cursor(9))

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath))
    }
  }

  // MARK: - Determinism

  @Test("Fresh render matches incremental render after indent")
  func determinismAfterIndent() {
    let initial = EditorState(markdown: "- First\n- Second\n- Third", selection: .cursor(16))
    let events: [EditorEvent] = [.indent]

    let results = EditorTestHarness.run(
      name: "determinism-indent",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 300))

    let finalState = results.last!.state
    let freshBitmap = SnapshotCapture.capture(
      text: finalState.markdown,
      cursorPosition: finalState.selection.head,
      size: NSSize(width: 600, height: 300))

    let incrementalBitmap = NSBitmapImageRep(
      data: try! Data(contentsOf: URL(fileURLWithPath: results.last!.imagePath)))!

    let comparison = BitmapComparator.compare(freshBitmap, incrementalBitmap)
    #expect(comparison.isMatch, "Fresh and incremental renders must match after indent")
  }

  @Test("Fresh render matches incremental render after outdent")
  func determinismAfterOutdent() {
    let initial = EditorState(
      markdown: "- First\n    - Second\n- Third", selection: .cursor(20))
    let events: [EditorEvent] = [.outdent]

    let results = EditorTestHarness.run(
      name: "determinism-outdent",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 300))

    let finalState = results.last!.state
    let freshBitmap = SnapshotCapture.capture(
      text: finalState.markdown,
      cursorPosition: finalState.selection.head,
      size: NSSize(width: 600, height: 300))

    let incrementalBitmap = NSBitmapImageRep(
      data: try! Data(contentsOf: URL(fileURLWithPath: results.last!.imagePath)))!

    let comparison = BitmapComparator.compare(freshBitmap, incrementalBitmap)
    #expect(comparison.isMatch, "Fresh and incremental renders must match after outdent")
  }

  // MARK: - Kitchen sink with indent

  @Test("Kitchen sink: indent and outdent in mixed list")
  func kitchenSinkIndentOutdent() {
    // Build a multi-item list, indent some, outdent, then check various cursor positions
    let markdown = "- First\n- Second\n- Third\n\nSome body text"
    let initial = EditorState(markdown: markdown, selection: .cursor(16))

    let events: [EditorEvent] = [
      .indent,  // indent "- Second"
      .setSelection(.cursor(7)),  // cursor on "- First"
      .setSelection(.cursor(24)),  // cursor on "- Third" (after indent, positions shifted)
      .setSelection(.cursor(30)),  // cursor on body text
      .setSelection(.cursor(20)),  // back on indented "- Second"
      .outdent,  // outdent it back
      .setSelection(.cursor(30)),  // cursor on body text
    ]

    let results = EditorTestHarness.run(
      name: "kitchen-sink-indent-outdent",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 400))

    // Should have initial + 7 events = 8 steps
    #expect(results.count == 8)

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath))
    }

    // After indent (step 1), Second should be nested
    #expect(results[1].state.markdown.contains("    - Second"))

    // After outdent (step 6), Second should be back at top level
    #expect(!results[6].state.markdown.contains("    - Second"))
    #expect(results[6].state.markdown.contains("- Second"))
  }
}
