import AppKit
import Foundation
import Testing

@testable import MarkdownEditor

/// Visual integration tests for Shift+Return (insertLineBreak) behavior.
@Suite("Line Break Visual Tests")
@MainActor
struct LineBreakVisualTests {

  // MARK: - Typing flow with Shift+Return

  @Test("Type list item, Shift+Return, continue typing -- single list item with continuation")
  func typeListItemWithContinuation() {
    var events: [EditorEvent] = []
    for c in "- First line" { events.append(.insertText(String(c))) }
    events.append(.insertLineBreak)
    for c in "second line" { events.append(.insertText(String(c))) }

    let results = EditorTestHarness.run(
      name: "linebreak-unordered-continuation",
      initial: EditorState(),
      events: events,
      size: NSSize(width: 600, height: 300))

    let finalState = results.last!.state
    #expect(finalState.markdown == "- First line\n  second line")

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }
  }

  @Test("Type ordered list item, Shift+Return, continue typing")
  func typeOrderedListItemWithContinuation() {
    var events: [EditorEvent] = []
    for c in "1. First line" { events.append(.insertText(String(c))) }
    events.append(.insertLineBreak)
    for c in "second line" { events.append(.insertText(String(c))) }

    let results = EditorTestHarness.run(
      name: "linebreak-ordered-continuation",
      initial: EditorState(),
      events: events,
      size: NSSize(width: 600, height: 300))

    let finalState = results.last!.state
    #expect(finalState.markdown == "1. First line\n   second line")

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }
  }

  // MARK: - Regular Enter vs Shift+Return comparison

  @Test("Regular Enter creates new list item, Shift+Return continues same item")
  func enterVsShiftReturn() {
    // Sequence: type item, Shift+Return (continuation), Enter (new item)
    var events: [EditorEvent] = []
    for c in "- Item content" { events.append(.insertText(String(c))) }
    events.append(.insertLineBreak)  // continuation
    for c in "more content" { events.append(.insertText(String(c))) }
    events.append(.insertNewline)  // new list item
    for c in "Next item" { events.append(.insertText(String(c))) }

    let results = EditorTestHarness.run(
      name: "linebreak-enter-vs-shiftreturn",
      initial: EditorState(),
      events: events,
      size: NSSize(width: 600, height: 300))

    let finalState = results.last!.state
    // When Enter is pressed on a continuation line, it creates a new list
    // item since the continuation line belongs to the parent "- " item.
    #expect(finalState.markdown == "- Item content\n  more content\n- Next item")

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }
  }

  // MARK: - Shift+Return outside list

  @Test("Shift+Return outside list is paragraph break visual")
  func shiftReturnOutsideListVisual() {
    var events: [EditorEvent] = []
    for c in "Hello" { events.append(.insertText(String(c))) }
    events.append(.insertLineBreak)
    for c in "World" { events.append(.insertText(String(c))) }

    let results = EditorTestHarness.run(
      name: "linebreak-outside-list",
      initial: EditorState(),
      events: events,
      size: NSSize(width: 600, height: 200))

    let finalState = results.last!.state
    #expect(finalState.markdown == "Hello\n\nWorld")

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }
  }

  // MARK: - Kitchen sink update: multi-line list item

  @Test("Multi-line list item in kitchen sink document")
  func kitchenSinkMultiLineListItem() {
    let markdown =
      "# Heading\n\n- Item one\n  continued here\n- Item two\n\n1. Ordered one\n   continued\n2. Ordered two\n\nBody text"
    let positions: [(String, Int)] = [
      ("inside-continuation-unordered", 25),  // inside "continued here"
      ("inside-first-item", 15),  // inside "Item one"
      ("inside-ordered-continuation", 65),  // inside ordered "continued"
      ("body", 95),  // in body text
    ]

    var events: [EditorEvent] = []
    for (_, offset) in positions.dropFirst() {
      events.append(.setSelection(.cursor(offset)))
    }

    let initial = EditorState(
      markdown: markdown, selection: .cursor(positions[0].1))

    let results = EditorTestHarness.run(
      name: "linebreak-kitchen-sink",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 400))

    #expect(results.count == positions.count)

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }
  }

  // MARK: - Determinism

  @Test("Fresh render matches incremental render for multi-line list item")
  func determinismMultiLineListItem() {
    var events: [EditorEvent] = []
    for c in "- First line" { events.append(.insertText(String(c))) }
    events.append(.insertLineBreak)
    for c in "second line" { events.append(.insertText(String(c))) }

    let results = EditorTestHarness.run(
      name: "determinism-linebreak",
      initial: EditorState(),
      events: events,
      size: NSSize(width: 600, height: 200))

    let finalState = results.last!.state
    let freshBitmap = SnapshotCapture.capture(
      text: finalState.markdown,
      cursorPosition: finalState.selection.head,
      size: NSSize(width: 600, height: 200))

    let incrementalBitmap = NSBitmapImageRep(
      data: try! Data(contentsOf: URL(fileURLWithPath: results.last!.imagePath)))!

    let comparison = BitmapComparator.compare(freshBitmap, incrementalBitmap)
    #expect(
      comparison.isMatch, "Fresh and incremental renders must match for multi-line list item")
  }
}
