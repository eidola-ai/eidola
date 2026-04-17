import AppKit
import Foundation
import Testing

@testable import MarkdownEditor

/// Visual tests for the 4 list bug fixes.
@Suite("List Bug Fix Visual Tests")
@MainActor
struct ListBugFixVisualTests {

  // MARK: - Nested list indentation visual

  @Test("Nested unordered list indentation visual")
  func nestedUnorderedListVisual() {
    let markdown = """
      - Top level item with some text that might wrap
      - Another top level item
        - Nested item one
        - Nested item two with longer text
          - Deeply nested item
      - Back to top level
      """
    let initial = EditorState(markdown: markdown, selection: .cursor(0))

    let events: [EditorEvent] = [
      .setSelection(.cursor(50)),  // second top-level item
      .setSelection(.cursor(80)),  // nested item
      .setSelection(.cursor(120)), // deeply nested
    ]

    let results = EditorTestHarness.run(
      name: "nested-unordered-indent",
      initial: initial,
      events: events,
      size: NSSize(width: 400, height: 400))

    #expect(results.count == 4)
    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath))
    }
  }

  // MARK: - Multi-line ordered list renumbering visual

  @Test("Multi-line ordered list items preserve numbering visually")
  func multiLineOrderedListVisual() {
    // Type a multi-line ordered list with continuation and verify numbers are correct
    var events: [EditorEvent] = []
    for c in "1. First item" { events.append(.insertText(String(c))) }
    events.append(.insertLineBreak)  // Shift+Return for continuation
    for c in "continued text" { events.append(.insertText(String(c))) }
    events.append(.insertNewline)  // Return for new item
    for c in "Second item" { events.append(.insertText(String(c))) }
    events.append(.insertNewline)  // Return for new item
    for c in "Third item" { events.append(.insertText(String(c))) }

    let results = EditorTestHarness.run(
      name: "multiline-ordered-renumber",
      initial: EditorState(),
      events: events,
      size: NSSize(width: 400, height: 300))

    let finalState = results.last!.state
    #expect(finalState.markdown == "1. First item\n   continued text\n2. Second item\n3. Third item")

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath))
    }
  }

  // MARK: - Shift+Return on continuation visual

  @Test("Shift+Return on continuation line visual")
  func shiftReturnContinuationVisual() {
    var events: [EditorEvent] = []
    for c in "- First line" { events.append(.insertText(String(c))) }
    events.append(.insertLineBreak)  // creates continuation
    for c in "second line" { events.append(.insertText(String(c))) }
    events.append(.insertLineBreak)  // Shift+Return ON the continuation line
    for c in "third line" { events.append(.insertText(String(c))) }

    let results = EditorTestHarness.run(
      name: "shiftreturn-on-continuation",
      initial: EditorState(),
      events: events,
      size: NSSize(width: 400, height: 300))

    let finalState = results.last!.state
    #expect(finalState.markdown == "- First line\n  second line\n  third line")

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath))
    }
  }

  // MARK: - Return on continuation creates new item visual

  @Test("Return on continuation creates new list item visual")
  func returnOnContinuationVisual() {
    var events: [EditorEvent] = []
    for c in "- First line" { events.append(.insertText(String(c))) }
    events.append(.insertLineBreak)  // creates continuation
    for c in "continued" { events.append(.insertText(String(c))) }
    events.append(.insertNewline)  // Return on continuation -> new item
    for c in "Second item" { events.append(.insertText(String(c))) }

    let results = EditorTestHarness.run(
      name: "return-on-continuation",
      initial: EditorState(),
      events: events,
      size: NSSize(width: 400, height: 300))

    let finalState = results.last!.state
    #expect(finalState.markdown == "- First line\n  continued\n- Second item")

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath))
    }
  }

  @Test("Return on ordered continuation creates next numbered item visual")
  func returnOnOrderedContinuationVisual() {
    var events: [EditorEvent] = []
    for c in "1. First item" { events.append(.insertText(String(c))) }
    events.append(.insertLineBreak)  // creates continuation
    for c in "continued" { events.append(.insertText(String(c))) }
    events.append(.insertNewline)  // Return on continuation -> new item "2. "
    for c in "Second item" { events.append(.insertText(String(c))) }

    let results = EditorTestHarness.run(
      name: "return-on-ordered-continuation",
      initial: EditorState(),
      events: events,
      size: NSSize(width: 400, height: 300))

    let finalState = results.last!.state
    #expect(finalState.markdown == "1. First item\n   continued\n2. Second item")

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath))
    }
  }
}
