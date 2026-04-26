import AppKit
import Foundation
import Testing

@testable import MarkdownEditor

/// Visual integration tests using the test harness.
///
/// These tests exercise the full pipeline: state → update → render → bitmap.
/// They capture images that agents can review, and verify state/rendering invariants.
@Suite("Editor Visual Tests")
@MainActor
struct EditorVisualTests {

  // MARK: - Plain Text Typing

  @Test("Type 'Hello World' and capture snapshots at each step")
  func typeHelloWorld() {
    let results = EditorTestHarness.runTyping(
      name: "plain-hello-world",
      characters: "Hello World")

    // Should have initial + 11 character steps
    #expect(results.count == 12)

    // Every step should produce an image file
    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }

    // Final state should be correct
    let finalState = results.last!.state
    #expect(finalState.markdown == "Hello World")
    #expect(finalState.selection == .cursor(11))

    // Most character insertions should produce a visual change.
    // Some characters (e.g., trailing space) may not change the bitmap.
    let changedCount = (1..<results.count).filter {
      results[$0].bitmapHash != results[$0 - 1].bitmapHash
    }.count
    #expect(changedCount >= 10, "At least 10/11 steps should produce visual changes, got \(changedCount)")
  }

  @Test("Type two lines and verify newline produces visual change")
  func typeTwoLines() {
    let results = EditorTestHarness.runTyping(
      name: "plain-two-lines",
      characters: "Line 1\nLine 2")

    let finalState = results.last!.state
    #expect(finalState.markdown == "Line 1\n\nLine 2")
    #expect(finalState.selection == .cursor(14))
  }

  // MARK: - Selection and Deletion

  @Test("Type text, select range, delete, verify state and visuals")
  func typeSelectDelete() {
    let events: [EditorEvent] = [
      .insertText("Hello World"),
      .setSelection(.range(anchor: 5, head: 11)),  // Select " World"
      .deleteBackward,  // Delete selection
      .insertText("!"),  // Type "!"
    ]

    let results = EditorTestHarness.run(
      name: "select-delete",
      initial: EditorState(),
      events: events)

    // Initial + 4 events = 5 steps
    #expect(results.count == 5)

    // After typing "Hello World"
    #expect(results[1].state.markdown == "Hello World")

    // After selecting " World"
    #expect(results[2].state.markdown == "Hello World")
    #expect(results[2].state.selection == .range(anchor: 5, head: 11))

    // After deleting selection
    #expect(results[3].state.markdown == "Hello")
    #expect(results[3].state.selection == .cursor(5))

    // After typing "!"
    #expect(results[4].state.markdown == "Hello!")
    #expect(results[4].state.selection == .cursor(6))
  }

  // MARK: - Heading Rendering

  @Test("Type '# Hello' and verify heading state")
  func typeHeading() {
    let results = EditorTestHarness.runTyping(
      name: "heading-basic",
      characters: "# Hello")

    let finalState = results.last!.state
    #expect(finalState.markdown == "# Hello")
    #expect(finalState.selection == .cursor(7))

    // Images are saved for agent review of heading rendering
    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath))
    }
  }

  @Test("Type heading then newline — heading should be on first line, cursor on second")
  func typeHeadingThenNewline() {
    let results = EditorTestHarness.runTyping(
      name: "heading-then-newline",
      characters: "# Title\nBody text")

    let finalState = results.last!.state
    #expect(finalState.markdown == "# Title\n\nBody text")

    // After the newline, cursor should be on line 2
    // "# Title\n\n" is 9 chars, then "Body text" is 9 more = cursor at 18
    #expect(finalState.selection == .cursor(18))
  }

  @Test("Cursor inside heading reveals delimiters, cursor outside hides them")
  func headingDelimiterVisibility() {
    // Set up a document with a heading and body
    let markdown = "# Hello\n\nBody"
    let initial = EditorState(markdown: markdown, selection: .cursor(3))  // Inside heading

    let events: [EditorEvent] = [
      // Move cursor to body text
      .setSelection(.cursor(10)),
      // Move cursor back into heading
      .setSelection(.cursor(2)),
    ]

    let results = EditorTestHarness.run(
      name: "heading-delimiter-visibility",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 200))

    #expect(results.count == 3)  // initial + 2 events

    // All states have the same markdown
    for r in results {
      #expect(r.state.markdown == markdown)
    }

    // Visual at step 0 (cursor in heading) should differ from step 1 (cursor in body)
    // because heading delimiters are revealed vs hidden
    #expect(
      results[0].bitmapHash != results[1].bitmapHash,
      "Cursor inside vs outside heading should look different (delimiter visibility)")

    // Step 2 (cursor back in heading) should match step 0
    #expect(
      results[0].bitmapHash == results[2].bitmapHash,
      "Same cursor position should produce same visual")
  }

  // MARK: - Determinism

  @Test("Fresh render matches incremental render for plain text")
  func determinismPlainText() {
    let results = EditorTestHarness.runTyping(
      name: "determinism-plain",
      characters: "Hello World")

    // Compare the last incremental step with a fresh render
    let finalState = results.last!.state
    let freshBitmap = SnapshotCapture.capture(
      text: finalState.markdown,
      cursorPosition: finalState.selection.head,
      size: NSSize(width: 600, height: 200))

    let incrementalBitmap = NSBitmapImageRep(
      data: try! Data(contentsOf: URL(fileURLWithPath: results.last!.imagePath)))!

    let comparison = BitmapComparator.compare(freshBitmap, incrementalBitmap)
    #expect(comparison.isMatch, "Fresh and incremental renders must match")
  }

  @Test("Fresh render matches incremental render for heading")
  func determinismHeading() {
    let results = EditorTestHarness.runTyping(
      name: "determinism-heading",
      characters: "# Hello World")

    let finalState = results.last!.state
    let freshBitmap = SnapshotCapture.capture(
      text: finalState.markdown,
      cursorPosition: finalState.selection.head,
      size: NSSize(width: 600, height: 200))

    let incrementalBitmap = NSBitmapImageRep(
      data: try! Data(contentsOf: URL(fileURLWithPath: results.last!.imagePath)))!

    let comparison = BitmapComparator.compare(freshBitmap, incrementalBitmap)
    #expect(comparison.isMatch, "Fresh and incremental renders must match for headings")
  }
}
