import AppKit
import Foundation
import Testing

@testable import MarkdownEditor

@Suite("Bold/Italic Visual Tests")
@MainActor
struct BoldItalicVisualTests {

  // MARK: - Typing flow

  @Test("Type '**bold**' character by character")
  func typeBold() {
    let results = EditorTestHarness.runTyping(
      name: "bold-typing",
      characters: "**bold**")

    // Initial + 8 characters = 9 steps
    #expect(results.count == 9)

    let finalState = results.last!.state
    #expect(finalState.markdown == "**bold**")
    #expect(finalState.selection == .cursor(8))

    // All images should exist
    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }

    // Step 7 (after typing "**bold*") and step 8 (after typing "**bold**")
    // should look different because bold kicks in at step 8.
    // At step 7 we have "**bold*" which is not valid bold.
    // At step 8 we have "**bold**" which is valid bold - delimiters dimmed, text bold.
    #expect(
      results[7].bitmapHash != results[8].bitmapHash,
      "Bold recognition at step 8 should produce visual change")
  }

  @Test("Type '*italic*' character by character")
  func typeItalic() {
    let results = EditorTestHarness.runTyping(
      name: "italic-typing",
      characters: "*italic*")

    #expect(results.count == 9)

    let finalState = results.last!.state
    #expect(finalState.markdown == "*italic*")
    #expect(finalState.selection == .cursor(8))

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath))
    }

    // Step 7 ("*italic") and step 8 ("*italic*") should differ
    #expect(
      results[7].bitmapHash != results[8].bitmapHash,
      "Italic recognition should produce visual change")
  }

  @Test("Type '***bold italic***' character by character")
  func typeBoldItalic() {
    let results = EditorTestHarness.runTyping(
      name: "bold-italic-typing",
      characters: "***bold italic***")

    let finalState = results.last!.state
    #expect(finalState.markdown == "***bold italic***")
    #expect(finalState.selection == .cursor(17))

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath))
    }
  }

  // MARK: - Cursor movement

  @Test("Cursor inside bold reveals delimiters, cursor outside hides them")
  func boldDelimiterVisibility() {
    let markdown = "hello **bold** world"
    let initial = EditorState(markdown: markdown, selection: .cursor(10))  // inside bold

    let events: [EditorEvent] = [
      .setSelection(.cursor(0)),   // move outside bold
      .setSelection(.cursor(10)),  // move back inside
    ]

    let results = EditorTestHarness.run(
      name: "bold-delimiter-visibility",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 200))

    #expect(results.count == 3)

    // Step 0 (cursor inside) vs step 1 (cursor outside) should differ
    #expect(
      results[0].bitmapHash != results[1].bitmapHash,
      "Cursor inside vs outside bold should look different")

    // Step 2 (cursor back inside) should match step 0
    #expect(
      results[0].bitmapHash == results[2].bitmapHash,
      "Same cursor position should produce same visual")
  }

  @Test("Cursor inside italic reveals delimiters, cursor outside hides them")
  func italicDelimiterVisibility() {
    let markdown = "hello *italic* world"
    let initial = EditorState(markdown: markdown, selection: .cursor(10))  // inside italic

    let events: [EditorEvent] = [
      .setSelection(.cursor(0)),   // move outside
      .setSelection(.cursor(10)),  // move back inside
    ]

    let results = EditorTestHarness.run(
      name: "italic-delimiter-visibility",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 200))

    #expect(results.count == 3)

    #expect(
      results[0].bitmapHash != results[1].bitmapHash,
      "Cursor inside vs outside italic should look different")
    #expect(
      results[0].bitmapHash == results[2].bitmapHash,
      "Same cursor position should produce same visual")
  }

  // MARK: - Bold inside heading

  @Test("Type '# **bold heading**' and verify rendering")
  func typeBoldInsideHeading() {
    let results = EditorTestHarness.runTyping(
      name: "bold-inside-heading",
      characters: "# **bold heading**")

    let finalState = results.last!.state
    #expect(finalState.markdown == "# **bold heading**")

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath))
    }
  }

  @Test("Bold inside heading: cursor outside hides all delimiters")
  func boldInsideHeadingCursorOutside() {
    let markdown = "# **bold heading**\n\nBody text"
    let initial = EditorState(markdown: markdown, selection: .cursor(25))  // in body

    let events: [EditorEvent] = [
      .setSelection(.cursor(8)),   // move into heading (bold content)
    ]

    let results = EditorTestHarness.run(
      name: "bold-heading-cursor-toggle",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 200))

    #expect(results.count == 2)

    // Step 0 (cursor in body, heading delimiters hidden)
    // vs step 1 (cursor in heading, delimiters revealed)
    #expect(
      results[0].bitmapHash != results[1].bitmapHash,
      "Cursor in body vs in heading should look different")
  }

  // MARK: - Determinism

  @Test("Fresh render matches incremental render for bold text")
  func determinismBold() {
    let results = EditorTestHarness.runTyping(
      name: "determinism-bold",
      characters: "**bold**")

    let finalState = results.last!.state
    let freshBitmap = SnapshotCapture.capture(
      text: finalState.markdown,
      cursorPosition: finalState.selection.head,
      size: NSSize(width: 600, height: 200))

    let incrementalBitmap = NSBitmapImageRep(
      data: try! Data(contentsOf: URL(fileURLWithPath: results.last!.imagePath)))!

    let comparison = BitmapComparator.compare(freshBitmap, incrementalBitmap)
    #expect(comparison.isMatch, "Fresh and incremental renders must match for bold")
  }

  @Test("Fresh render matches incremental render for italic text")
  func determinismItalic() {
    let results = EditorTestHarness.runTyping(
      name: "determinism-italic",
      characters: "*italic*")

    let finalState = results.last!.state
    let freshBitmap = SnapshotCapture.capture(
      text: finalState.markdown,
      cursorPosition: finalState.selection.head,
      size: NSSize(width: 600, height: 200))

    let incrementalBitmap = NSBitmapImageRep(
      data: try! Data(contentsOf: URL(fileURLWithPath: results.last!.imagePath)))!

    let comparison = BitmapComparator.compare(freshBitmap, incrementalBitmap)
    #expect(comparison.isMatch, "Fresh and incremental renders must match for italic")
  }
}
