import AppKit
import Foundation
import Testing

@testable import MarkdownEditor

@Suite("Strikethrough Visual Tests")
@MainActor
struct StrikethroughVisualTests {

  // MARK: - Typing flow

  @Test("Type '~~struck~~' character by character")
  func typeStrikethrough() {
    let results = EditorTestHarness.runTyping(
      name: "strikethrough-typing",
      characters: "~~struck~~")

    // Initial + 10 characters = 11 steps
    #expect(results.count == 11)

    let finalState = results.last!.state
    #expect(finalState.markdown == "~~struck~~")
    #expect(finalState.selection == .cursor(10))

    // All images should exist
    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }

    // Step 9 (after typing "~~struck~") and step 10 (after typing "~~struck~~")
    // should look different because strikethrough kicks in at step 10.
    #expect(
      results[9].bitmapHash != results[10].bitmapHash,
      "Strikethrough recognition at step 10 should produce visual change")
  }

  // MARK: - Cursor movement

  @Test("Cursor inside strikethrough reveals delimiters, cursor outside hides them")
  func strikethroughDelimiterVisibility() {
    let markdown = "hello ~~struck~~ world"
    let initial = EditorState(markdown: markdown, selection: .cursor(11))  // inside strikethrough

    let events: [EditorEvent] = [
      .setSelection(.cursor(0)),   // move outside
      .setSelection(.cursor(11)),  // move back inside
    ]

    let results = EditorTestHarness.run(
      name: "strikethrough-delimiter-visibility",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 200))

    #expect(results.count == 3)

    // Step 0 (cursor inside) vs step 1 (cursor outside) should differ
    #expect(
      results[0].bitmapHash != results[1].bitmapHash,
      "Cursor inside vs outside strikethrough should look different")

    // Step 2 (cursor back inside) should match step 0
    #expect(
      results[0].bitmapHash == results[2].bitmapHash,
      "Same cursor position should produce same visual")
  }

  // MARK: - Cursor at many positions

  @Test("Strikethrough cursor at many positions")
  func strikethroughCursorPositions() {
    let markdown = "hello ~~struck~~ world"
    // Interesting positions:
    // 0: far before (outside)
    // 5: just before opening ~~ (space before, outside)
    // 6: at opening ~~ start (at node start = inside)
    // 8: inside content start
    // 11: middle of content
    // 14: inside content end
    // 16: at closing ~~ end (at node end = inside)
    // 17: just after closing ~~ (outside)
    // 22: at end of text (outside)
    let positions = [0, 5, 6, 8, 11, 14, 16, 17, 22]

    let initial = EditorState(markdown: markdown, selection: .cursor(positions[0]))
    var events: [EditorEvent] = []
    for pos in positions.dropFirst() {
      events.append(.setSelection(.cursor(pos)))
    }

    let results = EditorTestHarness.run(
      name: "strikethrough-cursor-positions",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 200))

    #expect(results.count == positions.count)

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }

    // Position 0 (outside) and position 5 (just before) should both hide delimiters
    // Position 6 (at node start, inside) should show delimiters
    #expect(
      results[0].bitmapHash != results[2].bitmapHash,
      "Outside vs at node start should look different")

    // Position 17 (just after, outside) should hide delimiters again
    #expect(
      results[0].bitmapHash == results[7].bitmapHash,
      "Outside positions should look the same (pos 0 vs pos 17)")
  }

  // MARK: - Determinism

  @Test("Fresh render matches incremental render for strikethrough text")
  func determinismStrikethrough() {
    let results = EditorTestHarness.runTyping(
      name: "determinism-strikethrough",
      characters: "~~struck~~")

    let finalState = results.last!.state
    let freshBitmap = SnapshotCapture.capture(
      text: finalState.markdown,
      cursorPosition: finalState.selection.head,
      size: NSSize(width: 600, height: 200))

    let incrementalBitmap = NSBitmapImageRep(
      data: try! Data(contentsOf: URL(fileURLWithPath: results.last!.imagePath)))!

    let comparison = BitmapComparator.compare(freshBitmap, incrementalBitmap)
    #expect(comparison.isMatch, "Fresh and incremental renders must match for strikethrough")
  }
}
