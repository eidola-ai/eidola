import AppKit
import Foundation
import Testing

@testable import MarkdownEditor

@Suite("Image Visual Tests")
@MainActor
struct ImageVisualTests {

  // MARK: - Typing flow

  @Test("Type '![alt](https://example.com/img.png)' character by character")
  func typeImage() {
    let results = EditorTestHarness.runTyping(
      name: "image-typing",
      characters: "![alt](https://example.com/img.png)")

    let finalState = results.last!.state
    #expect(finalState.markdown == "![alt](https://example.com/img.png)")
    #expect(finalState.selection == .cursor(35))

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }

    // The closing `)` should trigger image recognition, producing a visual change
    let secondToLast = results[results.count - 2]
    let last = results[results.count - 1]
    #expect(
      secondToLast.bitmapHash != last.bitmapHash,
      "Image recognition should produce visual change when closing ) is typed")
  }

  // MARK: - Cursor movement

  @Test("Cursor inside image reveals delimiters, cursor outside hides them")
  func imageDelimiterVisibility() {
    let markdown = "hello ![photo](https://example.com/img.png) world"
    let initial = EditorState(markdown: markdown, selection: .cursor(10))  // inside alt text

    let events: [EditorEvent] = [
      .setSelection(.cursor(0)),  // move outside
      .setSelection(.cursor(10)),  // move back inside
    ]

    let results = EditorTestHarness.run(
      name: "image-delimiter-visibility",
      initial: initial,
      events: events,
      size: NSSize(width: 700, height: 200))

    #expect(results.count == 3)

    // Step 0 (inside) vs step 1 (outside) should differ
    #expect(
      results[0].bitmapHash != results[1].bitmapHash,
      "Cursor inside vs outside image should look different")

    // Step 2 (back inside) should match step 0
    #expect(
      results[0].bitmapHash == results[2].bitmapHash,
      "Same cursor position should produce same visual")
  }

  // MARK: - Cursor at many positions

  @Test("Image cursor at many positions")
  func imageCursorPositions() {
    let markdown = "text ![photo](https://example.com/img.png) more"
    // "![photo](https://example.com/img.png)" starts at position 5
    // Positions to test:
    // 0: before everything (outside)
    // 4: just before space before image (outside)
    // 5: on opening ! (at node start)
    // 6: on [
    // 7: first char of alt text
    // 9: middle of alt text
    // 11: last char of alt text
    // 12: on ] (closing delimiter start)
    // 13: on (
    // 25: middle of URL
    // 41: on closing )
    // 42: on space after image (at node end)
    // 46: end of text (outside)

    let positions = [0, 4, 5, 6, 7, 9, 11, 12, 13, 25, 41, 42, 46]
    let initial = EditorState(markdown: markdown, selection: .cursor(positions[0]))
    var events: [EditorEvent] = []
    for pos in positions.dropFirst() {
      events.append(.setSelection(.cursor(pos)))
    }

    let results = EditorTestHarness.run(
      name: "image-cursor-positions",
      initial: initial,
      events: events,
      size: NSSize(width: 700, height: 200))

    #expect(results.count == positions.count)

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }

    // Verify at least some positions produce different visuals
    var uniqueHashes = Set<Int>()
    for r in results {
      uniqueHashes.insert(r.bitmapHash)
    }
    #expect(uniqueHashes.count >= 2, "Expected at least 2 visually distinct states")
  }

  // MARK: - Determinism

  @Test("Fresh render matches incremental render for image")
  func determinismImage() {
    let results = EditorTestHarness.runTyping(
      name: "determinism-image",
      characters: "![alt](https://example.com/img.png)")

    let finalState = results.last!.state
    let freshBitmap = SnapshotCapture.capture(
      text: finalState.markdown,
      cursorPosition: finalState.selection.head,
      size: NSSize(width: 600, height: 200))

    let incrementalBitmap = NSBitmapImageRep(
      data: try! Data(contentsOf: URL(fileURLWithPath: results.last!.imagePath)))!

    let comparison = BitmapComparator.compare(freshBitmap, incrementalBitmap)
    #expect(comparison.isMatch, "Fresh and incremental renders must match for image")
  }
}
