import AppKit
import Foundation
import Testing

@testable import MarkdownEditor

@Suite("Link Visual Tests")
@MainActor
struct LinkVisualTests {

  // MARK: - Typing flow

  @Test("Type '[link](https://example.com)' character by character")
  func typeLink() {
    let results = EditorTestHarness.runTyping(
      name: "link-typing",
      characters: "[link](https://example.com)")

    let finalState = results.last!.state
    #expect(finalState.markdown == "[link](https://example.com)")
    #expect(finalState.selection == .cursor(27))

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }

    // The closing `)` should trigger link recognition, producing a visual change
    let secondToLast = results[results.count - 2]
    let last = results[results.count - 1]
    #expect(
      secondToLast.bitmapHash != last.bitmapHash,
      "Link recognition should produce visual change when closing ) is typed")
  }

  // MARK: - Cursor movement

  @Test("Cursor inside link reveals delimiters, cursor outside hides them")
  func linkDelimiterVisibility() {
    let markdown = "hello [link](https://example.com) world"
    let initial = EditorState(markdown: markdown, selection: .cursor(9))  // inside link text

    let events: [EditorEvent] = [
      .setSelection(.cursor(0)),  // move outside
      .setSelection(.cursor(9)),  // move back inside
    ]

    let results = EditorTestHarness.run(
      name: "link-delimiter-visibility",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 200))

    #expect(results.count == 3)

    // Step 0 (inside) vs step 1 (outside) should differ
    #expect(
      results[0].bitmapHash != results[1].bitmapHash,
      "Cursor inside vs outside link should look different")

    // Step 2 (back inside) should match step 0
    #expect(
      results[0].bitmapHash == results[2].bitmapHash,
      "Same cursor position should produce same visual")
  }

  // MARK: - Cursor at many positions

  @Test("Link cursor at many positions")
  func linkCursorPositions() {
    let markdown = "text [link](https://example.com) more"
    // "[link](https://example.com)" starts at position 5
    // Positions to test:
    // 0: before everything (outside)
    // 4: just before space before link (outside)
    // 5: on opening [ (at node start)
    // 6: first char of link text
    // 8: middle of link text
    // 9: last char of link text
    // 10: on ] (closing delimiter start)
    // 11: on (
    // 20: middle of URL
    // 31: on closing )
    // 32: on space after link (at node end)
    // 36: end of text (outside)

    let positions = [0, 4, 5, 6, 8, 9, 10, 11, 20, 31, 32, 36]
    let initial = EditorState(markdown: markdown, selection: .cursor(positions[0]))
    var events: [EditorEvent] = []
    for pos in positions.dropFirst() {
      events.append(.setSelection(.cursor(pos)))
    }

    let results = EditorTestHarness.run(
      name: "link-cursor-positions",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 200))

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

  @Test("Fresh render matches incremental render for link")
  func determinismLink() {
    let results = EditorTestHarness.runTyping(
      name: "determinism-link",
      characters: "[link](https://example.com)")

    let finalState = results.last!.state
    let freshBitmap = SnapshotCapture.capture(
      text: finalState.markdown,
      cursorPosition: finalState.selection.head,
      size: NSSize(width: 600, height: 200))

    let incrementalBitmap = NSBitmapImageRep(
      data: try! Data(contentsOf: URL(fileURLWithPath: results.last!.imagePath)))!

    let comparison = BitmapComparator.compare(freshBitmap, incrementalBitmap)
    #expect(comparison.isMatch, "Fresh and incremental renders must match for link")
  }
}
