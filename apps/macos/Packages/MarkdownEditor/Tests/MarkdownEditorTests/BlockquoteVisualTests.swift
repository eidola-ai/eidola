import AppKit
import Foundation
import Testing

@testable import MarkdownEditor

@Suite("Blockquote Visual Tests")
@MainActor
struct BlockquoteVisualTests {

  // MARK: - Typing flow

  @Test("Type a single-line blockquote character by character")
  func typeSingleLineBlockquote() {
    let results = EditorTestHarness.runTyping(
      name: "blockquote-typing-single",
      characters: "> Hello world",
      size: NSSize(width: 600, height: 200))

    let finalState = results.last!.state
    #expect(finalState.markdown == "> Hello world")

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }
  }

  @Test("Type a multi-line blockquote")
  func typeMultiLineBlockquote() {
    // Enter after "> Line one" auto-continues with "> ", so we only type "Line two"
    let results = EditorTestHarness.runTyping(
      name: "blockquote-typing-multi",
      characters: "> Line one\nLine two",
      size: NSSize(width: 600, height: 200))

    let finalState = results.last!.state
    #expect(finalState.markdown == "> Line one\n> Line two")

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }
  }

  // MARK: - Cursor movement

  @Test("Cursor inside blockquote reveals > prefix, cursor outside hides it")
  func blockquoteDelimiterVisibility() {
    let markdown = "> Hello world\n\nBody text here"
    let initial = EditorState(markdown: markdown, selection: .cursor(5))  // inside blockquote

    let events: [EditorEvent] = [
      .setSelection(.cursor(20)),  // move to body (outside)
      .setSelection(.cursor(5)),  // move back inside
    ]

    let results = EditorTestHarness.run(
      name: "blockquote-delimiter-visibility",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 200))

    #expect(results.count == 3)

    // Step 0 (inside) vs step 1 (outside) should differ
    #expect(
      results[0].bitmapHash != results[1].bitmapHash,
      "Cursor inside vs outside blockquote should look different")

    // Step 2 (back inside) should match step 0
    #expect(
      results[0].bitmapHash == results[2].bitmapHash,
      "Same cursor position should produce same visual")
  }

  // MARK: - Cursor at many positions

  @Test("Blockquote cursor at many positions")
  func blockquoteCursorPositions() {
    let markdown = "text\n\n> First line\n> Second line\n\nmore"
    // text = "text\n\n> First line\n> Second line\n\nmore"
    // Positions:
    // 0: start of "text"
    // 2: middle of "text"
    // 5: blank line
    // 6: at > of first line
    // 7: at space after > of first line
    // 8: start of "First" content
    // 12: middle of "First line"
    // 18: at end of first blockquote line (just before \n)
    // 19: at > of second line
    // 21: start of "Second" content
    // 27: middle of "Second line"
    // 31: at end of second blockquote line
    // 32: blank line after blockquote
    // 33: start of "more"

    let ns = markdown as NSString
    let positions = [0, 2, 5, 6, 7, 8, 12, 18, 19, 21, 27, 31, 32, min(33, ns.length)]
    let initial = EditorState(markdown: markdown, selection: .cursor(positions[0]))
    var events: [EditorEvent] = []
    for pos in positions.dropFirst() {
      events.append(.setSelection(.cursor(pos)))
    }

    let results = EditorTestHarness.run(
      name: "blockquote-cursor-positions",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 300))

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

  @Test("Fresh render matches incremental render for blockquote")
  func determinismBlockquote() {
    let results = EditorTestHarness.runTyping(
      name: "determinism-blockquote",
      characters: "> Hello world",
      size: NSSize(width: 600, height: 200))

    let finalState = results.last!.state
    let freshBitmap = SnapshotCapture.capture(
      text: finalState.markdown,
      cursorPosition: finalState.selection.head,
      size: NSSize(width: 600, height: 200))

    let incrementalBitmap = NSBitmapImageRep(
      data: try! Data(contentsOf: URL(fileURLWithPath: results.last!.imagePath)))!

    let comparison = BitmapComparator.compare(freshBitmap, incrementalBitmap)
    #expect(comparison.isMatch, "Fresh and incremental renders must match for blockquote")
  }

  // MARK: - Blockquote with surrounding content

  @Test("Blockquote surrounded by text")
  func blockquoteWithSurroundingText() {
    let markdown = "Before text\n\n> Quoted content\n\nAfter text"
    let initial = EditorState(markdown: markdown, selection: .cursor(0))

    let events: [EditorEvent] = [
      .setSelection(.cursor(16)),  // inside blockquote content
      .setSelection(.cursor(0)),  // back outside
      .setSelection(.cursor(35)),  // in "After text"
    ]

    let results = EditorTestHarness.run(
      name: "blockquote-with-surrounding",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 4)

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }

    // Outside (step 0) vs inside blockquote (step 1) should differ
    #expect(
      results[0].bitmapHash != results[1].bitmapHash,
      "Outside vs inside blockquote should look different")
  }

  // MARK: - Multi-line cursor movement

  @Test("Multi-line blockquote: cursor on different lines")
  func multiLineBlockquoteCursorMovement() {
    let markdown = "> Line one\n> Line two\n> Line three\n\nBody"
    let initial = EditorState(markdown: markdown, selection: .cursor(5))  // first line

    let events: [EditorEvent] = [
      .setSelection(.cursor(15)),  // second line
      .setSelection(.cursor(27)),  // third line
      .setSelection(.cursor(38)),  // body (outside)
      .setSelection(.cursor(5)),  // back to first line
    ]

    let results = EditorTestHarness.run(
      name: "blockquote-multiline-cursor",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 5)

    // All inside positions (0,1,2) should look the same (all > revealed)
    #expect(
      results[0].bitmapHash == results[1].bitmapHash,
      "Cursor on first vs second line should look the same")
    #expect(
      results[1].bitmapHash == results[2].bitmapHash,
      "Cursor on second vs third line should look the same")

    // Outside (step 3) should look different from inside (step 0)
    #expect(
      results[0].bitmapHash != results[3].bitmapHash,
      "Inside vs outside should look different")

    // Back inside (step 4) should match step 0
    #expect(
      results[0].bitmapHash == results[4].bitmapHash,
      "Same position should produce same visual")
  }
}
