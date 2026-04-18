import AppKit
import Foundation
import Testing

@testable import MarkdownEditor

@Suite("Inline Code Visual Tests")
@MainActor
struct InlineCodeVisualTests {

  // MARK: - Typing flow

  @Test("Type '`code`' character by character")
  func typeInlineCode() {
    let results = EditorTestHarness.runTyping(
      name: "inline-code-typing",
      characters: "`code`")

    // Initial + 6 characters = 7 steps
    #expect(results.count == 7)

    let finalState = results.last!.state
    #expect(finalState.markdown == "`code`")
    #expect(finalState.selection == .cursor(6))

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }

    // Step 5 ("`code") and step 6 ("`code`") should differ because
    // the closing backtick creates valid inline code.
    #expect(
      results[5].bitmapHash != results[6].bitmapHash,
      "Inline code recognition should produce visual change")
  }

  // MARK: - Cursor movement

  @Test("Cursor inside inline code reveals backticks, cursor outside hides them")
  func inlineCodeDelimiterVisibility() {
    let markdown = "hello `code` world"
    let initial = EditorState(markdown: markdown, selection: .cursor(8))  // inside code

    let events: [EditorEvent] = [
      .setSelection(.cursor(0)),  // move outside
      .setSelection(.cursor(8)),  // move back inside
    ]

    let results = EditorTestHarness.run(
      name: "inline-code-delimiter-visibility",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 200))

    #expect(results.count == 3)

    // Step 0 (inside) vs step 1 (outside) should differ
    #expect(
      results[0].bitmapHash != results[1].bitmapHash,
      "Cursor inside vs outside inline code should look different")

    // Step 2 (back inside) should match step 0
    #expect(
      results[0].bitmapHash == results[2].bitmapHash,
      "Same cursor position should produce same visual")
  }

  // MARK: - Cursor at many positions

  @Test("Inline code cursor at many positions")
  func inlineCodeCursorPositions() {
    let markdown = "text `code` more"
    // Positions:
    // 0: before everything
    // 4: just before space before backtick
    // 5: on opening backtick (at node start)
    // 6: first char of content
    // 8: middle of content
    // 10: last char of content
    // 11: on closing backtick
    // 12: just after closing backtick (at node end + 1... but actually still node end)
    // 13: space after
    // 16: end

    let positions = [0, 4, 5, 6, 8, 10, 11, 12, 13, 16]
    let initial = EditorState(markdown: markdown, selection: .cursor(positions[0]))
    var events: [EditorEvent] = []
    for pos in positions.dropFirst() {
      events.append(.setSelection(.cursor(pos)))
    }

    let results = EditorTestHarness.run(
      name: "inline-code-cursor-positions",
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

  @Test("Fresh render matches incremental render for inline code")
  func determinismInlineCode() {
    let results = EditorTestHarness.runTyping(
      name: "determinism-inline-code",
      characters: "`code`")

    let finalState = results.last!.state
    let freshBitmap = SnapshotCapture.capture(
      text: finalState.markdown,
      cursorPosition: finalState.selection.head,
      size: NSSize(width: 600, height: 200))

    let incrementalBitmap = NSBitmapImageRep(
      data: try! Data(contentsOf: URL(fileURLWithPath: results.last!.imagePath)))!

    let comparison = BitmapComparator.compare(freshBitmap, incrementalBitmap)
    #expect(comparison.isMatch, "Fresh and incremental renders must match for inline code")
  }
}
