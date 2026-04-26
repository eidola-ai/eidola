import AppKit
import Foundation
import Testing

@testable import MarkdownEditor

@Suite("Thematic Break Visual Tests")
@MainActor
struct ThematicBreakVisualTests {

  // MARK: - Typing flow

  @Test("Type a horizontal rule (---) character by character")
  func typeHorizontalRule() {
    let results = EditorTestHarness.runTyping(
      name: "thematic-break-typing-dashes",
      characters: "Above\n---\nBelow",
      size: NSSize(width: 600, height: 200))

    let finalState = results.last!.state
    #expect(finalState.markdown == "Above\n\n---\n\nBelow")

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }
  }

  @Test("Type asterisk horizontal rule (***)")
  func typeAsteriskRule() {
    let results = EditorTestHarness.runTyping(
      name: "thematic-break-typing-asterisks",
      characters: "Above\n***\nBelow",
      size: NSSize(width: 600, height: 200))

    let finalState = results.last!.state
    #expect(finalState.markdown == "Above\n\n***\n\nBelow")

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }
  }

  @Test("Type underscore horizontal rule (___)")
  func typeUnderscoreRule() {
    let results = EditorTestHarness.runTyping(
      name: "thematic-break-typing-underscores",
      characters: "Above\n___\nBelow",
      size: NSSize(width: 600, height: 200))

    let finalState = results.last!.state
    #expect(finalState.markdown == "Above\n\n___\n\nBelow")

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }
  }

  // MARK: - Cursor movement

  @Test("Cursor inside thematic break reveals raw text, cursor outside shows line")
  func thematicBreakDelimiterVisibility() {
    let markdown = "Above\n\n---\n\nBelow"
    let initial = EditorState(markdown: markdown, selection: .cursor(8))  // inside ---

    let events: [EditorEvent] = [
      .setSelection(.cursor(0)),  // move to "Above" (outside)
      .setSelection(.cursor(8)),  // move back inside
    ]

    let results = EditorTestHarness.run(
      name: "thematic-break-delimiter-visibility",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 200))

    #expect(results.count == 3)

    // Step 0 (inside) vs step 1 (outside) should differ
    #expect(
      results[0].bitmapHash != results[1].bitmapHash,
      "Cursor inside vs outside thematic break should look different")

    // Step 2 (back inside) should match step 0
    #expect(
      results[0].bitmapHash == results[2].bitmapHash,
      "Same cursor position should produce same visual")
  }

  // MARK: - Cursor at many positions

  @Test("Thematic break cursor at many positions")
  func thematicBreakCursorPositions() {
    let markdown = "Above text\n\n---\n\nBelow text"
    // Layout:
    // 0-9: "Above text"
    // 10: \n
    // 11: \n (blank line)
    // 12-14: "---"
    // 15: \n
    // 16: \n (blank line)
    // 17-26: "Below text"

    let ns = markdown as NSString
    let positions = [
      0,  // start of "Above text"
      5,  // middle of "Above text"
      10,  // end of "Above text" / newline
      11,  // blank line before ---
      12,  // start of ---
      13,  // middle of ---
      14,  // end of ---
      15,  // newline after ---
      16,  // blank line after ---
      17,  // start of "Below text"
      min(22, ns.length),  // middle of "Below text"
    ]

    let initial = EditorState(markdown: markdown, selection: .cursor(positions[0]))
    var events: [EditorEvent] = []
    for pos in positions.dropFirst() {
      events.append(.setSelection(.cursor(pos)))
    }

    let results = EditorTestHarness.run(
      name: "thematic-break-cursor-positions",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 300))

    #expect(results.count == positions.count)

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }

    // Verify visual diversity
    var uniqueHashes = Set<Int>()
    for r in results {
      uniqueHashes.insert(r.bitmapHash)
    }
    #expect(uniqueHashes.count >= 2, "Expected at least 2 visually distinct states")
  }

  // MARK: - Multiple thematic breaks

  @Test("Multiple thematic breaks with different markers")
  func multipleThematicBreaks() {
    let markdown = "Text\n\n---\n\nMiddle\n\n***\n\nEnd"
    let initial = EditorState(markdown: markdown, selection: .cursor(0))

    let events: [EditorEvent] = [
      .setSelection(.cursor(6)),  // on --- (inside)
      .setSelection(.cursor(15)),  // on "Middle"
      .setSelection(.cursor(22)),  // on *** (inside)
      .setSelection(.cursor(0)),  // back to "Text" (all hidden)
    ]

    let results = EditorTestHarness.run(
      name: "thematic-break-multiple",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 5)

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }
  }

  // MARK: - Determinism

  @Test("Fresh render matches incremental render for thematic break")
  func determinismThematicBreak() {
    let results = EditorTestHarness.runTyping(
      name: "determinism-thematic-break",
      characters: "Above\n---\nBelow",
      size: NSSize(width: 600, height: 200))

    let finalState = results.last!.state
    let freshBitmap = SnapshotCapture.capture(
      text: finalState.markdown,
      cursorPosition: finalState.selection.head,
      size: NSSize(width: 600, height: 200))

    let incrementalBitmap = NSBitmapImageRep(
      data: try! Data(contentsOf: URL(fileURLWithPath: results.last!.imagePath)))!

    let comparison = BitmapComparator.compare(freshBitmap, incrementalBitmap)
    #expect(comparison.isMatch, "Fresh and incremental renders must match for thematic break")
  }

  // MARK: - Thematic break with surrounding content

  @Test("Thematic break surrounded by formatted text")
  func thematicBreakWithFormattedSurroundings() {
    let markdown = "# Heading\n\n---\n\n**Bold text**"
    let initial = EditorState(markdown: markdown, selection: .cursor(0))

    let events: [EditorEvent] = [
      .setSelection(.cursor(5)),  // inside heading
      .setSelection(.cursor(12)),  // inside ---
      .setSelection(.cursor(20)),  // inside bold
      .setSelection(.cursor(0)),  // back to start
    ]

    let results = EditorTestHarness.run(
      name: "thematic-break-with-formatting",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 5)

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }
  }
}
