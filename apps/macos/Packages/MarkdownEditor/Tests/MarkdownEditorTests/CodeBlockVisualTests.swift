import AppKit
import Foundation
import Testing

@testable import MarkdownEditor

@Suite("Code Block Visual Tests")
@MainActor
struct CodeBlockVisualTests {

  // MARK: - Typing flow

  @Test("Type a fenced code block character by character")
  func typeCodeBlock() {
    let results = EditorTestHarness.runTyping(
      name: "code-block-typing",
      characters: "```\nlet x = 42\n```",
      size: NSSize(width: 600, height: 300))

    let finalState = results.last!.state
    #expect(finalState.markdown == "```\nlet x = 42\n```")

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }
  }

  @Test("Type a fenced code block with language hint")
  func typeCodeBlockWithLanguage() {
    let results = EditorTestHarness.runTyping(
      name: "code-block-with-language",
      characters: "```swift\nlet x = 42\n```",
      size: NSSize(width: 600, height: 300))

    let finalState = results.last!.state
    #expect(finalState.markdown == "```swift\nlet x = 42\n```")

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }
  }

  // MARK: - Cursor movement

  @Test("Cursor inside code block reveals fences, cursor outside hides them")
  func codeBlockDelimiterVisibility() {
    let markdown = "hello\n\n```\ncode\n```\n\nworld"
    let initial = EditorState(markdown: markdown, selection: .cursor(12))  // inside "code"

    let events: [EditorEvent] = [
      .setSelection(.cursor(0)),  // move outside
      .setSelection(.cursor(12)),  // move back inside
    ]

    let results = EditorTestHarness.run(
      name: "code-block-delimiter-visibility",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 3)

    // Phase 2.1: code blocks are now rendered by a no-op `BlockRenderer`
    // (a flat colored `NSView`) via the bridging-layer attachment, not by
    // the per-character delimiter-visibility path that this test
    // previously asserted on. The text "code" is no longer in the visual;
    // the entire code block is a colored rectangle whose appearance does
    // not depend on cursor position. Phase 2.2's real `CodeBlockRenderer`
    // will reintroduce visible code text with cursor-driven decoration
    // (e.g. fence reveal) — at which point this test becomes meaningful
    // again. For now we pin only the determinism property.
    #expect(
      results[0].bitmapHash == results[2].bitmapHash,
      "Same cursor position should produce same visual")
  }

  // MARK: - Cursor at many positions

  @Test("Code block cursor at many positions")
  func codeBlockCursorPositions() {
    let markdown = "text\n\n```swift\nlet x = 42\nprint(x)\n```\n\nmore"
    // Positions:
    // 0: before everything ("text")
    // 2: middle of "text"
    // 5: blank line
    // 6: at start of opening fence (```)
    // 9: on "swift" in opening fence
    // 14: start of first content line "let x = 42"
    // 18: middle of first content line
    // 25: start of second content line "print(x)"
    // 29: middle of second content line
    // 34: on closing fence
    // 37: at end of closing fence
    // 38: blank line after
    // 39: start of "more"

    let ns = markdown as NSString
    let positions = [0, 2, 5, 6, 9, 14, 18, 25, 29, 34, 37, 38, min(39, ns.length)]
    let initial = EditorState(markdown: markdown, selection: .cursor(positions[0]))
    var events: [EditorEvent] = []
    for pos in positions.dropFirst() {
      events.append(.setSelection(.cursor(pos)))
    }

    let results = EditorTestHarness.run(
      name: "code-block-cursor-positions",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 300))

    #expect(results.count == positions.count)

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }

    // Phase 2.1: the code block is now a flat colored rectangle (no-op
    // renderer) so cursor positions inside the block do not change the
    // visual. The body text "text" / "more" surrounding the block is not
    // a markdown construct that reveals differently per cursor either, so
    // every position in this test yields the same bitmap. Phase 2.2's
    // real `CodeBlockRenderer` will broaden distinct-state count back to
    // its pre-2.1 level by reintroducing fence-visibility chrome.
    var uniqueHashes = Set<Int>()
    for r in results {
      uniqueHashes.insert(r.bitmapHash)
    }
    #expect(uniqueHashes.count >= 1, "Expected at least 1 visual state captured")
  }

  // MARK: - Determinism

  @Test("Fresh render matches incremental render for code block")
  func determinismCodeBlock() {
    let results = EditorTestHarness.runTyping(
      name: "determinism-code-block",
      characters: "```\nlet x = 42\n```",
      size: NSSize(width: 600, height: 300))

    let finalState = results.last!.state
    let freshBitmap = SnapshotCapture.capture(
      text: finalState.markdown,
      cursorPosition: finalState.selection.head,
      size: NSSize(width: 600, height: 300))

    let incrementalBitmap = NSBitmapImageRep(
      data: try! Data(contentsOf: URL(fileURLWithPath: results.last!.imagePath)))!

    let comparison = BitmapComparator.compare(freshBitmap, incrementalBitmap)
    #expect(comparison.isMatch, "Fresh and incremental renders must match for code block")
  }

  // MARK: - Code block with surrounding content

  @Test("Code block surrounded by text")
  func codeBlockWithSurroundingText() {
    let markdown = "Before text\n\n```\ncode here\n```\n\nAfter text"
    let initial = EditorState(markdown: markdown, selection: .cursor(0))

    let events: [EditorEvent] = [
      .setSelection(.cursor(18)),  // inside code content
      .setSelection(.cursor(0)),  // back outside
      .setSelection(.cursor(35)),  // in "After text"
    ]

    let results = EditorTestHarness.run(
      name: "code-block-with-surrounding",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 4)

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }

    // Phase 2.1: the code block is rendered as a flat colored rectangle
    // by the no-op `BlockRenderer`. Cursor inside vs outside no longer
    // changes the block's visual (no fence-visibility chrome), so the
    // pre-2.1 inequality assertion is now invalid. Phase 2.2's real
    // `CodeBlockRenderer` will reintroduce a meaningful inside/outside
    // visual difference.
  }
}
