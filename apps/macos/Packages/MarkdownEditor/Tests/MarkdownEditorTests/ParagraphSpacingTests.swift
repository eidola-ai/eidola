import AppKit
import Testing

@testable import MarkdownEditor

/// Tests for paragraph spacing stability when cursor enters/leaves constructs.
///
/// The core bug: when the cursor leaves a construct whose delimiter starts at the
/// beginning of a paragraph line, the paragraph's top spacing shrinks because TextKit
/// computes line fragment metrics from null glyphs differently than visible glyphs.
@Suite("Paragraph Spacing Stability Tests")
@MainActor
struct ParagraphSpacingTests {

  // MARK: - Heading Spacing Stability

  @Test("Heading spacing visual snapshot stable when cursor moves in and out")
  func headingSpacingVisualSnapshot() {
    let markdown = "Some body text\n### Features\nMore body"
    let initial = EditorState(markdown: markdown, selection: .cursor(22))  // inside "Features"
    let events: [EditorEvent] = [
      .setSelection(.cursor(5)),   // move to body (heading hidden)
      .setSelection(.cursor(22)),  // back inside heading
    ]
    let results = EditorTestHarness.run(
      name: "heading-spacing-stability",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 200))

    // step 0 (cursor inside heading) and step 2 (cursor back inside) should match
    #expect(
      results[0].bitmapHash == results[2].bitmapHash,
      "Same cursor position should produce identical rendering")
  }

  @Test("Heading line fragment rect is stable via TK2 layout fragments")
  func headingLineFragmentRectStable() {
    // Use TK2 layout fragments directly to check line fragment positioning.
    let markdown = "Some body text\n### Features"
    let size = NSSize(width: 600, height: 200)

    let components = MarkdownTextViewFactory.create(size: size)

    // Render with cursor inside heading (delimiters visible)
    SnapshotCapture.apply(text: markdown, cursorPosition: 19, to: components)
    let insideFrame = headingFragmentFrame(in: components.textView, sourceOffset: 19)

    // Render with cursor outside heading (delimiters hidden)
    SnapshotCapture.apply(text: markdown, cursorPosition: 5, to: components)
    let outsideFrame = headingFragmentFrame(in: components.textView, sourceOffset: 19)

    let yShift = abs(insideFrame.origin.y - outsideFrame.origin.y)
    #expect(
      yShift < 1.0,
      "Heading line fragment Y should not shift. Inside: \(insideFrame.origin.y), Outside: \(outsideFrame.origin.y), Shift: \(yShift)"
    )

    // The heading fragment should also have the same height (including paragraphSpacingBefore).
    let heightShift = abs(insideFrame.size.height - outsideFrame.size.height)
    #expect(
      heightShift < 1.0,
      "Heading line fragment height should not shift. Inside: \(insideFrame.size.height), Outside: \(outsideFrame.size.height), Shift: \(heightShift)"
    )
  }

  // MARK: - Italic at Line Start Spacing Stability

  @Test("Italic at line start Y position does not shift when cursor leaves")
  func italicAtLineStartYPositionStable() {
    // Document: italic text at the start of line 2
    let markdown = "Normal text\n*This is italic*"
    let size = NSSize(width: 600, height: 200)

    let components = MarkdownTextViewFactory.create(size: size)

    // Cursor inside italic — char index 13 is 'T' in "This"
    SnapshotCapture.apply(text: markdown, cursorPosition: 15, to: components)
    let insideFrame = headingFragmentFrame(in: components.textView, sourceOffset: 13)

    // Cursor outside italic
    SnapshotCapture.apply(text: markdown, cursorPosition: 5, to: components)
    let outsideFrame = headingFragmentFrame(in: components.textView, sourceOffset: 13)

    let yShift = abs(insideFrame.origin.y - outsideFrame.origin.y)
    #expect(
      yShift < 1.0,
      "Italic line fragment Y should not shift. Inside: \(insideFrame.origin.y), Outside: \(outsideFrame.origin.y), Shift: \(yShift)"
    )
  }

  // MARK: - Helpers

  /// Return the layout-fragment frame (container coordinates) for the
  /// paragraph containing the source offset `sourceOffset`. Forces full
  /// document layout first.
  private func headingFragmentFrame(in textView: NSTextView, sourceOffset: Int) -> CGRect {
    guard let tlm = textView.textLayoutManager,
      let cs = textView.textContentStorage
    else {
      return .zero
    }
    tlm.ensureLayout(for: tlm.documentRange)

    var found: CGRect = .zero
    tlm.enumerateTextLayoutFragments(
      from: tlm.documentRange.location, options: [.ensuresLayout]
    ) { frag in
      guard let elementRange = frag.textElement?.elementRange else { return true }
      let start = cs.offset(from: cs.documentRange.location, to: elementRange.location)
      let length = cs.offset(from: elementRange.location, to: elementRange.endLocation)
      if sourceOffset >= start && sourceOffset < start + length {
        found = frag.layoutFragmentFrame
        return false
      }
      return true
    }
    return found
  }
}
