import AppKit
import Testing

@testable import MarkdownEditor

/// Phase 2 of the TextKit 2 migration — validates the
/// `NSTextContentStorageDelegate` that replaces TextKit 1 glyph hiding /
/// glyph substitution. Asserts that paragraphs are vended with display
/// strings reflecting the spec's hidden / bullet / checkbox index sets.
///
/// These tests build a TK2 NSTextView inline (rather than going through the
/// SwiftUI `MarkdownEditor`) so they can assert on the NSTextParagraph the
/// delegate produces directly, without relying on the viewport layout pass
/// which is unreliable in pure XCTest (per Phase 0 spike findings).
@MainActor
struct TextKit2GlyphHidingTests {

  // MARK: - Helpers

  private struct TK2Components {
    let textView: NSTextView
    let contentStorage: NSTextContentStorage
    let delegate: TextKit2ContentStorageDelegate
    let window: NSWindow
  }

  private static func makeComponents(
    size: NSSize = NSSize(width: 600, height: 400)
  ) -> TK2Components {
    // Spike 1 found that NSTextView(usingTextLayoutManager: true) is the only
    // reliable way to set up a TK2 stack — manual NSTextContentStorage wiring
    // doesn't link textStorage to contentStorage correctly.
    let textView = NSTextView(usingTextLayoutManager: true)
    textView.frame = NSRect(origin: .zero, size: size)
    textView.minSize = NSSize(width: 0, height: 0)
    textView.maxSize = NSSize(
      width: CGFloat.greatestFiniteMagnitude,
      height: CGFloat.greatestFiniteMagnitude)
    textView.font = MarkdownStyle.default.baseFont
    textView.isRichText = true
    textView.isAutomaticQuoteSubstitutionEnabled = false
    textView.textContainer?.containerSize = NSSize(
      width: size.width, height: CGFloat.greatestFiniteMagnitude)
    textView.textContainer?.widthTracksTextView = true

    let delegate = TextKit2ContentStorageDelegate()
    textView.textContentStorage?.delegate = delegate

    let window = NSWindow(
      contentRect: NSRect(origin: .zero, size: size),
      styleMask: .borderless, backing: .buffered, defer: true)
    window.contentView = textView

    return TK2Components(
      textView: textView,
      contentStorage: textView.textContentStorage!,
      delegate: delegate, window: window)
  }

  /// Drive the renderer + applicator end-to-end, then enumerate the laid-out
  /// paragraphs and return their display strings keyed on source-range start.
  private static func renderAndCollectDisplay(
    markdown: String, cursorPosition: Int = 0,
    style: MarkdownStyle = .default,
    components: TK2Components
  ) -> [Int: String] {
    let textView = components.textView
    textView.string = markdown
    let cursorRange = NSRange(location: cursorPosition, length: 0)
    textView.setSelectedRange(cursorRange)

    let spec = MarkdownRenderer.render(
      text: markdown, cursorRange: cursorRange, style: style)
    TextKit2RenderApplicator.apply(spec, to: textView)

    // Force the layout manager to ask the content delegate for paragraphs.
    if let tlm = textView.textLayoutManager {
      tlm.ensureLayout(for: tlm.documentRange)
    }

    var byStart: [Int: String] = [:]
    let cs = components.contentStorage
    if let tlm = textView.textLayoutManager {
      tlm.enumerateTextLayoutFragments(
        from: tlm.documentRange.location, options: []
      ) { frag in
        guard let element = frag.textElement,
          let paragraph = element as? NSTextParagraph
        else { return true }
        let elemRange = element.elementRange ?? frag.rangeInElement
        let start = cs.offset(from: cs.documentRange.location, to: elemRange.location)
        byStart[start] = paragraph.attributedString.string
        return true
      }
    }
    return byStart
  }

  // MARK: - Tests

  @Test
  func heading_paragraph_displays_without_hash_prefix() throws {
    let c = Self.makeComponents()
    let markdown = "# Heading line\n\nbody paragraph"
    // Cursor in the body paragraph (outside the heading) so the renderer
    // *hides* the `# ` delimiter rather than coloring it.
    let bodyOffset = ("# Heading line\n\n" as NSString).length
    let display = Self.renderAndCollectDisplay(
      markdown: markdown, cursorPosition: bodyOffset, components: c)

    // The heading paragraph starts at source 0 and should display without
    // the `# ` prefix.
    #expect(display[0] == "Heading line\n", "got: \(String(describing: display[0]))")
    // Hidden-prefix length for the heading should be exactly 2 (`# `).
    let headingRange = NSRange(location: 0, length: ("# Heading line\n" as NSString).length)
    #expect(c.delegate.computeHiddenPrefix(forParagraphSourceRange: headingRange) == 2)
  }

  @Test
  func bullet_marker_substitutes_to_unicode_bullet() throws {
    let c = Self.makeComponents()
    let markdown = "- first item\n- second item\n\nbody"
    // Cursor in body paragraph (outside the list items) so the renderer emits
    // bullet substitution rather than delimiter coloring.
    let bodyOffset = ("- first item\n- second item\n\n" as NSString).length
    let display = Self.renderAndCollectDisplay(
      markdown: markdown, cursorPosition: bodyOffset, components: c)

    // First bullet line starts at source 0; should display with `•` for `-`.
    #expect(display[0]?.hasPrefix("\u{2022} first item") == true,
      "expected bullet substitution, got: \(String(describing: display[0]))")
    // No hidden prefix on a substituted bullet line — `-` is a bullet
    // (substituted, not hidden), and the space after is visible.
    let bulletRange = NSRange(location: 0, length: ("- first item\n" as NSString).length)
    #expect(c.delegate.computeHiddenPrefix(forParagraphSourceRange: bulletRange) == 0)
  }

  @Test
  func unchecked_checkbox_substitutes_to_ballot_box() throws {
    let c = Self.makeComponents()
    let markdown = "- [ ] task one\n- [x] task two\n\nbody"
    let bodyOffset = ("- [ ] task one\n- [x] task two\n\n" as NSString).length
    let display = Self.renderAndCollectDisplay(
      markdown: markdown, cursorPosition: bodyOffset, components: c)

    let first = display[0] ?? ""
    #expect(first.contains("\u{25A1}"),
      "expected unchecked checkbox glyph in: \(first)")
  }

  @Test
  func checked_checkbox_substitutes_to_x_ballot_box() throws {
    let c = Self.makeComponents()
    let markdown = "- [x] done\n\nbody"
    let bodyOffset = ("- [x] done\n\n" as NSString).length
    let display = Self.renderAndCollectDisplay(
      markdown: markdown, cursorPosition: bodyOffset, components: c)

    let first = display[0] ?? ""
    #expect(first.contains("\u{2612}"),
      "expected checked checkbox glyph in: \(first)")
  }

  @Test
  func body_paragraph_passes_through_unchanged() throws {
    let c = Self.makeComponents()
    let display = Self.renderAndCollectDisplay(
      markdown: "Just plain body text.", components: c)
    #expect(display[0] == "Just plain body text.")
    let bodyRange = NSRange(location: 0, length: ("Just plain body text." as NSString).length)
    #expect(c.delegate.computeHiddenPrefix(forParagraphSourceRange: bodyRange) == 0)
  }

  @Test
  func multiple_headings_each_track_their_own_prefix() throws {
    let c = Self.makeComponents()
    let markdown = "# Heading 1\n\n## Heading 2\n\nbody"
    let bodyOffset = ("# Heading 1\n\n## Heading 2\n\n" as NSString).length
    _ = Self.renderAndCollectDisplay(
      markdown: markdown, cursorPosition: bodyOffset, components: c)

    // First heading at source 0: `# ` is 2 chars hidden.
    let h1Range = NSRange(location: 0, length: ("# Heading 1\n" as NSString).length)
    #expect(c.delegate.computeHiddenPrefix(forParagraphSourceRange: h1Range) == 2)
    // Second heading after "# Heading 1\n\n": `## ` is 3 chars.
    let h2Start = ("# Heading 1\n\n" as NSString).length
    let h2Range = NSRange(location: h2Start, length: ("## Heading 2\n" as NSString).length)
    #expect(c.delegate.computeHiddenPrefix(forParagraphSourceRange: h2Range) == 3)
  }
}
