import AppKit
import Testing

@testable import MarkdownEditor

/// Phase 2 of the TextKit 2 migration — validates the
/// `NSTextContentStorageDelegate` that replaces TextKit 1 glyph hiding /
/// glyph substitution. Asserts that paragraphs are vended with display
/// strings reflecting the spec's hidden / bullet / checkbox index sets.
///
/// **Length-matching invariant.** With the substitution-based approach,
/// `displayString.length == sourceRange.length` for every paragraph. Hidden
/// chars become `U+200B` (zero-width space), bullet markers become `•`,
/// checkboxes become `☐` / `☒` followed by two `U+200B` pads. The contract
/// these tests pin: visible glyphs land at the same display offset as
/// their source position, and total length matches.
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

  /// Single character that the delegate emits for every hidden source char.
  /// Visually invisible, takes one UTF-16 code unit so the display string's
  /// length equals the source range's length.
  private static let ZWSP = "\u{200B}"

  // MARK: - Tests

  @Test
  func heading_paragraph_displays_with_zwsp_for_hidden_prefix() throws {
    let c = Self.makeComponents()
    let markdown = "# Heading line\n\nbody paragraph"
    // Cursor in the body paragraph (outside the heading) so the renderer
    // *hides* the `# ` delimiter rather than coloring it.
    let bodyOffset = ("# Heading line\n\n" as NSString).length
    let display = Self.renderAndCollectDisplay(
      markdown: markdown, cursorPosition: bodyOffset, components: c)

    // The heading paragraph starts at source 0. Its source `# Heading line\n`
    // is length 15. Hidden chars `# ` (positions 0,1) become two ZWSPs;
    // `Heading line` and trailing `\n` pass through. Total display length
    // matches source length exactly — the length-matching invariant.
    let expected = Self.ZWSP + Self.ZWSP + "Heading line\n"
    #expect(display[0] == expected, "got: \(String(describing: display[0]))")
    let headingRange = NSRange(location: 0, length: ("# Heading line\n" as NSString).length)
    let displayLen = ((display[0] ?? "") as NSString).length
    #expect(
      displayLen == headingRange.length,
      "display length \(displayLen) must equal source length \(headingRange.length)")
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

    // First bullet line starts at source 0; `-` becomes `•` (1-for-1).
    // Display length matches the 13-char source length.
    #expect(display[0]?.hasPrefix("\u{2022} first item") == true,
      "expected bullet substitution, got: \(String(describing: display[0]))")
    let bulletRange = NSRange(location: 0, length: ("- first item\n" as NSString).length)
    let displayLen = ((display[0] ?? "") as NSString).length
    #expect(
      displayLen == bulletRange.length,
      "display length \(displayLen) must equal source length \(bulletRange.length)")
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
    // `- [ ] task one\n` is 15 chars. `- ` is bullet (1-for-1) + visible
    // space. `[ ]` is 3 source chars → ☐ + ZWSP + ZWSP (3 display chars).
    // Total display length must equal source length.
    let firstRange = NSRange(location: 0, length: ("- [ ] task one\n" as NSString).length)
    let displayLen = (first as NSString).length
    #expect(
      displayLen == firstRange.length,
      "checkbox paragraph length must match source: display=\(displayLen) source=\(firstRange.length)")
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
    let firstRange = NSRange(location: 0, length: ("- [x] done\n" as NSString).length)
    let displayLen = (first as NSString).length
    #expect(
      displayLen == firstRange.length,
      "checked-checkbox paragraph length must match source: display=\(displayLen) source=\(firstRange.length)")
  }

  @Test
  func body_paragraph_passes_through_unchanged() throws {
    let c = Self.makeComponents()
    let display = Self.renderAndCollectDisplay(
      markdown: "Just plain body text.", components: c)
    // No hidden / bullet / checkbox indexes at all → pass-through is exact.
    #expect(display[0] == "Just plain body text.")
  }

  @Test
  func multiple_headings_each_get_zwsp_substitution_for_their_prefix() throws {
    let c = Self.makeComponents()
    let markdown = "# Heading 1\n\n## Heading 2\n\nbody"
    let bodyOffset = ("# Heading 1\n\n## Heading 2\n\n" as NSString).length
    let display = Self.renderAndCollectDisplay(
      markdown: markdown, cursorPosition: bodyOffset, components: c)

    // First heading at source 0: `# ` is 2 chars hidden → 2 ZWSPs prefix.
    let h1Expected = Self.ZWSP + Self.ZWSP + "Heading 1\n"
    #expect(display[0] == h1Expected, "got: \(String(describing: display[0]))")
    // Second heading after "# Heading 1\n\n": `## ` is 3 chars → 3 ZWSPs.
    let h2Start = ("# Heading 1\n\n" as NSString).length
    let h2Expected = Self.ZWSP + Self.ZWSP + Self.ZWSP + "Heading 2\n"
    #expect(display[h2Start] == h2Expected, "got: \(String(describing: display[h2Start]))")
  }

  // MARK: - Length-matching invariant (the structural fix)

  /// For every paragraph the delegate vends, the display string must have
  /// the same UTF-16 length as the source range it covers. This is the
  /// invariant TK2's NSTextLocation model assumes everywhere — when it
  /// holds, hit-test, navigation, and rendering attributes work without
  /// per-paragraph translation.
  @Test
  func length_matching_invariant_holds_for_every_vended_paragraph() {
    let c = Self.makeComponents()
    let markdown = """
      # Heading 1

      ## Heading 2

      Body paragraph with **bold** and *italic* and `code`.

      - bullet item one
      - bullet item two

      - [ ] unchecked task
      - [x] checked task

      > blockquote line
      """
    let cursorPosition = (markdown as NSString).length
    let textView = c.textView
    textView.string = markdown
    textView.setSelectedRange(NSRange(location: cursorPosition, length: 0))
    let spec = MarkdownRenderer.render(
      text: markdown, cursorRange: NSRange(location: cursorPosition, length: 0),
      style: .default)
    TextKit2RenderApplicator.apply(spec, to: textView)
    if let tlm = textView.textLayoutManager {
      tlm.ensureLayout(for: tlm.documentRange)
    }
    let cs = c.contentStorage

    var checked = 0
    if let tlm = textView.textLayoutManager {
      tlm.enumerateTextLayoutFragments(
        from: tlm.documentRange.location, options: []
      ) { frag in
        guard let element = frag.textElement,
          let paragraph = element as? NSTextParagraph,
          let elemRange = element.elementRange
        else { return true }
        let start = cs.offset(from: cs.documentRange.location, to: elemRange.location)
        let length = cs.offset(from: elemRange.location, to: elemRange.endLocation)
        let displayLen = (paragraph.attributedString.string as NSString).length
        #expect(
          displayLen == length,
          "paragraph at source [\(start), \(start + length)) — display length \(displayLen), source length \(length)")
        checked += 1
        return true
      }
    }
    #expect(checked > 0, "should have enumerated at least one paragraph")
  }
}
