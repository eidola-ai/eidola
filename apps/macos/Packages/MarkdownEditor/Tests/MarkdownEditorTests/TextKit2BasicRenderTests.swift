import AppKit
import Testing

@testable import MarkdownEditor

/// Phase 1 of the TextKit 2 migration — validates that the parallel TK2
/// rendering path is wired up end-to-end for body-text-only rendering.
///
/// Phase 1 explicitly does not implement glyph hiding, custom fragments, or
/// rendering attributes, so this suite asserts only the subset of behavior
/// that should be in place: text appears verbatim (delimiters NOT hidden),
/// base typography is applied, and the pipeline does not crash.
@MainActor
struct TextKit2BasicRenderTests {

  /// Build an NSTextView using the same TextKit 2 setup the real
  /// MarkdownEditor coordinator uses in `makeNSView`.
  private static func makeTextKit2TextView(
    size: NSSize = NSSize(width: 600, height: 400)
  ) -> NSTextView {
    let textView = NSTextView(usingTextLayoutManager: true)
    textView.frame = NSRect(origin: .zero, size: size)
    textView.minSize = NSSize(width: 0, height: 0)
    textView.maxSize = NSSize(
      width: CGFloat.greatestFiniteMagnitude,
      height: CGFloat.greatestFiniteMagnitude)
    textView.isVerticallyResizable = true
    textView.isHorizontallyResizable = false
    textView.font = MarkdownStyle.default.baseFont
    textView.isRichText = true
    textView.isAutomaticQuoteSubstitutionEnabled = false
    textView.isAutomaticDashSubstitutionEnabled = false
    textView.isAutomaticTextReplacementEnabled = false

    textView.autoresizingMask = [.width]
    textView.textContainer?.widthTracksTextView = true
    textView.textContainer?.containerSize = NSSize(
      width: 0, height: CGFloat.greatestFiniteMagnitude)

    let window = NSWindow(
      contentRect: NSRect(origin: .zero, size: size),
      styleMask: .borderless, backing: .buffered, defer: true)
    window.contentView = textView

    return textView
  }

  @Test
  func tk2_path_renders_body_paragraph() throws {
    let textView = Self.makeTextKit2TextView()
    let markdown = "Just a plain body paragraph with no special markup."
    textView.string = markdown

    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(
      text: markdown, cursorRange: cursorRange, style: .default)
    TextKit2RenderApplicator.apply(spec, to: textView)

    #expect(textView.string == markdown)

    // Base font should be applied at the start of the storage.
    let attrs = textView.textStorage?.attributes(at: 0, effectiveRange: nil)
    let appliedFont = attrs?[.font] as? NSFont
    #expect(appliedFont != nil)
    #expect(appliedFont?.pointSize == MarkdownStyle.default.baseFont.pointSize)
  }

  @Test
  func tk2_path_does_not_hide_delimiters_in_phase_1() throws {
    // Phase 1 explicitly leaves glyph hiding to Phase 2. The source markdown
    // should appear verbatim — `# `, `**`, etc. are visible in the text view.
    let textView = Self.makeTextKit2TextView()
    let markdown = "# Heading\n\n**bold** word"
    textView.string = markdown

    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(
      text: markdown, cursorRange: cursorRange, style: .default)
    TextKit2RenderApplicator.apply(spec, to: textView)

    // The raw markdown stays in the text storage in both paths; in TK1 the
    // glyph layer hides delimiters. In Phase 1 of TK2 the glyph layer does
    // nothing, so the delimiters render visibly. We assert source-storage
    // identity here; a Phase 2 test will assert delimiters are visually hidden.
    #expect(textView.string == markdown)
  }

  @Test
  func tk2_path_applies_heading_font_via_styled_ranges() throws {
    let textView = Self.makeTextKit2TextView()
    let markdown = "# Heading line"
    textView.string = markdown

    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(
      text: markdown, cursorRange: cursorRange, style: .default)
    TextKit2RenderApplicator.apply(spec, to: textView)

    // The H1 font should be larger than the base font somewhere in the line.
    var sawLargerFont = false
    textView.textStorage?.enumerateAttribute(
      .font, in: NSRange(location: 0, length: (markdown as NSString).length),
      options: []
    ) { value, _, _ in
      if let font = value as? NSFont,
        font.pointSize > MarkdownStyle.default.baseFont.pointSize
      {
        sawLargerFont = true
      }
    }
    #expect(sawLargerFont, "expected a heading-sized font to be applied somewhere on the line")
  }

  @Test
  func tk2_applyCursorUpdate_is_a_safe_noop_in_phase_1() throws {
    // Cursor-driven updates are entirely deferred in Phase 1. The call must
    // not crash and must not mutate the text storage.
    let textView = Self.makeTextKit2TextView()
    let markdown = "Some body text."
    textView.string = markdown
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(
      text: markdown, cursorRange: cursorRange, style: .default)
    TextKit2RenderApplicator.apply(spec, to: textView)

    let snapshotBefore = textView.string
    TextKit2RenderApplicator.applyCursorUpdate(
      spec,
      previousHidden: IndexSet(),
      previousBullets: IndexSet(),
      to: textView)
    #expect(textView.string == snapshotBefore)
  }
}
