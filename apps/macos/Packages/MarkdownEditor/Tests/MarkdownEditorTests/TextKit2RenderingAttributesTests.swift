import AppKit
import Testing

@testable import MarkdownEditor

/// Phase 4 of the TextKit 2 migration — validates cursor-driven delimiter
/// dimming via `NSTextLayoutManager.setRenderingAttributes(_:for:)`.
///
/// The renderer emits `temporaryAttributes` (typically `.foregroundColor:
/// delimiterColor`) for the delimiter ranges of any markdown construct the
/// cursor is currently inside — heading, bold/italic, link, etc. The TK2
/// applicator writes those into the layout manager's rendering-attribute
/// store; cursor moves trigger a re-apply that refreshes them.
@MainActor
struct TextKit2RenderingAttributesTests {

  // MARK: - Helpers

  private struct TK2Components {
    let textView: NSTextView
    let contentDelegate: TextKit2ContentStorageDelegate
    let window: NSWindow
  }

  private static func makeComponents(
    size: NSSize = NSSize(width: 600, height: 400)
  ) -> TK2Components {
    let textView = NSTextView(usingTextLayoutManager: true)
    textView.frame = NSRect(origin: .zero, size: size)
    textView.minSize = NSSize(width: 0, height: 0)
    textView.maxSize = NSSize(
      width: CGFloat.greatestFiniteMagnitude,
      height: CGFloat.greatestFiniteMagnitude)
    textView.font = MarkdownStyle.default.baseFont
    textView.isRichText = true
    textView.textContainer?.containerSize = NSSize(
      width: size.width, height: CGFloat.greatestFiniteMagnitude)
    textView.textContainer?.widthTracksTextView = true

    let contentDelegate = TextKit2ContentStorageDelegate()
    textView.textContentStorage?.delegate = contentDelegate

    let window = NSWindow(
      contentRect: NSRect(origin: .zero, size: size),
      styleMask: .borderless, backing: .buffered, defer: true)
    window.contentView = textView

    return TK2Components(
      textView: textView, contentDelegate: contentDelegate, window: window)
  }

  /// Apply a render at the given cursor position and return all rendering
  /// attributes currently set on the layout manager, keyed on the source
  /// character offset of the range start.
  private static func renderAndCollectRenderingAttributes(
    markdown: String, cursorPosition: Int,
    components: TK2Components
  ) -> [(NSRange, [NSAttributedString.Key: Any])] {
    let textView = components.textView
    textView.string = markdown
    let cursorRange = NSRange(location: cursorPosition, length: 0)
    textView.setSelectedRange(cursorRange)
    let spec = MarkdownRenderer.render(
      text: markdown, cursorRange: cursorRange, style: .default)
    TextKit2RenderApplicator.apply(spec, to: textView)

    guard let tlm = textView.textLayoutManager,
      let storage = textView.textContentStorage
    else { return [] }

    var collected: [(NSRange, [NSAttributedString.Key: Any])] = []
    tlm.enumerateRenderingAttributes(
      from: tlm.documentRange.location, reverse: false
    ) { _, attrs, range in
      guard !attrs.isEmpty else { return true }
      let start = storage.offset(from: storage.documentRange.location, to: range.location)
      let length = storage.offset(from: range.location, to: range.endLocation)
      collected.append((NSRange(location: start, length: length), attrs))
      return true
    }
    return collected
  }

  // MARK: - Tests

  @Test
  func cursor_outside_heading_emits_no_rendering_attrs_for_delimiter() throws {
    let c = Self.makeComponents()
    let markdown = "# Heading\n\nbody"
    // Cursor in body — heading delimiters get hidden via the content delegate,
    // not coloured via rendering attributes.
    let bodyOffset = ("# Heading\n\n" as NSString).length
    let attrs = Self.renderAndCollectRenderingAttributes(
      markdown: markdown, cursorPosition: bodyOffset, components: c)

    // No rendering attributes should overlap the delimiter range 0..2.
    let delimiterRange = NSRange(location: 0, length: 2)
    for (range, _) in attrs {
      #expect(NSIntersectionRange(range, delimiterRange).length == 0,
        "expected no rendering attrs over `# `, found range \(range)")
    }
  }

  @Test
  func cursor_inside_heading_dims_the_hash_delimiter() throws {
    let c = Self.makeComponents()
    let markdown = "# Heading\n\nbody"
    // Cursor on `H` of "Heading" — inside the heading paragraph, so the
    // renderer emits temporaryAttributes for the `# ` delimiter.
    let attrs = Self.renderAndCollectRenderingAttributes(
      markdown: markdown, cursorPosition: 5, components: c)

    // At least one rendering attribute should overlap the delimiter range 0..2
    // and carry the delimiter color.
    let delimiterRange = NSRange(location: 0, length: 2)
    let expectedColor = MarkdownStyle.default.delimiterColor
    var sawDelimiterColor = false
    for (range, dict) in attrs {
      guard NSIntersectionRange(range, delimiterRange).length > 0 else { continue }
      if let color = dict[.foregroundColor] as? NSColor, color == expectedColor {
        sawDelimiterColor = true
      }
    }
    #expect(sawDelimiterColor,
      "expected delimiter color over `# ` when cursor inside heading; got \(attrs)")
  }

  @Test
  func cursor_inside_bold_dims_the_asterisk_pairs() throws {
    let c = Self.makeComponents()
    // "Some **bold** text" — cursor on the `b` of "bold" (inside the bold span).
    let markdown = "Some **bold** text"
    let cursorOffset = ("Some **" as NSString).length
    let attrs = Self.renderAndCollectRenderingAttributes(
      markdown: markdown, cursorPosition: cursorOffset, components: c)

    // The two `**` pairs at offsets 5..7 and 11..13 should carry delimiter color.
    let openingDelim = NSRange(location: 5, length: 2)
    let closingDelim = NSRange(location: 11, length: 2)
    let expectedColor = MarkdownStyle.default.delimiterColor

    var openingDimmed = false
    var closingDimmed = false
    for (range, dict) in attrs {
      guard let color = dict[.foregroundColor] as? NSColor, color == expectedColor
      else { continue }
      if NSIntersectionRange(range, openingDelim).length > 0 { openingDimmed = true }
      if NSIntersectionRange(range, closingDelim).length > 0 { closingDimmed = true }
    }
    #expect(openingDimmed, "expected opening `**` dimmed; got \(attrs)")
    #expect(closingDimmed, "expected closing `**` dimmed; got \(attrs)")
  }

  @Test
  func cursor_move_clears_stale_rendering_attrs() throws {
    let c = Self.makeComponents()
    let markdown = "# Heading\n\n**bold** body"
    let textView = c.textView

    // Place cursor inside heading first → expect `# ` dimmed.
    textView.string = markdown
    let firstCursor = NSRange(location: 5, length: 0)  // inside "Heading"
    textView.setSelectedRange(firstCursor)
    var spec = MarkdownRenderer.render(
      text: markdown, cursorRange: firstCursor, style: .default)
    TextKit2RenderApplicator.apply(spec, to: textView)

    // Now move cursor to inside the bold word → expect `# ` no longer dimmed,
    // but `**` pairs dimmed instead.
    let secondCursor = NSRange(
      location: ("# Heading\n\n**" as NSString).length + 1, length: 0)
    textView.setSelectedRange(secondCursor)
    spec = MarkdownRenderer.render(
      text: markdown, cursorRange: secondCursor, style: .default)
    TextKit2RenderApplicator.apply(spec, to: textView)

    guard let tlm = textView.textLayoutManager,
      let storage = textView.textContentStorage
    else { return }

    let expectedColor = MarkdownStyle.default.delimiterColor
    let headingDelim = NSRange(location: 0, length: 2)
    var headingStillDimmed = false
    tlm.enumerateRenderingAttributes(
      from: tlm.documentRange.location, reverse: false
    ) { _, attrs, range in
      let start = storage.offset(from: storage.documentRange.location, to: range.location)
      let length = storage.offset(from: range.location, to: range.endLocation)
      let nsRange = NSRange(location: start, length: length)
      if NSIntersectionRange(nsRange, headingDelim).length > 0 {
        if let color = attrs[.foregroundColor] as? NSColor, color == expectedColor {
          headingStillDimmed = true
        }
      }
      return true
    }
    #expect(!headingStillDimmed,
      "heading delimiter should no longer be dimmed after cursor moved to bold")
  }
}
