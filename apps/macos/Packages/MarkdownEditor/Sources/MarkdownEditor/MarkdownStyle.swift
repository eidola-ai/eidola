import AppKit
import SwiftUI

/// Theme configuration for markdown rendering.
///
/// Consumed by `MarkdownRenderer` and `RenderApplicator`. The `MarkdownEditor`
/// view reads a `MarkdownStyle` from the SwiftUI environment, so callers can
/// override fonts, colors, and spacing with the `.markdownStyle(_:)` modifier
/// or build a custom style from scratch.
public struct MarkdownStyle: Equatable, @unchecked Sendable {
  public static let `default` = MarkdownStyle()

  // MARK: - Base Typography

  /// The base font for body text. All other sizes are derived relative to this.
  public var baseFont: NSFont = .systemFont(ofSize: 15)

  /// Multiplier applied to every line's natural height (1.0 = font metrics only).
  public var lineHeightMultiple: CGFloat = 1.4

  /// Color for regular text.
  public var textColor: NSColor = .labelColor

  /// Color for structural delimiters (`#`, `**`, etc.) when the cursor is inside.
  public var delimiterColor: NSColor = .tertiaryLabelColor

  /// Left padding before the blockquote border line.
  public var blockquoteBorderLeftPadding: CGFloat = 6

  // MARK: - Block Spacing

  /// Default space after a body paragraph (points).
  public var paragraphSpacing: CGFloat = 8

  /// Space before H1–H2 headings.
  public var headingSpacingBeforeMajor: CGFloat = 24

  /// Space before H3–H6 headings.
  public var headingSpacingBeforeMinor: CGFloat = 16

  /// Space after any heading.
  public var headingSpacingAfter: CGFloat = 8

  /// Space after each list item.
  public var listItemSpacing: CGFloat = 2

  /// Extra space before a paragraph that follows a list or blockquote.
  /// Added on top of the normal `paragraphSpacing` to visually separate
  /// the end of a container block from the next body paragraph.
  public var spacingAfterContainerBlock: CGFloat = 16

  /// Space before/after a fenced code block.
  public var codeBlockSpacing: CGFloat = 6

  // MARK: - Headings

  public func headingFont(level: Int) -> NSFont {
    let baseSize = baseFont.pointSize
    let sizes: [CGFloat] = [
      baseSize * 1.85, baseSize * 1.45,
      baseSize * 1.2, baseSize * 1.1,
      baseSize, baseSize * 0.95,
    ]
    let size = sizes[min(level - 1, sizes.count - 1)]
    return .systemFont(ofSize: size, weight: level <= 2 ? .bold : .semibold)
  }

  func headingAttributes(level: Int) -> [NSAttributedString.Key: Any] {
    let font = headingFont(level: level)
    let paragraphStyle = NSMutableParagraphStyle()
    paragraphStyle.paragraphSpacingBefore =
      level <= 2 ? headingSpacingBeforeMajor : headingSpacingBeforeMinor
    paragraphStyle.paragraphSpacing = headingSpacingAfter
    return [
      .font: font,
      .foregroundColor: textColor,
      .paragraphStyle: paragraphStyle.copy() as! NSParagraphStyle,
    ]
  }

  // MARK: - Lists

  /// Indentation per nesting level for list items.
  public var listIndent: CGFloat = 20

  /// Build list item attributes with `headIndent` matching the actual rendered
  /// marker width so wrapped/continuation lines align with the content start.
  ///
  /// - `markerText`: The text that is actually displayed as the marker.
  ///   For unordered items this is `"•"` (the bullet glyph that replaces `- `).
  ///   For ordered items this is the full marker like `"1. "` or `"22. "`.
  /// - `baseIndent`: Extra indentation from enclosing block constructs (e.g. blockquotes).
  func listItemAttributes(indentLevel: Int, markerText: String = "•", baseIndent: CGFloat = 0)
    -> [NSAttributedString.Key: Any]
  {
    let paragraphStyle = NSMutableParagraphStyle()
    let bulletPosition = baseIndent + listIndent * CGFloat(indentLevel)
    let markerWidth = (markerText as NSString).size(withAttributes: [.font: baseFont]).width
    paragraphStyle.firstLineHeadIndent = bulletPosition
    paragraphStyle.headIndent = bulletPosition + markerWidth
    paragraphStyle.paragraphSpacing = listItemSpacing
    return [
      .font: baseFont,
      .foregroundColor: textColor,
      .paragraphStyle: paragraphStyle.copy() as! NSParagraphStyle,
    ]
  }

  // MARK: - Strikethrough

  var strikethroughAttributes: [NSAttributedString.Key: Any] {
    [
      .strikethroughStyle: NSUnderlineStyle.single.rawValue
    ]
  }

  // MARK: - Inline Code

  public var codeFont: NSFont {
    .monospacedSystemFont(ofSize: baseFont.pointSize - 1.5, weight: .regular)
  }
  public var codeBackgroundColor: NSColor = .quaternaryLabelColor.withAlphaComponent(0.5)

  var inlineCodeAttributes: [NSAttributedString.Key: Any] {
    [
      .font: codeFont,
      .backgroundColor: codeBackgroundColor,
    ]
  }

  // MARK: - Code Blocks

  func codeBlockAttributes(baseIndent: CGFloat = 0) -> [NSAttributedString.Key: Any] {
    let paragraphStyle = NSMutableParagraphStyle()
    paragraphStyle.headIndent = 12 + baseIndent
    paragraphStyle.firstLineHeadIndent = 12 + baseIndent
    // Negative tailIndent extends the paragraph to the right margin.
    paragraphStyle.tailIndent = -12
    paragraphStyle.paragraphSpacing = codeBlockSpacing
    paragraphStyle.paragraphSpacingBefore = codeBlockSpacing
    // Note: .backgroundColor is NOT set here. Full-width code block backgrounds
    // are drawn by CodeBlockBackgroundLayoutManager, which uses the line fragment
    // rect (full container width) instead of the glyph extent. This ensures
    // hidden fence lines get the same background as content lines.
    return [
      .font: codeFont,
      .paragraphStyle: paragraphStyle.copy() as! NSParagraphStyle,
    ]
  }

  // MARK: - Links

  var linkColor: NSColor { .linkColor }

  func linkAttributes(destination: String?) -> [NSAttributedString.Key: Any] {
    var attrs: [NSAttributedString.Key: Any] = [
      .foregroundColor: linkColor,
      .underlineStyle: NSUnderlineStyle.single.rawValue,
    ]
    if let destination = destination, let url = URL(string: destination) {
      attrs[.link] = url
    }
    return attrs
  }

  // MARK: - Images

  var imageColor: NSColor { .secondaryLabelColor }

  var imageAttributes: [NSAttributedString.Key: Any] {
    [
      .foregroundColor: imageColor
    ]
  }

  // MARK: - Thematic Breaks

  var thematicBreakColor: NSColor { .separatorColor }

  /// Attributes for thematic break when cursor is outside: transparent text + thick strikethrough.
  /// This creates a horizontal line effect without custom drawing.
  var thematicBreakAttributes: [NSAttributedString.Key: Any] {
    [
      .foregroundColor: NSColor.clear,
      .strikethroughStyle: NSUnderlineStyle.thick.rawValue,
      .strikethroughColor: thematicBreakColor,
    ]
  }

  // MARK: - Blockquotes

  public var blockquoteIndent: CGFloat = 20

  func blockquoteAttributes(depth: Int = 1, cursorInside: Bool = false, baseIndent: CGFloat = 0)
    -> [NSAttributedString.Key: Any]
  {
    let paragraphStyle = NSMutableParagraphStyle()
    if cursorInside {
      paragraphStyle.headIndent = baseIndent
      paragraphStyle.firstLineHeadIndent = baseIndent
    } else {
      let totalIndent = baseIndent + blockquoteIndent * CGFloat(depth)
      paragraphStyle.headIndent = totalIndent
      paragraphStyle.firstLineHeadIndent = totalIndent
    }
    paragraphStyle.paragraphSpacing = paragraphSpacing
    return [
      .foregroundColor: NSColor.secondaryLabelColor,
      .paragraphStyle: paragraphStyle.copy() as! NSParagraphStyle,
    ]
  }

  /// Paragraph spacing used in the base attributes and matched by constructs
  /// (e.g. thematic breaks) to avoid vertical shifts.
  var baseParagraphSpacing: CGFloat { paragraphSpacing }

  var baseAttributes: [NSAttributedString.Key: Any] {
    let ps = NSMutableParagraphStyle()
    ps.paragraphSpacing = baseParagraphSpacing
    ps.lineHeightMultiple = lineHeightMultiple
    return [
      .font: baseFont,
      .foregroundColor: textColor,
      .paragraphStyle: ps.copy() as! NSParagraphStyle,
    ]
  }

  func textWidth(_ text: String, font: NSFont? = nil) -> CGFloat {
    (text as NSString).size(withAttributes: [.font: font ?? baseFont]).width
  }

  public static func == (lhs: MarkdownStyle, rhs: MarkdownStyle) -> Bool {
    lhs.baseFont == rhs.baseFont
      && lhs.lineHeightMultiple == rhs.lineHeightMultiple
      && lhs.textColor == rhs.textColor
      && lhs.delimiterColor == rhs.delimiterColor
      && lhs.paragraphSpacing == rhs.paragraphSpacing
      && lhs.headingSpacingBeforeMajor == rhs.headingSpacingBeforeMajor
      && lhs.headingSpacingBeforeMinor == rhs.headingSpacingBeforeMinor
      && lhs.headingSpacingAfter == rhs.headingSpacingAfter
      && lhs.listItemSpacing == rhs.listItemSpacing
      && lhs.spacingAfterContainerBlock == rhs.spacingAfterContainerBlock
      && lhs.listIndent == rhs.listIndent
      && lhs.codeBlockSpacing == rhs.codeBlockSpacing
      && lhs.codeBackgroundColor == rhs.codeBackgroundColor
      && lhs.blockquoteIndent == rhs.blockquoteIndent
      && lhs.blockquoteBorderLeftPadding == rhs.blockquoteBorderLeftPadding
  }
}

// MARK: - SwiftUI Environment

private struct MarkdownStyleKey: EnvironmentKey {
  static let defaultValue = MarkdownStyle.default
}

extension EnvironmentValues {
  /// The markdown editor style used by `MarkdownEditor` views.
  public var markdownStyle: MarkdownStyle {
    get { self[MarkdownStyleKey.self] }
    set { self[MarkdownStyleKey.self] = newValue }
  }
}

extension View {
  /// Sets the markdown editor style for all `MarkdownEditor` views in this hierarchy.
  public func markdownStyle(_ style: MarkdownStyle) -> some View {
    environment(\.markdownStyle, style)
  }
}
