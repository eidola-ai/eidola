import AppKit

/// Theme configuration for markdown rendering.
struct MarkdownStyle {
  static let `default` = MarkdownStyle()

  var baseFontSize: CGFloat = 15

  var baseFont: NSFont { .systemFont(ofSize: baseFontSize) }

  var textColor: NSColor { .labelColor }
  var delimiterColor: NSColor { .tertiaryLabelColor }

  func headingFont(level: Int) -> NSFont {
    let sizes: [CGFloat] = [28, 22, 18, 16, 15, 14]
    let size = sizes[min(level - 1, sizes.count - 1)]
    return .systemFont(ofSize: size, weight: level <= 2 ? .bold : .semibold)
  }

  func headingAttributes(level: Int) -> [NSAttributedString.Key: Any] {
    let font = headingFont(level: level)
    let paragraphStyle = NSMutableParagraphStyle()
    paragraphStyle.paragraphSpacingBefore = level <= 2 ? 16 : 10
    paragraphStyle.paragraphSpacing = 6
    return [
      .font: font,
      .foregroundColor: textColor,
      .paragraphStyle: paragraphStyle.copy() as! NSParagraphStyle,
    ]
  }

  /// Indentation per nesting level for list items.
  var listIndent: CGFloat = 20

  /// Build list item attributes with `headIndent` matching the actual rendered
  /// marker width so wrapped/continuation lines align with the content start.
  ///
  /// - `markerText`: The text that is actually displayed as the marker.
  ///   For unordered items this is `"•"` (the bullet glyph that replaces `- `).
  ///   For ordered items this is the full marker like `"1. "` or `"22. "`.
  func listItemAttributes(indentLevel: Int, markerText: String = "•") -> [NSAttributedString.Key: Any] {
    let paragraphStyle = NSMutableParagraphStyle()
    let bulletPosition = listIndent * CGFloat(indentLevel)
    let markerWidth = (markerText as NSString).size(withAttributes: [.font: baseFont]).width
    paragraphStyle.firstLineHeadIndent = bulletPosition
    paragraphStyle.headIndent = bulletPosition + markerWidth
    paragraphStyle.paragraphSpacing = 2
    return [
      .font: baseFont,
      .foregroundColor: textColor,
      .paragraphStyle: paragraphStyle.copy() as! NSParagraphStyle,
    ]
  }

  // MARK: - Inline Code

  var codeFontSize: CGFloat { baseFontSize - 1.5 }
  var codeFont: NSFont { .monospacedSystemFont(ofSize: codeFontSize, weight: .regular) }
  var codeBackgroundColor: NSColor { .quaternaryLabelColor.withAlphaComponent(0.5) }

  var inlineCodeAttributes: [NSAttributedString.Key: Any] {
    [
      .font: codeFont,
      .backgroundColor: codeBackgroundColor,
    ]
  }

  // MARK: - Code Blocks

  var codeBlockAttributes: [NSAttributedString.Key: Any] {
    let paragraphStyle = NSMutableParagraphStyle()
    paragraphStyle.headIndent = 12
    paragraphStyle.firstLineHeadIndent = 12
    // Negative tailIndent extends the paragraph to the right margin.
    paragraphStyle.tailIndent = -12
    paragraphStyle.paragraphSpacing = 2
    paragraphStyle.paragraphSpacingBefore = 2
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

  var baseAttributes: [NSAttributedString.Key: Any] {
    let paragraphStyle = NSMutableParagraphStyle()
    paragraphStyle.paragraphSpacing = 4
    return [
      .font: baseFont,
      .foregroundColor: textColor,
      .paragraphStyle: paragraphStyle.copy() as! NSParagraphStyle,
    ]
  }
}
