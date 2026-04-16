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
