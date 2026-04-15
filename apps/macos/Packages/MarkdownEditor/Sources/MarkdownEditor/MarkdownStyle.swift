import AppKit

/// Theme configuration for markdown rendering.
struct MarkdownStyle {
  static let `default` = MarkdownStyle()

  var baseFontSize: CGFloat = 15
  var codeFontSize: CGFloat = 13.5

  var baseFont: NSFont { .systemFont(ofSize: baseFontSize) }
  var boldFont: NSFont { .boldSystemFont(ofSize: baseFontSize) }
  var italicFont: NSFont { NSFontManager.shared.convert(baseFont, toHaveTrait: .italicFontMask) }
  var boldItalicFont: NSFont {
    NSFontManager.shared.convert(boldFont, toHaveTrait: .italicFontMask)
  }
  var codeFont: NSFont { .monospacedSystemFont(ofSize: codeFontSize, weight: .regular) }

  var textColor: NSColor { .labelColor }
  var delimiterColor: NSColor { .tertiaryLabelColor }
  var codeBackgroundColor: NSColor { NSColor.quaternaryLabelColor.withAlphaComponent(0.3) }
  var blockquoteBarColor: NSColor { .separatorColor }
  var linkColor: NSColor { .linkColor }
  var thematicBreakColor: NSColor { .separatorColor }

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

  var strongAttributes: [NSAttributedString.Key: Any] {
    [.font: boldFont]
  }

  var emphasisAttributes: [NSAttributedString.Key: Any] {
    [.font: italicFont]
  }

  var inlineCodeAttributes: [NSAttributedString.Key: Any] {
    [
      .font: codeFont,
      .backgroundColor: codeBackgroundColor,
    ]
  }

  var codeBlockAttributes: [NSAttributedString.Key: Any] {
    let paragraphStyle = NSMutableParagraphStyle()
    paragraphStyle.headIndent = 12
    paragraphStyle.firstLineHeadIndent = 12
    paragraphStyle.tailIndent = -12
    return [
      .font: codeFont,
      .backgroundColor: codeBackgroundColor,
      .paragraphStyle: paragraphStyle.copy() as! NSParagraphStyle,
    ]
  }

  func linkAttributes(destination: String?) -> [NSAttributedString.Key: Any] {
    var attrs: [NSAttributedString.Key: Any] = [
      .foregroundColor: linkColor,
      .underlineStyle: NSUnderlineStyle.single.rawValue,
    ]
    if let destination, let url = URL(string: destination) {
      attrs[.link] = url
    }
    return attrs
  }

  var strikethroughAttributes: [NSAttributedString.Key: Any] {
    [.strikethroughStyle: NSUnderlineStyle.single.rawValue]
  }

  var blockquoteAttributes: [NSAttributedString.Key: Any] {
    let paragraphStyle = NSMutableParagraphStyle()
    paragraphStyle.headIndent = 20
    paragraphStyle.firstLineHeadIndent = 20
    return [
      .foregroundColor: NSColor.secondaryLabelColor,
      .paragraphStyle: paragraphStyle.copy() as! NSParagraphStyle,
    ]
  }

  func listItemAttributes(indentLevel: Int) -> [NSAttributedString.Key: Any] {
    let paragraphStyle = NSMutableParagraphStyle()
    let indent = CGFloat(indentLevel + 1) * 20
    paragraphStyle.headIndent = indent
    paragraphStyle.firstLineHeadIndent = indent - 20
    let tabStop = NSTextTab(textAlignment: .left, location: indent)
    paragraphStyle.tabStops = [tabStop]
    return [
      .paragraphStyle: paragraphStyle.copy() as! NSParagraphStyle
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
