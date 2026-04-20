import AppKit

/// A parsed markdown construct with its ranges and visual attributes.
struct SyntaxNode {
  let type: SyntaxNodeType
  /// Full range including delimiters.
  let range: NSRange
  /// Content range excluding delimiters.
  let contentRange: NSRange
  /// Ranges of syntax delimiter characters (e.g. `# `).
  let delimiterRanges: [NSRange]
  /// Visual attributes to apply to the content range.
  let attributes: [NSAttributedString.Key: Any]
}

enum SyntaxNodeType: Sendable {
  case heading(level: Int)
  case strong
  case emphasis
  case unorderedListItem(indentLevel: Int)
  case checkboxListItem(checked: Bool, indentLevel: Int)
  /// - `markerPadding`: Extra kern to add after the marker so content aligns
  ///   with the widest marker in this list. Zero if this IS the widest.
  case orderedListItem(indentLevel: Int, markerPadding: CGFloat)
  case inlineCode
  /// - `listBaseIndent`: Extra indentation from enclosing list items. Zero for top-level code blocks.
  case codeBlock(language: String?, listBaseIndent: CGFloat)
  case link(destination: String?)
  case image(destination: String?)
  case strikethrough
  /// - `listBaseIndent`: Extra indentation from enclosing list items (blockquote nested inside a list). Zero for top-level blockquotes.
  case blockquote(depth: Int, listBaseIndent: CGFloat)
  case thematicBreak
}
