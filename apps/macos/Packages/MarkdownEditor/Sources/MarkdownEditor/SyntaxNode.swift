import AppKit

/// A parsed markdown construct with its ranges and visual attributes.
struct SyntaxNode {
  let type: SyntaxNodeType
  /// Full range including delimiters.
  let range: NSRange
  /// Content range excluding delimiters.
  let contentRange: NSRange
  /// Ranges of syntax delimiter characters (e.g. `**`, `` ` ``, `# `).
  let delimiterRanges: [NSRange]
  /// Visual attributes to apply to the content range.
  let attributes: [NSAttributedString.Key: Any]
}

enum SyntaxNodeType: Sendable {
  case heading(level: Int)
  case strong
  case emphasis
  case inlineCode
  case codeBlock(language: String?)
  case unorderedListItem
  case orderedListItem
  case blockquote
  case link(destination: String?)
  case strikethrough
  case thematicBreak
}
