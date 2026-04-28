import AppKit

/// A parsed markdown document with explicit block nesting plus flat inline spans.
struct MarkdownDocument {
  let blocks: [MarkdownBlock]
  let inlineNodes: [InlineSyntaxNode]
}

/// A block-level markdown construct.
struct MarkdownBlock {
  let kind: MarkdownBlockKind
  let range: NSRange
  let children: [MarkdownBlock]
}

enum MarkdownBlockKind {
  case paragraph
  case heading(level: Int, contentRange: NSRange, delimiterRanges: [NSRange])
  case blockquote(prefixRanges: [NSRange])
  case unorderedList
  case orderedList(widestMarkerText: String)
  case listItem(ListItemSyntax)
  case codeBlock(
    language: String?,
    contentRange: NSRange,
    openingFenceRange: NSRange,
    closingFenceRange: NSRange?
  )
  case thematicBreak
}

struct ListItemSyntax {
  let kind: ListItemKind
  let markerRange: NSRange
  let leadingWhitespaceRange: NSRange?
  let markerText: String
}

enum ListItemKind {
  case unordered
  case checkbox(checked: Bool)
  case ordered(widestMarkerText: String)
}

/// An inline markdown construct with delimiter ownership and a content range.
struct InlineSyntaxNode {
  let kind: InlineSyntaxKind
  let range: NSRange
  let contentRange: NSRange
  let delimiterRanges: [NSRange]
  let attributes: [NSAttributedString.Key: Any]
}

enum InlineSyntaxKind {
  case strong
  case emphasis
  case inlineCode
  case link(destination: String?)
  case image(destination: String?)
  case strikethrough
}
