import AppKit
import Markdown

/// Walks a swift-markdown AST and produces `SyntaxNode` values with NSRange positions.
@MainActor
struct MarkdownParser: @preconcurrency MarkupWalker {
  private let converter: SourceRangeConverter
  private let style: MarkdownStyle
  private(set) var nodes: [SyntaxNode] = []

  /// Current list nesting depth (for indentation).
  private var listDepth = 0

  init(converter: SourceRangeConverter, style: MarkdownStyle = .default) {
    self.converter = converter
    self.style = style
  }

  // MARK: - Block Elements

  mutating func visitHeading(_ heading: Heading) -> () {
    guard let sourceRange = heading.range else { return descendInto(heading) }
    let range = converter.nsRange(from: sourceRange)

    let delimiterLength: Int
    if let firstChild = heading.children.first(where: { $0.range != nil }),
      let childRange = firstChild.range
    {
      let childStart = converter.utf16Offset(from: childRange.lowerBound)
      delimiterLength = childStart - range.location
    } else {
      delimiterLength = heading.level + 1
    }

    let delimiterRange = NSRange(location: range.location, length: delimiterLength)
    let contentRange = NSRange(
      location: range.location + delimiterLength,
      length: max(0, range.length - delimiterLength)
    )

    nodes.append(
      SyntaxNode(
        type: .heading(level: heading.level),
        range: range,
        contentRange: contentRange,
        delimiterRanges: [delimiterRange],
        attributes: style.headingAttributes(level: heading.level)
      ))
    descendInto(heading)
  }

  mutating func visitCodeBlock(_ codeBlock: CodeBlock) -> () {
    guard let sourceRange = codeBlock.range else { return }
    let range = converter.nsRange(from: sourceRange)

    let text = converter.substringForRange(range)
    let openFenceEnd = text?.firstIndex(of: "\n").map {
      text!.distance(from: text!.startIndex, to: text!.index(after: $0))
    } ?? 3

    let closingFenceStart: Int
    if let text, let lastNewline = text.lastIndex(of: "\n"),
      lastNewline > text.startIndex
    {
      closingFenceStart = text.distance(from: text.startIndex, to: lastNewline) + 1
    } else {
      closingFenceStart = max(0, range.length - 3)
    }

    let openDelimiter = NSRange(location: range.location, length: openFenceEnd)
    let closeDelimiter = NSRange(
      location: range.location + closingFenceStart,
      length: range.length - closingFenceStart
    )
    let contentRange = NSRange(
      location: range.location + openFenceEnd,
      length: max(0, closingFenceStart - openFenceEnd)
    )

    nodes.append(
      SyntaxNode(
        type: .codeBlock(language: codeBlock.language),
        range: range,
        contentRange: contentRange,
        delimiterRanges: [openDelimiter, closeDelimiter],
        attributes: style.codeBlockAttributes
      ))
  }

  mutating func visitBlockQuote(_ blockQuote: BlockQuote) -> () {
    guard let sourceRange = blockQuote.range else { return descendInto(blockQuote) }
    let range = converter.nsRange(from: sourceRange)

    nodes.append(
      SyntaxNode(
        type: .blockquote,
        range: range,
        contentRange: range,
        delimiterRanges: [],
        attributes: style.blockquoteAttributes
      ))
    descendInto(blockQuote)
  }

  mutating func visitUnorderedList(_ list: UnorderedList) -> () {
    listDepth += 1
    descendInto(list)
    listDepth -= 1
  }

  mutating func visitOrderedList(_ list: OrderedList) -> () {
    listDepth += 1
    descendInto(list)
    listDepth -= 1
  }

  mutating func visitListItem(_ listItem: ListItem) -> () {
    guard let sourceRange = listItem.range else { return descendInto(listItem) }
    let range = converter.nsRange(from: sourceRange)

    let isOrdered = listItem.parent is OrderedList
    let type: SyntaxNodeType = isOrdered ? .orderedListItem : .unorderedListItem

    nodes.append(
      SyntaxNode(
        type: type,
        range: range,
        contentRange: range,
        delimiterRanges: [],
        attributes: style.listItemAttributes(indentLevel: listDepth)
      ))
    descendInto(listItem)
  }

  mutating func visitThematicBreak(_ thematicBreak: ThematicBreak) -> () {
    guard let sourceRange = thematicBreak.range else { return }
    let range = converter.nsRange(from: sourceRange)

    nodes.append(
      SyntaxNode(
        type: .thematicBreak,
        range: range,
        contentRange: range,
        delimiterRanges: [],
        attributes: [
          .strikethroughStyle: NSUnderlineStyle.thick.rawValue,
          .strikethroughColor: style.thematicBreakColor,
          .foregroundColor: NSColor.clear,
        ]
      ))
  }

  // MARK: - Inline Elements

  mutating func visitStrong(_ strong: Strong) -> () {
    guard let sourceRange = strong.range else { return descendInto(strong) }
    let range = converter.nsRange(from: sourceRange)
    let (opening, closing, content) = inlineDelimiters(
      nodeRange: range, markup: strong)

    nodes.append(
      SyntaxNode(
        type: .strong,
        range: range,
        contentRange: content,
        delimiterRanges: [opening, closing].compactMap { $0 },
        attributes: style.strongAttributes
      ))
    descendInto(strong)
  }

  mutating func visitEmphasis(_ emphasis: Emphasis) -> () {
    guard let sourceRange = emphasis.range else { return descendInto(emphasis) }
    let range = converter.nsRange(from: sourceRange)
    let (opening, closing, content) = inlineDelimiters(
      nodeRange: range, markup: emphasis)

    nodes.append(
      SyntaxNode(
        type: .emphasis,
        range: range,
        contentRange: content,
        delimiterRanges: [opening, closing].compactMap { $0 },
        attributes: style.emphasisAttributes
      ))
    descendInto(emphasis)
  }

  mutating func visitInlineCode(_ inlineCode: InlineCode) -> () {
    guard let sourceRange = inlineCode.range else { return }
    let range = converter.nsRange(from: sourceRange)

    let openDelimiter = NSRange(location: range.location, length: 1)
    let closeDelimiter = NSRange(location: range.location + range.length - 1, length: 1)
    let contentRange = NSRange(
      location: range.location + 1,
      length: max(0, range.length - 2)
    )

    nodes.append(
      SyntaxNode(
        type: .inlineCode,
        range: range,
        contentRange: contentRange,
        delimiterRanges: [openDelimiter, closeDelimiter],
        attributes: style.inlineCodeAttributes
      ))
  }

  mutating func visitLink(_ link: Markdown.Link) -> () {
    guard let sourceRange = link.range else { return descendInto(link) }
    let range = converter.nsRange(from: sourceRange)

    let openBracket = NSRange(location: range.location, length: 1)

    let contentStart = range.location + 1
    let contentEnd: Int
    if let lastChild = link.children.reversed().first(where: { $0.range != nil }),
      let childRange = lastChild.range
    {
      contentEnd = converter.utf16Offset(from: childRange.upperBound)
    } else {
      contentEnd = contentStart
    }
    let contentRange = NSRange(location: contentStart, length: contentEnd - contentStart)
    let urlPart = NSRange(
      location: contentEnd, length: range.location + range.length - contentEnd)

    nodes.append(
      SyntaxNode(
        type: .link(destination: link.destination),
        range: range,
        contentRange: contentRange,
        delimiterRanges: [openBracket, urlPart],
        attributes: style.linkAttributes(destination: link.destination)
      ))
    descendInto(link)
  }

  mutating func visitStrikethrough(_ strikethrough: Strikethrough) -> () {
    guard let sourceRange = strikethrough.range else { return descendInto(strikethrough) }
    let range = converter.nsRange(from: sourceRange)
    let (opening, closing, content) = inlineDelimiters(
      nodeRange: range, markup: strikethrough)

    nodes.append(
      SyntaxNode(
        type: .strikethrough,
        range: range,
        contentRange: content,
        delimiterRanges: [opening, closing].compactMap { $0 },
        attributes: style.strikethroughAttributes
      ))
    descendInto(strikethrough)
  }

  // MARK: - Helpers

  private func inlineDelimiters(nodeRange: NSRange, markup: some Markup)
    -> (opening: NSRange?, closing: NSRange?, content: NSRange)
  {
    let children = markup.children.filter { $0.range != nil }
    guard let firstChild = children.first, let firstChildRange = firstChild.range,
      let lastChild = children.last, let lastChildRange = lastChild.range
    else {
      return (nil, nil, nodeRange)
    }

    let contentStart = converter.utf16Offset(from: firstChildRange.lowerBound)
    let contentEnd = converter.utf16Offset(from: lastChildRange.upperBound)

    let openingLength = contentStart - nodeRange.location
    let closingLength = (nodeRange.location + nodeRange.length) - contentEnd

    let opening =
      openingLength > 0
      ? NSRange(location: nodeRange.location, length: openingLength) : nil
    let closing =
      closingLength > 0
      ? NSRange(location: contentEnd, length: closingLength) : nil
    let content = NSRange(location: contentStart, length: contentEnd - contentStart)

    return (opening, closing, content)
  }
}

extension SourceRangeConverter {
  func substringForRange(_ range: NSRange) -> String? {
    let nsString = string as NSString
    guard range.location + range.length <= nsString.length else { return nil }
    return nsString.substring(with: range)
  }
}
