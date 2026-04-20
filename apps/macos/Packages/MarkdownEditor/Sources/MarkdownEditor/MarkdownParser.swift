import AppKit
import Markdown

/// Builds an explicit block tree plus flat inline syntax spans from a markdown document.
@MainActor
struct MarkdownParser {
  private struct ParseContext {
    var blockquoteDepth: Int
  }

  private let converter: SourceRangeConverter
  private let style: MarkdownStyle
  private let nsText: NSString

  private(set) var document: MarkdownDocument?
  private(set) var inlineNodes: [InlineSyntaxNode] = []

  init(converter: SourceRangeConverter, style: MarkdownStyle = .default) {
    self.converter = converter
    self.style = style
    self.nsText = converter.string as NSString
  }

  mutating func visit(_ document: Document) {
    inlineNodes = []
    let blocks = parseBlocks(in: document, context: ParseContext(blockquoteDepth: 0))
    self.document = MarkdownDocument(blocks: blocks, inlineNodes: inlineNodes)
  }

  // MARK: - Block Parsing

  private mutating func parseBlocks(in markup: Markup, context: ParseContext) -> [MarkdownBlock] {
    var blocks: [MarkdownBlock] = []

    for child in markup.children {
      switch child {
      case let heading as Heading:
        if let block = parseHeading(heading) { blocks.append(block) }
      case let paragraph as Paragraph:
        if let block = parseParagraph(paragraph) { blocks.append(block) }
      case let blockQuote as BlockQuote:
        if let block = parseBlockQuote(blockQuote, context: context) { blocks.append(block) }
      case let unorderedList as UnorderedList:
        if let block = parseUnorderedList(unorderedList, context: context) { blocks.append(block) }
      case let orderedList as OrderedList:
        if let block = parseOrderedList(orderedList, context: context) { blocks.append(block) }
      case let codeBlock as CodeBlock:
        if let block = parseCodeBlock(codeBlock) { blocks.append(block) }
      case let thematicBreak as ThematicBreak:
        if let block = parseThematicBreak(thematicBreak) { blocks.append(block) }
      default:
        break
      }
    }

    return blocks
  }

  private mutating func parseHeading(_ heading: Heading) -> MarkdownBlock? {
    guard let sourceRange = heading.range else { return nil }
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

    let delimiterRanges: [NSRange]
    let contentRange: NSRange

    if delimiterLength == 0 {
      let headingText = converter.substringForRange(range) ?? ""
      if let newlineIdx = headingText.firstIndex(of: "\n") {
        let contentLen = headingText.distance(from: headingText.startIndex, to: newlineIdx)
        contentRange = NSRange(location: range.location, length: contentLen)
        let delimStart = range.location + contentLen
        delimiterRanges = [NSRange(location: delimStart, length: range.length - contentLen)]
      } else {
        contentRange = range
        delimiterRanges = []
      }
    } else {
      delimiterRanges = [NSRange(location: range.location, length: delimiterLength)]
      contentRange = NSRange(
        location: range.location + delimiterLength,
        length: max(0, range.length - delimiterLength))
    }

    collectInlineNodes(in: heading)
    return MarkdownBlock(
      kind: .heading(level: heading.level, contentRange: contentRange, delimiterRanges: delimiterRanges),
      range: range,
      children: []
    )
  }

  private mutating func parseParagraph(_ paragraph: Paragraph) -> MarkdownBlock? {
    guard let sourceRange = paragraph.range else { return nil }
    let range = converter.nsRange(from: sourceRange)
    collectInlineNodes(in: paragraph)
    return MarkdownBlock(kind: .paragraph, range: range, children: [])
  }

  private mutating func parseBlockQuote(
    _ blockQuote: BlockQuote,
    context: ParseContext
  ) -> MarkdownBlock? {
    guard let range = blockRange(for: blockQuote) else { return nil }
    let depth = context.blockquoteDepth + 1
    let prefixRanges = blockquotePrefixRanges(in: range, depth: depth)
    let children = parseBlocks(
      in: blockQuote,
      context: ParseContext(blockquoteDepth: depth))

    return MarkdownBlock(
      kind: .blockquote(prefixRanges: prefixRanges),
      range: range,
      children: children
    )
  }

  private mutating func parseUnorderedList(
    _ unorderedList: UnorderedList,
    context: ParseContext
  ) -> MarkdownBlock? {
    guard let range = blockRange(for: unorderedList) else { return nil }
    let children = unorderedList.children.compactMap { child -> MarkdownBlock? in
      guard let item = child as? ListItem else { return nil }
      return parseListItem(item, orderedWidestMarkerText: nil, context: context)
    }

    return MarkdownBlock(kind: .unorderedList, range: range, children: children)
  }

  private mutating func parseOrderedList(
    _ orderedList: OrderedList,
    context: ParseContext
  ) -> MarkdownBlock? {
    guard let range = blockRange(for: orderedList) else { return nil }
    let widestMarkerText = widestOrderedMarkerText(in: orderedList) ?? "1. "
    let children = orderedList.children.compactMap { child -> MarkdownBlock? in
      guard let item = child as? ListItem else { return nil }
      return parseListItem(item, orderedWidestMarkerText: widestMarkerText, context: context)
    }

    return MarkdownBlock(
      kind: .orderedList(widestMarkerText: widestMarkerText),
      range: range,
      children: children
    )
  }

  private mutating func parseListItem(
    _ listItem: ListItem,
    orderedWidestMarkerText: String?,
    context: ParseContext
  ) -> MarkdownBlock? {
    guard let sourceRange = listItem.range else { return nil }
    let range = converter.nsRange(from: sourceRange)

    var markerLength: Int
    if let firstChild = listItem.children.first(where: { $0.range != nil }),
      let childRange = firstChild.range
    {
      let childStart = converter.utf16Offset(from: childRange.lowerBound)
      markerLength = childStart - range.location
    } else {
      markerLength = orderedWidestMarkerText == nil ? 2 : 3
    }

    let detectedCheckbox = checkboxMarkerInfo(at: range.location)
    if let checkbox = detectedCheckbox {
      markerLength = max(markerLength, checkbox.length)
    }

    let markerRange = NSRange(location: range.location, length: markerLength)
    let lineStart = nsText.lineRange(for: NSRange(location: range.location, length: 0)).location
    let effectiveLineStart = positionAfterBlockquotePrefixes(
      from: lineStart,
      depth: context.blockquoteDepth)
    let leadingWhitespaceLength = max(0, range.location - effectiveLineStart)
    let leadingWhitespaceRange =
      leadingWhitespaceLength > 0
      ? NSRange(location: effectiveLineStart, length: leadingWhitespaceLength)
      : nil

    let markerText = converter.substringForRange(markerRange)
      ?? (orderedWidestMarkerText == nil ? "- " : "1. ")

    let kind: ListItemKind
    if let checkbox = listItem.checkbox {
      kind = .checkbox(checked: checkbox == .checked)
    } else if let checkbox = detectedCheckbox {
      kind = .checkbox(checked: checkbox.checked)
    } else if let widestMarkerText = orderedWidestMarkerText {
      kind = .ordered(widestMarkerText: widestMarkerText)
    } else {
      kind = .unordered
    }

    let children = parseBlocks(in: listItem, context: context)
    return MarkdownBlock(
      kind: .listItem(
        ListItemSyntax(
          kind: kind,
          markerRange: markerRange,
          leadingWhitespaceRange: leadingWhitespaceRange,
          markerText: markerText
        )),
      range: range,
      children: children
    )
  }

  private mutating func parseCodeBlock(_ codeBlock: CodeBlock) -> MarkdownBlock? {
    guard let sourceRange = codeBlock.range else { return nil }
    var range = expandedCodeBlockRange(from: converter.nsRange(from: sourceRange))
    while range.length > 0, nsText.character(at: range.location) == UInt16(0x000A) {
      range.location += 1
      range.length -= 1
    }
    guard range.length > 0 else { return nil }

    let fullText = converter.substringForRange(range) ?? ""
    let openingFenceTextLength: Int
    if let firstNewline = fullText.firstIndex(of: "\n") {
      openingFenceTextLength = fullText.distance(from: fullText.startIndex, to: firstNewline)
    } else {
      openingFenceTextLength = fullText.count
    }
    let openingLineLength = openingFenceTextLength + (fullText.contains("\n") ? 1 : 0)

    var closingLineTotalLength = 0
    var closingFenceOnlyLength = 0
    let trimmed = fullText.hasSuffix("\n") ? String(fullText.dropLast()) : fullText
    if let lastNewline = trimmed.lastIndex(of: "\n") {
      let afterLastNewline = trimmed[trimmed.index(after: lastNewline)...]
      let lastLine = String(afterLastNewline)
      if isFenceLine(lastLine) {
        closingLineTotalLength =
          fullText.count - fullText.distance(from: fullText.startIndex, to: lastNewline) - 1
        // Strip blockquote prefixes and whitespace to find where the actual
        // fence characters start, so the fence range excludes `>` markers.
        let prefixLen = fencePrefixLength(in: lastLine)
        closingFenceOnlyLength = closingLineTotalLength - prefixLen
      }
    }

    let openingFenceRange = NSRange(location: range.location, length: openingFenceTextLength)
    let closingFenceRange: NSRange? =
      closingFenceOnlyLength > 0
      ? NSRange(
        location: range.location + range.length - closingFenceOnlyLength,
        length: closingFenceOnlyLength)
      : nil

    let contentStart = range.location + openingLineLength
    let contentLength = max(0, range.length - openingLineLength - closingLineTotalLength)
    let contentRange = NSRange(location: contentStart, length: contentLength)

    return MarkdownBlock(
      kind: .codeBlock(
        language: codeBlock.language,
        contentRange: contentRange,
        openingFenceRange: openingFenceRange,
        closingFenceRange: closingFenceRange),
      range: range,
      children: []
    )
  }

  private func parseThematicBreak(_ thematicBreak: ThematicBreak) -> MarkdownBlock? {
    guard let sourceRange = thematicBreak.range else { return nil }
    let range = converter.nsRange(from: sourceRange)
    return MarkdownBlock(kind: .thematicBreak, range: range, children: [])
  }

  private func checkboxMarkerInfo(at itemLocation: Int) -> (length: Int, checked: Bool)? {
    let lineRange = nsText.lineRange(for: NSRange(location: itemLocation, length: 0))
    let lineEnd = min(lineRange.location + lineRange.length, nsText.length)
    var length = max(0, lineEnd - itemLocation)

    while length > 0 {
      let ch = nsText.character(at: itemLocation + length - 1)
      if ch == UInt16(0x000A) || ch == UInt16(0x000D) {
        length -= 1
      } else {
        break
      }
    }

    guard length >= 6 else { return nil }
    let lineText = nsText.substring(with: NSRange(location: itemLocation, length: length))
    let chars = Array(lineText)
    guard chars.count >= 6 else { return nil }
    guard "-*+".contains(chars[0]), chars[1] == " ", chars[2] == "[", chars[4] == "]", chars[5] == " "
    else {
      return nil
    }

    switch chars[3] {
    case " ":
      return (length: 6, checked: false)
    case "x", "X":
      return (length: 6, checked: true)
    default:
      return nil
    }
  }

  // MARK: - Inline Parsing

  private mutating func collectInlineNodes(in markup: Markup) {
    for child in markup.children {
      switch child {
      case let strong as Strong:
        inlineNodes.append(makeStrongNode(strong))
        collectInlineNodes(in: strong)
      case let emphasis as Emphasis:
        inlineNodes.append(makeEmphasisNode(emphasis))
        collectInlineNodes(in: emphasis)
      case let inlineCode as InlineCode:
        inlineNodes.append(makeInlineCodeNode(inlineCode))
      case let link as Markdown.Link:
        inlineNodes.append(makeLinkNode(link))
        collectInlineNodes(in: link)
      case let image as Markdown.Image:
        inlineNodes.append(makeImageNode(image))
      case let strikethrough as Strikethrough:
        inlineNodes.append(makeStrikethroughNode(strikethrough))
        collectInlineNodes(in: strikethrough)
      default:
        collectInlineNodes(in: child)
      }
    }
  }

  private func makeStrongNode(_ strong: Strong) -> InlineSyntaxNode {
    let range = converter.nsRange(from: strong.range!)
    let delimiterWidth = 2
    let nestedInEmphasis = strong.parent is Emphasis && strong.parent?.range == strong.range
    let inset = nestedInEmphasis ? 1 : 0
    let contentRange = NSRange(
      location: range.location + delimiterWidth + inset,
      length: max(0, range.length - (delimiterWidth + inset) * 2))
    let delimiterRanges = [
      NSRange(location: range.location + inset, length: delimiterWidth),
      NSRange(location: range.location + range.length - delimiterWidth - inset, length: delimiterWidth),
    ]
    return InlineSyntaxNode(
      kind: .strong,
      range: range,
      contentRange: contentRange,
      delimiterRanges: delimiterRanges,
      attributes: [:]
    )
  }

  private func makeEmphasisNode(_ emphasis: Emphasis) -> InlineSyntaxNode {
    let range = converter.nsRange(from: emphasis.range!)
    let delimiterWidth = 1
    let contentRange = NSRange(
      location: range.location + delimiterWidth,
      length: max(0, range.length - delimiterWidth * 2))
    let delimiterRanges = [
      NSRange(location: range.location, length: delimiterWidth),
      NSRange(location: range.location + range.length - delimiterWidth, length: delimiterWidth),
    ]
    return InlineSyntaxNode(
      kind: .emphasis,
      range: range,
      contentRange: contentRange,
      delimiterRanges: delimiterRanges,
      attributes: [:]
    )
  }

  private func makeInlineCodeNode(_ inlineCode: InlineCode) -> InlineSyntaxNode {
    let range = converter.nsRange(from: inlineCode.range!)
    let delimiterWidth = 1
    let contentRange = NSRange(
      location: range.location + delimiterWidth,
      length: max(0, range.length - delimiterWidth * 2))
    let delimiterRanges = [
      NSRange(location: range.location, length: delimiterWidth),
      NSRange(location: range.location + range.length - delimiterWidth, length: delimiterWidth),
    ]
    return InlineSyntaxNode(
      kind: .inlineCode,
      range: range,
      contentRange: contentRange,
      delimiterRanges: delimiterRanges,
      attributes: style.inlineCodeAttributes
    )
  }

  private func makeLinkNode(_ link: Markdown.Link) -> InlineSyntaxNode {
    let range = converter.nsRange(from: link.range!)
    let openingDelimiterRange = NSRange(location: range.location, length: 1)
    let contentStart = range.location + 1

    let fullText = converter.substringForRange(range) ?? ""
    let closingBracketOffset: Int
    if let bracketParenRange = fullText.range(of: "](", options: .backwards) {
      closingBracketOffset = fullText.distance(from: fullText.startIndex, to: bracketParenRange.lowerBound)
    } else {
      closingBracketOffset = max(1, range.length - 1)
    }

    let contentLength = max(0, closingBracketOffset - 1)
    let contentRange = NSRange(location: contentStart, length: contentLength)
    let closingDelimiterRange = NSRange(
      location: range.location + closingBracketOffset,
      length: range.length - closingBracketOffset)

    return InlineSyntaxNode(
      kind: .link(destination: link.destination),
      range: range,
      contentRange: contentRange,
      delimiterRanges: [openingDelimiterRange, closingDelimiterRange],
      attributes: style.linkAttributes(destination: link.destination)
    )
  }

  private func makeImageNode(_ image: Markdown.Image) -> InlineSyntaxNode {
    let range = converter.nsRange(from: image.range!)
    let openingDelimiterRange = NSRange(location: range.location, length: 2)
    let contentStart = range.location + 2
    let fullText = converter.substringForRange(range) ?? ""

    let closingBracketOffset: Int
    if let bracketParenRange = fullText.range(of: "](", options: .backwards) {
      closingBracketOffset = fullText.distance(from: fullText.startIndex, to: bracketParenRange.lowerBound)
    } else {
      closingBracketOffset = max(2, range.length - 1)
    }

    let contentLength = max(0, closingBracketOffset - 2)
    let contentRange = NSRange(location: contentStart, length: contentLength)
    let closingDelimiterRange = NSRange(
      location: range.location + closingBracketOffset,
      length: range.length - closingBracketOffset)

    return InlineSyntaxNode(
      kind: .image(destination: image.source),
      range: range,
      contentRange: contentRange,
      delimiterRanges: [openingDelimiterRange, closingDelimiterRange],
      attributes: style.imageAttributes
    )
  }

  private func makeStrikethroughNode(_ strikethrough: Strikethrough) -> InlineSyntaxNode {
    let range = converter.nsRange(from: strikethrough.range!)
    let delimiterWidth: Int
    if let firstChild = strikethrough.children.first(where: { $0.range != nil }),
      let childRange = firstChild.range
    {
      delimiterWidth = converter.utf16Offset(from: childRange.lowerBound) - range.location
    } else {
      delimiterWidth = 2
    }

    let contentRange = NSRange(
      location: range.location + delimiterWidth,
      length: max(0, range.length - delimiterWidth * 2))
    let delimiterRanges = [
      NSRange(location: range.location, length: delimiterWidth),
      NSRange(location: range.location + range.length - delimiterWidth, length: delimiterWidth),
    ]

    return InlineSyntaxNode(
      kind: .strikethrough,
      range: range,
      contentRange: contentRange,
      delimiterRanges: delimiterRanges,
      attributes: style.strikethroughAttributes
    )
  }

  // MARK: - Range Helpers

  private func blockRange(for markup: Markup) -> NSRange? {
    if let sourceRange = markup.range {
      return converter.nsRange(from: sourceRange)
    }

    var start = Int.max
    var end = 0
    var found = false
    for child in markup.children {
      if let childRange = blockRange(for: child) {
        start = min(start, childRange.location)
        end = max(end, childRange.location + childRange.length)
        found = true
      }
    }
    guard found else { return nil }
    return NSRange(location: start, length: end - start)
  }

  private func widestOrderedMarkerText(in orderedList: OrderedList) -> String? {
    var widestMarker: String?
    var widestWidth: CGFloat = 0

    for child in orderedList.children {
      guard let item = child as? ListItem, let itemRange = item.range else { continue }
      let itemStart = converter.utf16Offset(from: itemRange.lowerBound)
      let markerLength: Int
      if let firstChild = item.children.first(where: { $0.range != nil }),
        let childRange = firstChild.range
      {
        markerLength = converter.utf16Offset(from: childRange.lowerBound) - itemStart
      } else {
        markerLength = 3
      }

      let markerRange = NSRange(location: itemStart, length: markerLength)
      guard let markerText = converter.substringForRange(markerRange) else { continue }
      let width = (markerText as NSString).size(withAttributes: [.font: style.baseFont]).width
      if width > widestWidth {
        widestWidth = width
        widestMarker = markerText
      }
    }

    return widestMarker
  }

  private func blockquotePrefixRanges(in range: NSRange, depth: Int) -> [NSRange] {
    let rangeEnd = range.location + range.length
    var pos = range.location
    var delimiterRanges: [NSRange] = []

    while pos < rangeEnd {
      let actualLineStart = nsText.lineRange(for: NSRange(location: pos, length: 0)).location
      var linePos = actualLineStart
      linePos = skipWhitespace(from: linePos)

      var skippedLevels = 0
      while skippedLevels < depth - 1, linePos < nsText.length {
        linePos = skipWhitespace(from: linePos)
        guard linePos < nsText.length, nsText.character(at: linePos) == UInt16(0x003E) else { break }
        linePos += 1
        if linePos < nsText.length, nsText.character(at: linePos) == UInt16(0x0020) {
          linePos += 1
        }
        skippedLevels += 1
      }

      linePos = skipWhitespace(from: linePos)
      if linePos < nsText.length, nsText.character(at: linePos) == UInt16(0x003E) {
        let length =
          linePos + 1 < nsText.length && nsText.character(at: linePos + 1) == UInt16(0x0020)
          ? 2 : 1
        delimiterRanges.append(NSRange(location: linePos, length: length))
      }

      while pos < rangeEnd {
        if pos < nsText.length, nsText.character(at: pos) == UInt16(0x000A) {
          pos += 1
          break
        }
        pos += 1
      }
    }

    return delimiterRanges
  }

  private func positionAfterBlockquotePrefixes(from lineStart: Int, depth: Int) -> Int {
    guard depth > 0 else { return lineStart }

    var pos = lineStart
    pos = skipWhitespace(from: pos)

    var skippedLevels = 0
    while skippedLevels < depth, pos < nsText.length {
      guard nsText.character(at: pos) == UInt16(0x003E) else { break }
      pos += 1
      if pos < nsText.length, nsText.character(at: pos) == UInt16(0x0020) {
        pos += 1
      }
      skippedLevels += 1
      pos = skipWhitespace(from: pos)
    }

    return pos
  }

  private func skipWhitespace(from start: Int) -> Int {
    var pos = start
    while pos < nsText.length {
      let ch = nsText.character(at: pos)
      if ch == UInt16(0x0020) || ch == UInt16(0x0009) {
        pos += 1
      } else {
        break
      }
    }
    return pos
  }

  /// Returns the number of leading characters that are blockquote prefixes
  /// and whitespace before the actual fence content (e.g. `> ` before `` ``` ``).
  private func fencePrefixLength(in line: String) -> Int {
    var remaining = line[...]
    while true {
      while let first = remaining.first, first == " " || first == "\t" {
        remaining.removeFirst()
      }
      if remaining.first == ">" {
        remaining.removeFirst()
        if remaining.first == " " {
          remaining.removeFirst()
        }
      } else {
        break
      }
    }
    return line.count - remaining.count
  }

  private func expandedCodeBlockRange(from initialRange: NSRange) -> NSRange {
    var expandedStart = initialRange.location
    var expandedEnd = initialRange.location + initialRange.length

    let firstLineRange = nsText.lineRange(for: NSRange(location: initialRange.location, length: 0))
    let firstLineText = nsText.substring(with: firstLineRange)
    if !isFenceLine(firstLineText), firstLineRange.location > 0 {
      let previousProbe = max(0, firstLineRange.location - 1)
      let previousLineRange = nsText.lineRange(for: NSRange(location: previousProbe, length: 0))
      if isFenceLine(nsText.substring(with: previousLineRange)) {
        expandedStart = previousLineRange.location
      }
    }

    let lastProbe = max(initialRange.location, min(nsText.length - 1, expandedEnd - 1))
    let lastLineRange = nsText.lineRange(for: NSRange(location: lastProbe, length: 0))
    let lastLineText = nsText.substring(with: lastLineRange)
    if !isFenceLine(lastLineText), expandedEnd < nsText.length {
      let nextLineRange = nsText.lineRange(for: NSRange(location: expandedEnd, length: 0))
      if isFenceLine(nsText.substring(with: nextLineRange)) {
        expandedEnd = nextLineRange.location + nextLineRange.length
      }
    }

    return NSRange(location: expandedStart, length: max(0, expandedEnd - expandedStart))
  }

  private func isFenceLine(_ line: String) -> Bool {
    var remaining = line[...]

    while true {
      while let first = remaining.first, first == " " || first == "\t" || first == "\n" {
        remaining.removeFirst()
      }
      if remaining.first == ">" {
        remaining.removeFirst()
        if remaining.first == " " {
          remaining.removeFirst()
        }
      } else {
        break
      }
    }

    let trimmed = remaining.trimmingCharacters(in: .whitespacesAndNewlines)
    return trimmed.hasPrefix("```") || trimmed.hasPrefix("~~~")
  }
}

extension SourceRangeConverter {
  func substringForRange(_ range: NSRange) -> String? {
    let nsString = string as NSString
    guard range.location + range.length <= nsString.length else { return nil }
    return nsString.substring(with: range)
  }
}
