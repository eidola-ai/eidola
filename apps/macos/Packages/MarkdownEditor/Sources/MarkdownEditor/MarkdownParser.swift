import AppKit
import Markdown

/// Walks a swift-markdown AST and produces `SyntaxNode` values with NSRange positions.
@MainActor
struct MarkdownParser: @preconcurrency MarkupWalker {
  private let converter: SourceRangeConverter
  private let style: MarkdownStyle
  private(set) var nodes: [SyntaxNode] = []
  /// Tracks total nesting depth across all list types during traversal.
  private var listDepth = 0
  /// The widest marker text and its rendered width among items in the
  /// current ordered list, so all items get consistent alignment.
  private var currentOrderedListWidestMarker: String?
  private var currentOrderedListWidestWidth: CGFloat = 0

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

    // Setext-style headings (content\n===) have delimiterLength == 0 because the
    // content starts at the heading start (no `# ` prefix). The underline is the
    // delimiter that spans from the \n to the end of the heading range.
    // EditorUpdate normalizes setext → ATX when the cursor moves away from the
    // underline, so setext headings are transient.
    let delimiterRanges: [NSRange]
    let contentRange: NSRange

    if delimiterLength == 0 {
      // Setext: content is first line, delimiter is \n + underline
      let headingText = converter.substringForRange(range) ?? ""
      if let newlineIdx = headingText.firstIndex(of: "\n") {
        let contentLen = headingText.distance(from: headingText.startIndex, to: newlineIdx)
        contentRange = NSRange(location: range.location, length: contentLen)
        // Delimiter covers from the \n through end of range (the underline)
        let delimStart = range.location + contentLen
        let delimLen = range.length - contentLen
        delimiterRanges = [NSRange(location: delimStart, length: delimLen)]
      } else {
        // Fallback: no newline found, treat entire range as content
        contentRange = range
        delimiterRanges = []
      }
    } else {
      // ATX: delimiter is the `# ` prefix
      delimiterRanges = [NSRange(location: range.location, length: delimiterLength)]
      contentRange = NSRange(
        location: range.location + delimiterLength,
        length: max(0, range.length - delimiterLength))
    }

    nodes.append(
      SyntaxNode(
        type: .heading(level: heading.level),
        range: range,
        contentRange: contentRange,
        delimiterRanges: delimiterRanges,
        attributes: style.headingAttributes(level: heading.level)
      ))
    descendInto(heading)
  }

  mutating func visitBlockQuote(_ blockQuote: BlockQuote) -> () {
    guard let sourceRange = blockQuote.range else { return descendInto(blockQuote) }
    let range = converter.nsRange(from: sourceRange)
    let nsText = converter.string as NSString

    // Find all `> ` prefixes on each line within the blockquote range.
    // These are the delimiter ranges that get hidden/revealed.
    var delimiterRanges: [NSRange] = []
    let rangeEnd = range.location + range.length

    var pos = range.location
    while pos < rangeEnd {
      // Check if this line starts with `> `
      if pos < nsText.length, nsText.character(at: pos) == UInt16(0x003E) {  // '>'
        // Check for space after '>'
        let nextPos = pos + 1
        if nextPos < nsText.length, nsText.character(at: nextPos) == UInt16(0x0020) {  // ' '
          delimiterRanges.append(NSRange(location: pos, length: 2))  // "> "
        } else {
          // Just ">" without space
          delimiterRanges.append(NSRange(location: pos, length: 1))
        }
      }

      // Advance to next line
      while pos < rangeEnd {
        if pos < nsText.length, nsText.character(at: pos) == UInt16(0x000A) {  // \n
          pos += 1
          break
        }
        pos += 1
      }
      // If we didn't hit a newline, we're past the end
      if pos >= rangeEnd { break }
    }

    // Content range is the full range (content is interspersed with `> ` prefixes)
    let contentRange = range

    nodes.append(
      SyntaxNode(
        type: .blockquote,
        range: range,
        contentRange: contentRange,
        delimiterRanges: delimiterRanges,
        attributes: style.blockquoteAttributes
      ))
    descendInto(blockQuote)
  }

  mutating func visitUnorderedList(_ unorderedList: UnorderedList) -> () {
    listDepth += 1
    descendInto(unorderedList)
    listDepth -= 1
  }

  mutating func visitOrderedList(_ orderedList: OrderedList) -> () {
    // Find the widest marker among all items so every item in this list
    // gets the same headIndent, preventing jagged content alignment.
    let font = style.baseFont
    var widestMarker = "1. "
    var widestWidth: CGFloat = 0

    for child in orderedList.children {
      guard let item = child as? ListItem, let itemRange = item.range else { continue }
      let itemStart = converter.utf16Offset(from: itemRange.lowerBound)
      let markerLen: Int
      if let firstChild = item.children.first(where: { $0.range != nil }),
        let childRange = firstChild.range
      {
        markerLen = converter.utf16Offset(from: childRange.lowerBound) - itemStart
      } else {
        markerLen = 3
      }
      let markerRange = NSRange(location: itemStart, length: markerLen)
      if let text = converter.substringForRange(markerRange) {
        let width = (text as NSString).size(withAttributes: [.font: font]).width
        if width > widestWidth {
          widestWidth = width
          widestMarker = text
        }
      }
    }

    let previousWidest = currentOrderedListWidestMarker
    let previousWidth = currentOrderedListWidestWidth
    currentOrderedListWidestMarker = widestMarker
    currentOrderedListWidestWidth = widestWidth
    listDepth += 1
    descendInto(orderedList)
    listDepth -= 1
    currentOrderedListWidestMarker = previousWidest
    currentOrderedListWidestWidth = previousWidth
  }

  mutating func visitListItem(_ listItem: ListItem) -> () {
    let isUnordered = listItem.parent is UnorderedList
    let isOrdered = listItem.parent is OrderedList

    guard isUnordered || isOrdered else {
      return descendInto(listItem)
    }
    guard let sourceRange = listItem.range else { return descendInto(listItem) }
    let range = converter.nsRange(from: sourceRange)

    // The delimiter is the marker character(s) plus the space after it.
    // For unordered: "- " or "* " or "+ " (marker char + space)
    // For ordered: "1. " or "12. " etc. (digits + ". ")
    // swift-markdown's ListItem range starts at the marker. The first child's
    // range starts at the content after the marker + space.
    let markerLength: Int
    if let firstChild = listItem.children.first(where: { $0.range != nil }),
      let childRange = firstChild.range
    {
      let childStart = converter.utf16Offset(from: childRange.lowerBound)
      markerLength = childStart - range.location
    } else {
      markerLength = isOrdered ? 3 : 2  // fallback: "1. " or "- "
    }

    // Compute leading whitespace before the marker. For nested items, the
    // source has indentation spaces that need to be hidden so the paragraph
    // style controls indentation instead.
    let nsText = converter.string as NSString
    let lineStart = nsText.lineRange(for: NSRange(location: range.location, length: 0)).location
    let leadingWhitespaceLength = range.location - lineStart

    let contentRange = NSRange(
      location: range.location + markerLength,
      length: max(0, range.length - markerLength)
    )

    if isUnordered {
      // Delimiter includes leading whitespace + marker ("    - ")
      let delimiterRange = NSRange(location: lineStart, length: leadingWhitespaceLength + markerLength)
      let indentLevel = listDepth

      // Check for checkbox list item: ListItem.checkbox is set by swift-markdown.
      // When a checkbox is present, swift-markdown's first child starts AFTER the
      // full "- [ ] " prefix, so markerLength already includes the checkbox text.
      // The delimiter range (leading whitespace + markerLength) covers everything.
      if let checkbox = listItem.checkbox {
        let isChecked = checkbox == .checked
        let markerText = isChecked ? "\u{2612} " : "\u{25A1} "

        nodes.append(
          SyntaxNode(
            type: .checkboxListItem(checked: isChecked, indentLevel: indentLevel),
            range: range,
            contentRange: contentRange,
            delimiterRanges: [delimiterRange],
            attributes: style.listItemAttributes(indentLevel: indentLevel, markerText: markerText)
          ))
      } else {
        nodes.append(
          SyntaxNode(
            type: .unorderedListItem(indentLevel: indentLevel),
            range: range,
            contentRange: contentRange,
            delimiterRanges: [delimiterRange],
            attributes: style.listItemAttributes(indentLevel: indentLevel, markerText: "• ")
          ))
      }
    } else {
      // Ordered list: leading whitespace is a delimiter (hidden when cursor
      // outside), but the number marker stays visible.
      let indentLevel = listDepth

      // Use the widest marker in this list so all items align consistently.
      let widestMarkerText = currentOrderedListWidestMarker ?? "1. "

      // Compute padding: difference between widest marker width and this item's marker width.
      let font = style.baseFont
      let thisMarkerRange = NSRange(location: range.location, length: markerLength)
      let thisMarkerText = converter.substringForRange(thisMarkerRange) ?? "1. "
      let thisWidth = (thisMarkerText as NSString).size(withAttributes: [.font: font]).width
      let padding = max(0, currentOrderedListWidestWidth - thisWidth)

      var delimiterRanges: [NSRange] = []
      if leadingWhitespaceLength > 0 {
        delimiterRanges.append(NSRange(location: lineStart, length: leadingWhitespaceLength))
      }

      nodes.append(
        SyntaxNode(
          type: .orderedListItem(indentLevel: indentLevel, markerPadding: padding),
          range: range,
          contentRange: contentRange,
          delimiterRanges: delimiterRanges,
          attributes: style.listItemAttributes(indentLevel: indentLevel, markerText: widestMarkerText)
        ))
    }
    descendInto(listItem)
  }

  mutating func visitThematicBreak(_ thematicBreak: ThematicBreak) -> () {
    guard let sourceRange = thematicBreak.range else { return }
    let range = converter.nsRange(from: sourceRange)
    guard range.length > 0 else { return }

    // The entire thematic break (e.g. `---`, `***`, `___`) is both the delimiter
    // and the content — there is no separate content. We treat the full range as
    // the delimiter range so it can be hidden/revealed based on cursor position.
    nodes.append(
      SyntaxNode(
        type: .thematicBreak,
        range: range,
        contentRange: range,
        delimiterRanges: [range],
        attributes: style.thematicBreakAttributes
      ))
  }

  mutating func visitCodeBlock(_ codeBlock: CodeBlock) -> () {
    guard let sourceRange = codeBlock.range else { return }
    let range = converter.nsRange(from: sourceRange)
    guard range.length > 0 else { return }

    let fullText = converter.substringForRange(range) ?? ""
    let language = codeBlock.language

    // Find the opening fence text (everything before the first \n).
    // The \n itself is NOT part of the delimiter — it must remain visible
    // so TextKit preserves the paragraph boundary. Hiding the \n would
    // collapse the fence paragraph into the content paragraph, causing
    // glyph-layout issues (first content character clipped).
    var openingFenceTextLength = 0
    if let firstNewline = fullText.firstIndex(of: "\n") {
      openingFenceTextLength = fullText.distance(from: fullText.startIndex, to: firstNewline)
    } else {
      openingFenceTextLength = fullText.count
    }
    // Opening fence line occupies fenceText + \n in the source
    let openingLineLength = openingFenceTextLength + (fullText.contains("\n") ? 1 : 0)

    // Find the closing fence line.
    // The closing fence is the last line of the code block. We search
    // backwards for the last \n, then check if everything after it is a fence.
    var closingFenceLength = 0
    let trimmed = fullText.hasSuffix("\n") ? String(fullText.dropLast()) : fullText
    if let lastNewline = trimmed.lastIndex(of: "\n") {
      let afterLastNewline = trimmed[trimmed.index(after: lastNewline)...]
      let lastLine = String(afterLastNewline).trimmingCharacters(in: .whitespaces)
      if lastLine.hasPrefix("```") || lastLine.hasPrefix("~~~") {
        closingFenceLength = fullText.count - fullText.distance(from: fullText.startIndex, to: lastNewline) - 1
      }
    }

    // Opening fence delimiter: just the fence text, excluding the \n
    let openingFenceRange = NSRange(location: range.location, length: openingFenceTextLength)
    let closingFenceRange = NSRange(
      location: range.location + range.length - closingFenceLength,
      length: closingFenceLength)

    // Content starts after the opening fence line (including \n)
    let contentStart = range.location + openingLineLength
    let contentLength = max(0, range.length - openingLineLength - closingFenceLength)
    let contentRange = NSRange(location: contentStart, length: contentLength)

    var delimiterRanges: [NSRange] = [openingFenceRange]
    if closingFenceLength > 0 {
      delimiterRanges.append(closingFenceRange)
    }

    nodes.append(
      SyntaxNode(
        type: .codeBlock(language: language),
        range: range,
        contentRange: contentRange,
        delimiterRanges: delimiterRanges,
        attributes: style.codeBlockAttributes
      ))
  }

  // MARK: - Inline Elements

  mutating func visitStrong(_ strong: Strong) -> () {
    guard let sourceRange = strong.range else { return descendInto(strong) }
    let range = converter.nsRange(from: sourceRange)

    // Strong uses ** (2 chars) as delimiters. In nested `***bold***`,
    // swift-markdown gives both Emphasis and Strong the same range as the
    // outer Emphasis. When that happens, Strong's ** delimiters are the
    // inner 2 asterisks (offset inward by the Emphasis delimiter width of 1).
    let delimiterWidth = 2
    let nestedInEmphasis = strong.parent is Emphasis
      && strong.parent?.range == strong.range
    let inset = nestedInEmphasis ? 1 : 0
    let contentRange = NSRange(
      location: range.location + delimiterWidth + inset,
      length: max(0, range.length - (delimiterWidth + inset) * 2))
    let delimiterRanges = [
      NSRange(location: range.location + inset, length: delimiterWidth),
      NSRange(
        location: range.location + range.length - delimiterWidth - inset, length: delimiterWidth),
    ]

    nodes.append(
      SyntaxNode(
        type: .strong,
        range: range,
        contentRange: contentRange,
        delimiterRanges: delimiterRanges,
        attributes: [:]
      ))
    descendInto(strong)
  }

  mutating func visitInlineCode(_ inlineCode: InlineCode) -> () {
    guard let sourceRange = inlineCode.range else { return }
    let range = converter.nsRange(from: sourceRange)

    // Inline code uses backtick delimiters (1 char each side for single backtick).
    let delimiterWidth = 1
    let contentRange = NSRange(
      location: range.location + delimiterWidth,
      length: max(0, range.length - delimiterWidth * 2))
    let delimiterRanges = [
      NSRange(location: range.location, length: delimiterWidth),
      NSRange(location: range.location + range.length - delimiterWidth, length: delimiterWidth),
    ]

    nodes.append(
      SyntaxNode(
        type: .inlineCode,
        range: range,
        contentRange: contentRange,
        delimiterRanges: delimiterRanges,
        attributes: style.inlineCodeAttributes
      ))
  }

  mutating func visitLink(_ link: Markdown.Link) -> () {
    guard let sourceRange = link.range else { return descendInto(link) }
    let range = converter.nsRange(from: sourceRange)

    // Link syntax: [text](url)
    // Opening delimiter: `[` (1 char)
    // Content: the link text between `[` and `]`
    // Closing delimiter: `](url)` - everything from `]` to the end of the range
    let openingDelimiterRange = NSRange(location: range.location, length: 1)

    // Find the `]` position by looking at the first child's range or calculating
    // from the link text length. The content is between `[` and `]`.
    let contentStart = range.location + 1

    // To find where the content ends (where `]` is), we look at the raw text.
    // The link text content ends where `](` begins.
    let fullText = converter.substringForRange(range) ?? ""
    let closingBracketOffset: Int
    // Search for `](` from the end, backwards, to handle nested brackets
    if let bracketParenRange = fullText.range(of: "](", options: .backwards) {
      closingBracketOffset = fullText.distance(from: fullText.startIndex, to: bracketParenRange.lowerBound)
    } else {
      // Fallback: content is everything except first and last char
      closingBracketOffset = max(1, range.length - 1)
    }

    let contentLength = max(0, closingBracketOffset - 1)
    let contentRange = NSRange(location: contentStart, length: contentLength)

    // Closing delimiter: from `]` to end of range (covers `](url)`)
    let closingDelimiterStart = range.location + closingBracketOffset
    let closingDelimiterLength = range.length - closingBracketOffset
    let closingDelimiterRange = NSRange(
      location: closingDelimiterStart, length: closingDelimiterLength)

    let delimiterRanges = [openingDelimiterRange, closingDelimiterRange]

    let destination = link.destination

    nodes.append(
      SyntaxNode(
        type: .link(destination: destination),
        range: range,
        contentRange: contentRange,
        delimiterRanges: delimiterRanges,
        attributes: style.linkAttributes(destination: destination)
      ))
    descendInto(link)
  }

  mutating func visitImage(_ image: Markdown.Image) -> () {
    guard let sourceRange = image.range else { return descendInto(image) }
    let range = converter.nsRange(from: sourceRange)

    // Image syntax: ![alt text](url)
    // Opening delimiter: `![` (2 chars)
    // Content: the alt text between `![` and `]`
    // Closing delimiter: `](url)` - everything from `]` to the end of the range
    let openingDelimiterRange = NSRange(location: range.location, length: 2)

    let contentStart = range.location + 2

    // Find where `](` begins to determine where alt text ends.
    let fullText = converter.substringForRange(range) ?? ""
    let closingBracketOffset: Int
    if let bracketParenRange = fullText.range(of: "](", options: .backwards) {
      closingBracketOffset = fullText.distance(from: fullText.startIndex, to: bracketParenRange.lowerBound)
    } else {
      closingBracketOffset = max(2, range.length - 1)
    }

    let contentLength = max(0, closingBracketOffset - 2)
    let contentRange = NSRange(location: contentStart, length: contentLength)

    // Closing delimiter: from `]` to end of range (covers `](url)`)
    let closingDelimiterStart = range.location + closingBracketOffset
    let closingDelimiterLength = range.length - closingBracketOffset
    let closingDelimiterRange = NSRange(
      location: closingDelimiterStart, length: closingDelimiterLength)

    let delimiterRanges = [openingDelimiterRange, closingDelimiterRange]

    let destination = image.source

    nodes.append(
      SyntaxNode(
        type: .image(destination: destination),
        range: range,
        contentRange: contentRange,
        delimiterRanges: delimiterRanges,
        attributes: style.imageAttributes
      ))
    descendInto(image)
  }

  mutating func visitStrikethrough(_ strikethrough: Strikethrough) -> () {
    guard let sourceRange = strikethrough.range else { return descendInto(strikethrough) }
    let range = converter.nsRange(from: sourceRange)

    // Strikethrough uses ~~ (2 chars) or ~ (1 char) as delimiters.
    // Detect actual width from the first child's position.
    let delimiterWidth: Int
    if let firstChild = strikethrough.children.first(where: { $0.range != nil }),
      let childRange = firstChild.range
    {
      delimiterWidth = converter.utf16Offset(from: childRange.lowerBound) - range.location
    } else {
      delimiterWidth = 2  // fallback
    }
    let contentRange = NSRange(
      location: range.location + delimiterWidth,
      length: max(0, range.length - delimiterWidth * 2))
    let delimiterRanges = [
      NSRange(location: range.location, length: delimiterWidth),
      NSRange(location: range.location + range.length - delimiterWidth, length: delimiterWidth),
    ]

    nodes.append(
      SyntaxNode(
        type: .strikethrough,
        range: range,
        contentRange: contentRange,
        delimiterRanges: delimiterRanges,
        attributes: style.strikethroughAttributes
      ))
    descendInto(strikethrough)
  }

  mutating func visitEmphasis(_ emphasis: Emphasis) -> () {
    guard let sourceRange = emphasis.range else { return descendInto(emphasis) }
    let range = converter.nsRange(from: sourceRange)

    // Emphasis uses * (1 char) as delimiters.
    let delimiterWidth = 1
    let contentRange = NSRange(
      location: range.location + delimiterWidth,
      length: max(0, range.length - delimiterWidth * 2))
    let delimiterRanges = [
      NSRange(location: range.location, length: delimiterWidth),
      NSRange(location: range.location + range.length - delimiterWidth, length: delimiterWidth),
    ]

    nodes.append(
      SyntaxNode(
        type: .emphasis,
        range: range,
        contentRange: contentRange,
        delimiterRanges: delimiterRanges,
        attributes: [:]
      ))
    descendInto(emphasis)
  }
}

extension SourceRangeConverter {
  func substringForRange(_ range: NSRange) -> String? {
    let nsString = string as NSString
    guard range.location + range.length <= nsString.length else { return nil }
    return nsString.substring(with: range)
  }
}
