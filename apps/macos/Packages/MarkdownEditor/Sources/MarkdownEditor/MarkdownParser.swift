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
