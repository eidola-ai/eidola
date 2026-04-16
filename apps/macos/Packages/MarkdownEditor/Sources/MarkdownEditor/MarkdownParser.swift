import AppKit
import Markdown

/// Walks a swift-markdown AST and produces `SyntaxNode` values with NSRange positions.
@MainActor
struct MarkdownParser: @preconcurrency MarkupWalker {
  private let converter: SourceRangeConverter
  private let style: MarkdownStyle
  private(set) var nodes: [SyntaxNode] = []

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
