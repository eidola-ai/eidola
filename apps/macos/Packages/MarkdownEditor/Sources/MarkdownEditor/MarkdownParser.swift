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
}

extension SourceRangeConverter {
  func substringForRange(_ range: NSRange) -> String? {
    let nsString = string as NSString
    guard range.location + range.length <= nsString.length else { return nil }
    return nsString.substring(with: range)
  }
}
