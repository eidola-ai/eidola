import AppKit
import Markdown
import STTextView

/// Orchestrates markdown parsing, attribute application, and cursor-aware syntax reveal.
@MainActor
final class SyntaxHighlighter {
  let style: MarkdownStyle
  private(set) var nodes: [SyntaxNode] = []

  init(style: MarkdownStyle = .default) {
    self.style = style
  }

  /// Re-parse the full document and apply all attributes.
  func highlight(textView: STTextView) {
    let text = textView.text ?? ""
    guard !text.isEmpty else {
      self.nodes = []
      return
    }

    let nsString = text as NSString
    let fullRange = NSRange(location: 0, length: nsString.length)

    // 1. Parse
    let document = Document(parsing: text)
    let converter = SourceRangeConverter(string: text)
    var parser = MarkdownParser(converter: converter, style: style)
    parser.visit(document)
    self.nodes = parser.nodes

    // 2. Reset all attributes (stored + rendering) to base
    textView.setAttributes(style.baseAttributes, range: fullRange)
    // Clear stale rendering attributes from previous highlight pass
    textView.removeRenderingAttribute(.foregroundColor, range: fullRange)
    textView.removeRenderingAttribute(.font, range: fullRange)

    // 3. Apply node attributes
    for node in nodes {
      let safeRange = clamp(node.range, to: nsString.length)
      let safeContentRange = clamp(node.contentRange, to: nsString.length)

      if !node.attributes.isEmpty {
        textView.addAttributes(node.attributes, range: safeContentRange)
      }

      // Delimiters get the content font (so heading `#` is the right size)
      // but with dimmed color
      for delim in node.delimiterRanges {
        let safeDelim = clamp(delim, to: nsString.length)
        guard safeDelim.length > 0 else { continue }
        var delimAttrs = node.attributes
        delimAttrs[.foregroundColor] = style.delimiterColor
        textView.addAttributes(delimAttrs, range: safeDelim)
      }

      // Code blocks: background on the full range including fences
      if case .codeBlock = node.type {
        textView.addAttributes(
          [.backgroundColor: style.codeBackgroundColor], range: safeRange)
      }
    }

    // 4. Apply delimiter visibility
    updateDelimiterVisibility(textView: textView)
  }

  /// Show/hide syntax delimiters based on current cursor position.
  func updateDelimiterVisibility(textView: STTextView) {
    let cursorRange = currentCursorRange(in: textView)
    let textLength = ((textView.text ?? "") as NSString).length

    for node in nodes {
      let cursorInNode = cursorRange.map { cursorOverlaps($0, node: node.range) } ?? false

      for delim in node.delimiterRanges {
        let safeDelim = clamp(delim, to: textLength)
        guard safeDelim.length > 0 else { continue }

        if cursorInNode {
          // Reveal: show delimiters with dimmed color
          textView.addRenderingAttributes(
            [.foregroundColor: style.delimiterColor], range: safeDelim)
        } else {
          // Hide: make delimiters invisible (still take up space)
          textView.addRenderingAttributes(
            [.foregroundColor: NSColor.clear], range: safeDelim)
        }
      }
    }
  }

  private func currentCursorRange(in textView: STTextView) -> NSRange? {
    let textLayoutManager = textView.textLayoutManager
    guard let selection = textLayoutManager.textSelections.first,
      let selRange = selection.textRanges.first,
      let contentManager = textLayoutManager.textContentManager
    else { return nil }

    let docStart = contentManager.documentRange.location
    let start = textLayoutManager.offset(from: docStart, to: selRange.location)
    let end = textLayoutManager.offset(from: docStart, to: selRange.endLocation)
    guard start >= 0, end >= start else { return nil }
    return NSRange(location: start, length: end - start)
  }

  private func cursorOverlaps(_ cursor: NSRange, node: NSRange) -> Bool {
    let cursorEnd = cursor.location + cursor.length
    let nodeEnd = node.location + node.length
    return cursor.location < nodeEnd && cursorEnd > node.location
      || cursor.location >= node.location && cursor.location <= nodeEnd
  }

  private func clamp(_ range: NSRange, to maxLength: Int) -> NSRange {
    let start = min(range.location, maxLength)
    let length = min(range.length, maxLength - start)
    return NSRange(location: start, length: max(0, length))
  }
}
