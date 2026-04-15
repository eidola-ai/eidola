import AppKit
import Markdown

/// Orchestrates markdown parsing, attribute application, and cursor-aware syntax reveal.
@MainActor
final class SyntaxHighlighter {
  let style: MarkdownStyle
  private(set) var nodes: [SyntaxNode] = []

  init(style: MarkdownStyle = .default) {
    self.style = style
  }

  /// Re-parse the full document and apply all attributes.
  func highlight(textView: NSTextView) {
    let text = textView.string
    guard let textStorage = textView.textStorage else { return }
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

    // 2. Reset all attributes to base.
    // Batch text storage edits to avoid multiple
    // layout passes during attribute application.
    textStorage.beginEditing()
    textStorage.setAttributes(style.baseAttributes, range: fullRange)

    // 3. Apply node attributes
    for node in nodes {
      let safeRange = clamp(node.range, to: nsString.length)
      let safeContentRange = clamp(node.contentRange, to: nsString.length)

      if !node.attributes.isEmpty {
        textStorage.addAttributes(node.attributes, range: safeContentRange)
      }

      // Delimiters get the content font (so heading `#` is the right size)
      // but with dimmed color
      for delim in node.delimiterRanges {
        let safeDelim = clamp(delim, to: nsString.length)
        guard safeDelim.length > 0 else { continue }
        var delimAttrs = node.attributes
        delimAttrs[.foregroundColor] = style.delimiterColor
        textStorage.addAttributes(delimAttrs, range: safeDelim)
      }

      // Code blocks: background on the full range including fences
      if case .codeBlock = node.type {
        textStorage.addAttributes(
          [.backgroundColor: style.codeBackgroundColor], range: safeRange)
      }
    }
    textStorage.endEditing()

    // 4. Clear stale temporary attributes and apply delimiter visibility
    if let layoutManager = textView.layoutManager {
      layoutManager.removeTemporaryAttribute(
        .foregroundColor, forCharacterRange: fullRange)
    }
    updateDelimiterVisibility(textView: textView)
  }

  /// Show/hide syntax delimiters based on current cursor position.
  func updateDelimiterVisibility(textView: NSTextView) {
    guard let layoutManager = textView.layoutManager else { return }
    let cursorRange = textView.selectedRange()
    let textLength = (textView.string as NSString).length

    for node in nodes {
      let cursorInNode = cursorOverlaps(cursorRange, node: node.range)

      for delim in node.delimiterRanges {
        let safeDelim = clamp(delim, to: textLength)
        guard safeDelim.length > 0 else { continue }

        if cursorInNode {
          layoutManager.addTemporaryAttributes(
            [.foregroundColor: style.delimiterColor], forCharacterRange: safeDelim)
        } else {
          layoutManager.addTemporaryAttributes(
            [.foregroundColor: NSColor.clear], forCharacterRange: safeDelim)
        }
      }
    }
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
