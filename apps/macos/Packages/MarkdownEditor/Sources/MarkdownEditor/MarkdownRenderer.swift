import AppKit
import Markdown

/// Pure computation: given editor state, produce a complete rendering specification.
///
/// This function is deterministic and idempotent — the same inputs always
/// produce the same output. It holds no state and has no side effects.
@MainActor
enum MarkdownRenderer {

  /// Produce a complete rendering spec from editor state.
  static func render(
    state: EditorState,
    style: MarkdownStyle = .default
  ) -> RenderSpec {
    return render(text: state.markdown, cursorRange: state.selection.nsRange, style: style)
  }

  /// Produce a complete rendering spec for the given markdown text and cursor position.
  static func render(
    text: String,
    cursorRange: NSRange,
    style: MarkdownStyle = .default
  ) -> RenderSpec {
    let textLength = (text as NSString).length

    guard textLength > 0 else {
      return RenderSpec(
        baseAttributes: style.baseAttributes,
        styledRanges: [],
        fontTraits: [],
        hiddenIndexes: IndexSet(),
        bulletIndexes: IndexSet(),
        temporaryAttributes: []
      )
    }

    // 1. Parse
    let document = Document(parsing: text)
    let converter = SourceRangeConverter(string: text)
    var parser = MarkdownParser(converter: converter, style: style)
    parser.visit(document)
    let nodes = parser.nodes

    // 2. Build the spec from nodes + cursor position
    return buildSpec(
      nodes: nodes, cursorRange: cursorRange, textLength: textLength, style: style)
  }

  /// Build a spec from pre-parsed nodes.
  static func buildSpec(
    nodes: [SyntaxNode],
    cursorRange: NSRange,
    textLength: Int,
    style: MarkdownStyle
  ) -> RenderSpec {
    var styledRanges: [RenderSpec.StyledRange] = []
    var hiddenIndexes = IndexSet()
    var temporaryAttributes: [RenderSpec.StyledRange] = []

    for node in nodes {
      let safeContentRange = clamp(node.contentRange, to: textLength)
      let cursorInNode = cursorOverlaps(cursorRange, node: node.range, textLength: textLength)

      // Content attributes
      if !node.attributes.isEmpty {
        styledRanges.append(
          RenderSpec.StyledRange(range: safeContentRange, attributes: node.attributes))
      }

      // Delimiter visibility (hidden vs revealed)
      for delim in node.delimiterRanges {
        let safeDelim = clamp(delim, to: textLength)
        guard safeDelim.length > 0 else { continue }

        if cursorInNode {
          // Cursor inside: reveal delimiters with dimmed color via temporary attributes
          temporaryAttributes.append(
            RenderSpec.StyledRange(
              range: safeDelim,
              attributes: [.foregroundColor: style.delimiterColor]))
        } else {
          // Cursor outside: hide delimiters
          let range = safeDelim.location..<(safeDelim.location + safeDelim.length)
          hiddenIndexes.insert(integersIn: range)
        }
      }
    }

    return RenderSpec(
      baseAttributes: style.baseAttributes,
      styledRanges: styledRanges,
      fontTraits: [],
      hiddenIndexes: hiddenIndexes,
      bulletIndexes: IndexSet(),
      temporaryAttributes: temporaryAttributes
    )
  }

  // MARK: - Private Helpers

  /// Check if cursor range overlaps with a node range (inclusive of boundaries).
  private static func cursorOverlaps(
    _ cursor: NSRange, node: NSRange, textLength: Int
  ) -> Bool {
    let safeNode = clamp(node, to: textLength)
    let cursorEnd = cursor.location + cursor.length
    let nodeEnd = safeNode.location + safeNode.length
    return cursor.location < nodeEnd && cursorEnd > safeNode.location
      || cursor.location >= safeNode.location && cursor.location <= nodeEnd
  }

  static func clamp(_ range: NSRange, to maxLength: Int) -> NSRange {
    let start = min(range.location, maxLength)
    let length = min(range.length, maxLength - start)
    return NSRange(location: start, length: max(0, length))
  }
}
