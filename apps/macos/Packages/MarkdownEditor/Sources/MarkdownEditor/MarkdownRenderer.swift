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
      nodes: nodes, cursorRange: cursorRange, text: text, textLength: textLength, style: style)
  }

  /// Build a spec from pre-parsed nodes.
  static func buildSpec(
    nodes: [SyntaxNode],
    cursorRange: NSRange,
    text: String,
    textLength: Int,
    style: MarkdownStyle
  ) -> RenderSpec {
    let nsText = text as NSString
    var styledRanges: [RenderSpec.StyledRange] = []
    var hiddenIndexes = IndexSet()
    var temporaryAttributes: [RenderSpec.StyledRange] = []

    for node in nodes {
      let safeContentRange = clamp(node.contentRange, to: textLength)
      let safeNodeRange = clamp(node.range, to: textLength)

      // Extend the node range to cover the full line content (excluding trailing
      // newline) for both cursor overlap detection and styling. The parser may not
      // include trailing whitespace in the node range, but the cursor sitting after
      // trailing whitespace on the same line should still be considered "inside" the
      // heading, and the trailing whitespace should get heading styling.
      let lineRange = nsText.lineRange(for: safeNodeRange)
      // lineRange includes the trailing \n if present; strip it for cursor detection
      // so that a cursor at the start of the next line is NOT considered inside.
      let lineEnd = lineRange.location + lineRange.length
      let lineContentEnd: Int
      if lineEnd > lineRange.location
        && lineEnd <= textLength
        && nsText.character(at: lineEnd - 1) == UInt16(0x000A)  // \n
      {
        lineContentEnd = lineEnd - 1
      } else {
        lineContentEnd = lineEnd
      }
      let lineContentRange = NSRange(
        location: lineRange.location,
        length: lineContentEnd - lineRange.location)
      let cursorInNode = cursorOverlaps(
        cursorRange, node: lineContentRange, textLength: textLength)

      // Content attributes — apply to the full line range so delimiters and
      // any trailing whitespace inherit the heading's paragraph style (font size,
      // spacing). Using lineRange ensures consistent styling across the entire line.
      if !node.attributes.isEmpty {
        styledRanges.append(
          RenderSpec.StyledRange(range: lineRange, attributes: node.attributes))
      }

      // Delimiter visibility (hidden vs revealed)
      for delim in node.delimiterRanges {
        let safeDelim = clamp(delim, to: textLength)
        guard safeDelim.length > 0 else { continue }

        if cursorInNode {
          // Cursor inside: reveal delimiters with dimmed color via temporary attributes.
          // The delimiter already has the heading font/paragraph style from the node-wide
          // styled range above, so we only need to override the foreground color.
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
  ///
  /// The node range passed here is typically the line range (extended from the parser's
  /// node range to include trailing whitespace on the same line). A zero-width cursor
  /// at the very end of the document that equals nodeEnd is considered inside, because
  /// the user is still "on" that line. But a cursor at nodeEnd when there is more text
  /// after it means the cursor is on the next line, so it is NOT considered inside.
  static func cursorOverlaps(
    _ cursor: NSRange, node: NSRange, textLength: Int
  ) -> Bool {
    let cursorEnd = cursor.location + cursor.length
    let nodeEnd = node.location + node.length
    // Standard overlap check (cursor range intersects node range)
    if cursor.location < nodeEnd && cursorEnd > node.location {
      return true
    }
    // Zero-width cursor at end of document, exactly at node end.
    // This handles the case where the user is typing at the end of the last line
    // and the cursor sits one past the last character.
    if cursor.length == 0 && cursor.location == nodeEnd && nodeEnd == textLength {
      return true
    }
    return false
  }

  static func clamp(_ range: NSRange, to maxLength: Int) -> NSRange {
    let start = min(range.location, maxLength)
    let length = min(range.length, maxLength - start)
    return NSRange(location: start, length: max(0, length))
  }
}
