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
    var fontTraits: [RenderSpec.TraitApplication] = []
    var hiddenIndexes = IndexSet()
    var temporaryAttributes: [RenderSpec.StyledRange] = []

    for node in nodes {
      let safeContentRange = clamp(node.contentRange, to: textLength)
      let safeNodeRange = clamp(node.range, to: textLength)

      switch node.type {
      case .heading:
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
        applyDelimiterVisibility(
          delimiterRanges: node.delimiterRanges,
          cursorInNode: cursorInNode,
          textLength: textLength,
          style: style,
          hiddenIndexes: &hiddenIndexes,
          temporaryAttributes: &temporaryAttributes)

      case .strong:
        let cursorInNode = cursorOverlaps(
          cursorRange, node: safeNodeRange, textLength: textLength)

        // Apply bold trait additively to the content range
        if safeContentRange.length > 0 {
          fontTraits.append(
            RenderSpec.TraitApplication(range: safeContentRange, trait: .boldFontMask))
        }

        // Delimiter visibility
        applyDelimiterVisibility(
          delimiterRanges: node.delimiterRanges,
          cursorInNode: cursorInNode,
          textLength: textLength,
          style: style,
          hiddenIndexes: &hiddenIndexes,
          temporaryAttributes: &temporaryAttributes)

      case .emphasis:
        let cursorInNode = cursorOverlaps(
          cursorRange, node: safeNodeRange, textLength: textLength)

        // Apply italic trait additively to the content range
        if safeContentRange.length > 0 {
          fontTraits.append(
            RenderSpec.TraitApplication(range: safeContentRange, trait: .italicFontMask))
        }

        // Delimiter visibility
        applyDelimiterVisibility(
          delimiterRanges: node.delimiterRanges,
          cursorInNode: cursorInNode,
          textLength: textLength,
          style: style,
          hiddenIndexes: &hiddenIndexes,
          temporaryAttributes: &temporaryAttributes)
      }
    }

    return RenderSpec(
      baseAttributes: style.baseAttributes,
      styledRanges: styledRanges,
      fontTraits: fontTraits,
      hiddenIndexes: hiddenIndexes,
      bulletIndexes: IndexSet(),
      temporaryAttributes: temporaryAttributes
    )
  }

  // MARK: - Private Helpers

  /// Apply delimiter hide/reveal logic shared by all constructs.
  private static func applyDelimiterVisibility(
    delimiterRanges: [NSRange],
    cursorInNode: Bool,
    textLength: Int,
    style: MarkdownStyle,
    hiddenIndexes: inout IndexSet,
    temporaryAttributes: inout [RenderSpec.StyledRange]
  ) {
    for delim in delimiterRanges {
      let safeDelim = clamp(delim, to: textLength)
      guard safeDelim.length > 0 else { continue }

      if cursorInNode {
        temporaryAttributes.append(
          RenderSpec.StyledRange(
            range: safeDelim,
            attributes: [.foregroundColor: style.delimiterColor]))
      } else {
        let range = safeDelim.location..<(safeDelim.location + safeDelim.length)
        hiddenIndexes.insert(integersIn: range)
      }
    }
  }

  /// Check if cursor range overlaps with a node range (inclusive of boundaries).
  ///
  /// A zero-width cursor at exactly `nodeStart` or `nodeEnd` is considered inside.
  /// At the start boundary the user is about to type into the construct; at the end
  /// boundary the user has just finished the construct and is still "touching" it.
  /// This is important for headings (where `lineContentRange` excludes the trailing
  /// `\n`) and inline constructs (where the cursor right after the closing delimiter
  /// should still reveal the delimiters).
  static func cursorOverlaps(
    _ cursor: NSRange, node: NSRange, textLength: Int
  ) -> Bool {
    let cursorEnd = cursor.location + cursor.length
    let nodeEnd = node.location + node.length
    // Standard overlap check (cursor range intersects node range)
    if cursor.location < nodeEnd && cursorEnd > node.location {
      return true
    }
    // Zero-width cursor at a node boundary
    if cursor.length == 0 {
      if cursor.location == node.location {
        return true
      }
      if cursor.location == nodeEnd {
        return true
      }
    }
    return false
  }

  static func clamp(_ range: NSRange, to maxLength: Int) -> NSRange {
    let start = min(range.location, maxLength)
    let length = min(range.length, maxLength - start)
    return NSRange(location: start, length: max(0, length))
  }
}
