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
    var bulletIndexes = IndexSet()
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

      case .orderedListItem(_, let markerPadding):
        // Ordered list items: number marker is always visible, but leading
        // whitespace (for nested items) is hidden when cursor is outside.
        let lineRange = nsText.lineRange(for: safeNodeRange)
        let lineEnd = lineRange.location + lineRange.length
        let lineContentEnd: Int
        if lineEnd > lineRange.location
          && lineEnd <= textLength
          && nsText.character(at: lineEnd - 1) == UInt16(0x000A)
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

        if !node.attributes.isEmpty {
          styledRanges.append(
            RenderSpec.StyledRange(range: lineRange, attributes: node.attributes))
        }

        // Pad shorter markers so content aligns with the widest marker in
        // this list. Apply kern to the last character of the marker (the space
        // before content) to push content rightward.
        if markerPadding > 0.5, safeContentRange.location > 0 {
          let spaceIndex = safeContentRange.location - 1
          styledRanges.append(
            RenderSpec.StyledRange(
              range: NSRange(location: spaceIndex, length: 1),
              attributes: [.kern: markerPadding]))
        }

        // Leading whitespace is always hidden (paragraph style handles indentation).
        for delim in node.delimiterRanges {
          let safeDelim = clamp(delim, to: textLength)
          guard safeDelim.length > 0 else { continue }
          hiddenIndexes.insert(
            integersIn: safeDelim.location..<(safeDelim.location + safeDelim.length))
        }

        // Hide leading whitespace on continuation lines within this list item.
        hideContinuationWhitespace(
          in: nsText, nodeRange: safeNodeRange, textLength: textLength,
          hiddenIndexes: &hiddenIndexes)

      case .unorderedListItem:
        // Extend to line range for cursor detection (same pattern as headings).
        let lineRange = nsText.lineRange(for: safeNodeRange)
        let lineEnd = lineRange.location + lineRange.length
        let lineContentEnd: Int
        if lineEnd > lineRange.location
          && lineEnd <= textLength
          && nsText.character(at: lineEnd - 1) == UInt16(0x000A)
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

        // Apply list item attributes (indentation) to the full line range.
        if !node.attributes.isEmpty {
          styledRanges.append(
            RenderSpec.StyledRange(range: lineRange, attributes: node.attributes))
        }

        // Delimiter is: [leading whitespace][marker char][space]
        // Leading whitespace is ALWAYS hidden (paragraph style handles indentation).
        // The marker portion is hidden/revealed based on cursor position.
        for delim in node.delimiterRanges {
          let safeDelim = clamp(delim, to: textLength)
          guard safeDelim.length > 0 else { continue }

          let markerCharIndex = safeNodeRange.location
          let leadingStart = safeDelim.location
          let delimEnd = safeDelim.location + safeDelim.length

          // Always hide leading whitespace
          if markerCharIndex > leadingStart {
            hiddenIndexes.insert(integersIn: leadingStart..<markerCharIndex)
          }

          // Marker portion: from markerCharIndex to delimEnd
          let markerRange = NSRange(
            location: markerCharIndex,
            length: delimEnd - markerCharIndex)

          if cursorInNode {
            // Cursor inside: show marker dimmed
            if markerRange.length > 0 {
              temporaryAttributes.append(
                RenderSpec.StyledRange(
                  range: markerRange,
                  attributes: [.foregroundColor: style.delimiterColor]))
            }
          } else {
            // Cursor outside: replace marker char with bullet, hide space
            if markerCharIndex < delimEnd {
              bulletIndexes.insert(markerCharIndex)
            }
            if markerCharIndex + 1 < delimEnd {
              hiddenIndexes.insert(integersIn: (markerCharIndex + 1)..<delimEnd)
            }
          }
        }

        // Hide leading whitespace on continuation lines within this list item.
        hideContinuationWhitespace(
          in: nsText, nodeRange: safeNodeRange, textLength: textLength,
          hiddenIndexes: &hiddenIndexes)
      }
    }

    return RenderSpec(
      baseAttributes: style.baseAttributes,
      styledRanges: styledRanges,
      fontTraits: fontTraits,
      hiddenIndexes: hiddenIndexes,
      bulletIndexes: bulletIndexes,
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

  /// Hide leading whitespace on continuation lines within a list item's range.
  ///
  /// Continuation lines are lines after the first within a list item node that
  /// start with whitespace. The whitespace exists in the markdown source to tell
  /// the parser the line belongs to the list item, but visually the paragraph
  /// style's `headIndent` already handles indentation, so the whitespace must
  /// be hidden to avoid double indentation.
  private static func hideContinuationWhitespace(
    in nsText: NSString,
    nodeRange: NSRange,
    textLength: Int,
    hiddenIndexes: inout IndexSet
  ) {
    let nodeEnd = min(nodeRange.location + nodeRange.length, textLength)
    var pos = nodeRange.location

    // Skip the first line (it has the marker, handled separately)
    while pos < nodeEnd {
      if nsText.character(at: pos) == UInt16(0x000A) {  // \n
        pos += 1
        break
      }
      pos += 1
    }

    // Scan subsequent lines for leading whitespace
    while pos < nodeEnd {
      let lineStart = pos
      // Count leading whitespace (spaces and tabs)
      while pos < nodeEnd {
        let ch = nsText.character(at: pos)
        if ch == UInt16(0x0020) || ch == UInt16(0x0009) {  // space or tab
          pos += 1
        } else {
          break
        }
      }
      let wsCount = pos - lineStart
      if wsCount > 0 {
        hiddenIndexes.insert(integersIn: lineStart..<(lineStart + wsCount))
      }

      // Skip to end of this line
      while pos < nodeEnd {
        if nsText.character(at: pos) == UInt16(0x000A) {
          pos += 1
          break
        }
        pos += 1
      }
    }
  }

  static func clamp(_ range: NSRange, to maxLength: Int) -> NSRange {
    let start = min(range.location, maxLength)
    let length = min(range.length, maxLength - start)
    return NSRange(location: start, length: max(0, length))
  }
}
