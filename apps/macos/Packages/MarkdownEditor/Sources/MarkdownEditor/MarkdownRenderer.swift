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
        uncheckedCheckboxIndexes: IndexSet(),
        checkedCheckboxIndexes: IndexSet(),
        temporaryAttributes: [],
        codeBlockCharacterRanges: [],
        blockquoteCharacterRanges: []
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
    var uncheckedCheckboxIndexes = IndexSet()
    var checkedCheckboxIndexes = IndexSet()
    var temporaryAttributes: [RenderSpec.StyledRange] = []
    var codeBlockCharacterRanges: [RenderSpec.CodeBlockRange] = []
    var blockquoteCharacterRanges: [RenderSpec.BlockquoteRange] = []
    // Track blockquote ranges where cursor is inside, so child constructs
    // (like list items) can adjust their baseIndent accordingly.
    var cursorInsideBlockquoteRanges: [(range: NSRange, depth: Int, listBaseIndent: CGFloat)] = []

    for node in nodes {
      let safeContentRange = clamp(node.contentRange, to: textLength)
      let safeNodeRange = clamp(node.range, to: textLength)

      switch node.type {
      case .heading:
        // Detect setext headings: content starts at node start (no `# ` prefix),
        // delimiter is the underline on the next line.
        let isSetext = safeContentRange.location == safeNodeRange.location
          && !node.delimiterRanges.isEmpty

        // For setext headings with a single `-` underline, suppress heading
        // styling when the cursor is on/after the underline. A single `-` is
        // ambiguous (could be the start of a list item), so we don't want the
        // jarring visual change to heading font while the user is typing.
        // Two or more dashes (`--`) are unambiguous and DO get heading styling.
        if isSetext, let delim = node.delimiterRanges.first {
          let safeDelim = clamp(delim, to: textLength)
          // The delimiter includes the \n + underline chars. Extract just the
          // underline text (after the \n), trimming whitespace.
          let delimText = nsText.substring(with: safeDelim)
            .trimmingCharacters(in: .whitespacesAndNewlines)
          let isSingleDash = delimText == "-"

          if isSingleDash {
            // Check if cursor is on the underline line
            let underlineLineRange = nsText.lineRange(
              for: NSRange(location: safeDelim.location + safeDelim.length - 1, length: 0))
            let cursorOnUnderline = cursorRange.location >= underlineLineRange.location
              && cursorRange.location <= underlineLineRange.location + underlineLineRange.length

            if cursorOnUnderline {
              // Skip heading styling entirely — render as plain text
              break
            }
          }
        }

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
        // When inside a blockquote, merge blockquote indentation into the heading's
        // paragraph style so the heading aligns with other blockquote content.
        if !node.attributes.isEmpty {
          let adjustedAttrs = adjustAttributesForBlockquoteHeading(
            node.attributes, nodeRange: safeNodeRange,
            cursorInsideBlockquoteRanges: cursorInsideBlockquoteRanges,
            nodes: nodes, style: style)
          styledRanges.append(
            RenderSpec.StyledRange(range: lineRange, attributes: adjustedAttrs))
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

      case .strikethrough:
        let cursorInNode = cursorOverlaps(
          cursorRange, node: safeNodeRange, textLength: textLength)

        // Apply strikethrough attribute to content range
        if safeContentRange.length > 0, !node.attributes.isEmpty {
          styledRanges.append(
            RenderSpec.StyledRange(range: safeContentRange, attributes: node.attributes))
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
          let adjustedAttrs = adjustAttributesForCursorActiveBlockquote(
            node.attributes, nodeRange: safeNodeRange,
            cursorInsideBlockquoteRanges: cursorInsideBlockquoteRanges,
            nodes: nodes, style: style)
          styledRanges.append(
            RenderSpec.StyledRange(range: lineRange, attributes: adjustedAttrs))
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

        // Leading whitespace handling: when directly inside a cursor-active
        // blockquote, keep whitespace visible (it's part of the raw markdown
        // the user is editing and provides nesting visual). Otherwise hide it
        // (paragraph style handles indentation).
        let orderedDirectBQ = isDirectlyInsideCursorActiveBlockquote(
          nodeRange: safeNodeRange,
          cursorInsideBlockquoteRanges: cursorInsideBlockquoteRanges,
          nodes: nodes)
        if !orderedDirectBQ {
          for delim in node.delimiterRanges {
            let safeDelim = clamp(delim, to: textLength)
            guard safeDelim.length > 0 else { continue }
            hiddenIndexes.insert(
              integersIn: safeDelim.location..<(safeDelim.location + safeDelim.length))
          }
          hideContinuationWhitespace(
            in: nsText, nodeRange: safeNodeRange, textLength: textLength,
            hiddenIndexes: &hiddenIndexes)
          applyContinuationParagraphStyle(
            in: nsText, nodeRange: safeNodeRange, nodeAttributes: node.attributes,
            textLength: textLength, styledRanges: &styledRanges)
        }

      case .checkboxListItem(let checked, _):
        // Checkbox list items: same pattern as unordered but with checkbox glyph
        // substitution instead of bullet. The delimiter covers the full
        // "- [ ] " or "- [x] " prefix (plus leading whitespace for nested items).
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
          let adjustedAttrs = adjustAttributesForCursorActiveBlockquote(
            node.attributes, nodeRange: safeNodeRange,
            cursorInsideBlockquoteRanges: cursorInsideBlockquoteRanges,
            nodes: nodes, style: style)
          styledRanges.append(
            RenderSpec.StyledRange(range: lineRange, attributes: adjustedAttrs))
        }

        let checkboxDirectBQ = isDirectlyInsideCursorActiveBlockquote(
          nodeRange: safeNodeRange,
          cursorInsideBlockquoteRanges: cursorInsideBlockquoteRanges,
          nodes: nodes)

        for delim in node.delimiterRanges {
          let safeDelim = clamp(delim, to: textLength)
          guard safeDelim.length > 0 else { continue }

          let markerCharIndex = safeNodeRange.location
          let leadingStart = safeDelim.location
          let delimEnd = safeDelim.location + safeDelim.length

          // Hide leading whitespace only when NOT in cursor-active blockquote
          if !checkboxDirectBQ, markerCharIndex > leadingStart {
            hiddenIndexes.insert(integersIn: leadingStart..<markerCharIndex)
          }

          // Marker portion: from markerCharIndex to delimEnd
          // For checkbox: "- [ ] " or "- [x] " (6 chars)
          let markerRange = NSRange(
            location: markerCharIndex,
            length: delimEnd - markerCharIndex)

          if cursorInNode {
            // Cursor inside: show full marker dimmed
            if markerRange.length > 0 {
              temporaryAttributes.append(
                RenderSpec.StyledRange(
                  range: markerRange,
                  attributes: [.foregroundColor: style.delimiterColor]))
            }
          } else {
            // Cursor outside: replace marker char with checkbox glyph,
            // hide middle chars, keep trailing space visible for spacing.
            if markerCharIndex < delimEnd {
              if checked {
                checkedCheckboxIndexes.insert(markerCharIndex)
              } else {
                uncheckedCheckboxIndexes.insert(markerCharIndex)
              }
            }
            // Hide chars between glyph and trailing space (but keep the space)
            if markerCharIndex + 1 < delimEnd - 1 {
              hiddenIndexes.insert(integersIn: (markerCharIndex + 1)..<(delimEnd - 1))
            }
          }
        }

        if !checkboxDirectBQ {
          hideContinuationWhitespace(
            in: nsText, nodeRange: safeNodeRange, textLength: textLength,
            hiddenIndexes: &hiddenIndexes)
          applyContinuationParagraphStyle(
            in: nsText, nodeRange: safeNodeRange, nodeAttributes: node.attributes,
            textLength: textLength, styledRanges: &styledRanges)
        }

      case .inlineCode:
        let cursorInNode = cursorOverlaps(
          cursorRange, node: safeNodeRange, textLength: textLength)

        // Apply code attributes (monospace font + background) to content range
        if safeContentRange.length > 0, !node.attributes.isEmpty {
          styledRanges.append(
            RenderSpec.StyledRange(range: safeContentRange, attributes: node.attributes))
        }

        // Delimiter visibility
        applyDelimiterVisibility(
          delimiterRanges: node.delimiterRanges,
          cursorInNode: cursorInNode,
          textLength: textLength,
          style: style,
          hiddenIndexes: &hiddenIndexes,
          temporaryAttributes: &temporaryAttributes)

      case .link:
        let cursorInNode = cursorOverlaps(
          cursorRange, node: safeNodeRange, textLength: textLength)

        // Apply link attributes (blue + underline + URL) to content range
        if safeContentRange.length > 0, !node.attributes.isEmpty {
          styledRanges.append(
            RenderSpec.StyledRange(range: safeContentRange, attributes: node.attributes))
        }

        // Delimiter visibility
        applyDelimiterVisibility(
          delimiterRanges: node.delimiterRanges,
          cursorInNode: cursorInNode,
          textLength: textLength,
          style: style,
          hiddenIndexes: &hiddenIndexes,
          temporaryAttributes: &temporaryAttributes)

      case .image:
        let cursorInNode = cursorOverlaps(
          cursorRange, node: safeNodeRange, textLength: textLength)

        // Apply image attributes (secondary color) to content range
        if safeContentRange.length > 0, !node.attributes.isEmpty {
          styledRanges.append(
            RenderSpec.StyledRange(range: safeContentRange, attributes: node.attributes))
        }

        // Apply italic trait to content range
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

      case .codeBlock(_, let parsedBaseIndent):
        // Code block: the entire multi-line block is the construct.
        // Cursor anywhere within reveals the fences; cursor outside hides them.
        // Use the full node range (including fences) for cursor detection.
        let cursorInNode = cursorOverlaps(
          cursorRange, node: safeNodeRange, textLength: textLength)

        // Compute effective base indent: the parsed value includes both list
        // and blockquote contributions. Reduce the blockquote portion when
        // cursor is inside a parent blockquote (same logic as list items).
        let effectiveBaseIndent: CGFloat
        if isDirectlyInsideCursorActiveBlockquote(
          nodeRange: safeNodeRange,
          cursorInsideBlockquoteRanges: cursorInsideBlockquoteRanges,
          nodes: nodes)
        {
          // Directly inside cursor-active blockquote: visible chars provide layout.
          let parentBQ = cursorInsideBlockquoteRanges
            .filter { bq in
              bq.range.location <= safeNodeRange.location
                && bq.range.location + bq.range.length >= safeNodeRange.location + safeNodeRange.length
            }
            .max(by: { $0.depth < $1.depth })
          effectiveBaseIndent = parentBQ?.listBaseIndent ?? 0
        } else {
          let parentDepth = deepestCursorActiveBlockquoteDepth(
            forNodeAt: safeNodeRange,
            cursorInsideBlockquoteRanges: cursorInsideBlockquoteRanges)
          if parentDepth > 0 {
            let reduction = style.blockquoteIndent * CGFloat(parentDepth)
            effectiveBaseIndent = max(0, parsedBaseIndent - reduction)
          } else {
            effectiveBaseIndent = parsedBaseIndent
          }
        }

        // Apply code attributes (font, paragraph style) to the FULL node
        // range — including fence lines.
        if safeNodeRange.length > 0, !node.attributes.isEmpty {
          var attrs = node.attributes
          if effectiveBaseIndent > 0, let existingPS = attrs[.paragraphStyle] as? NSParagraphStyle {
            let newPS = NSMutableParagraphStyle()
            newPS.setParagraphStyle(existingPS)
            newPS.headIndent += effectiveBaseIndent
            newPS.firstLineHeadIndent += effectiveBaseIndent
            attrs[.paragraphStyle] = newPS.copy() as! NSParagraphStyle
          }
          styledRanges.append(
            RenderSpec.StyledRange(range: safeNodeRange, attributes: attrs))
        }

        // Record the code block range for background drawing.
        if safeNodeRange.length > 0 {
          codeBlockCharacterRanges.append(
            RenderSpec.CodeBlockRange(range: safeNodeRange, baseIndent: effectiveBaseIndent))
        }

        // Delimiter visibility (hidden vs revealed)
        applyDelimiterVisibility(
          delimiterRanges: node.delimiterRanges,
          cursorInNode: cursorInNode,
          textLength: textLength,
          style: style,
          hiddenIndexes: &hiddenIndexes,
          temporaryAttributes: &temporaryAttributes)

      case .blockquote(let depth, let listBaseIndent):
        // Blockquote: the entire multi-line block is the construct.
        // Cursor anywhere within reveals the `> ` prefixes; cursor outside hides them.
        let cursorInNode = cursorOverlaps(
          cursorRange, node: safeNodeRange, textLength: textLength)

        // Compute effective depth: when cursor is inside a parent blockquote,
        // that parent's `> ` prefix is visible (no indent needed), so we only
        // need indentation for the remaining depth levels.
        let parentCursorDepth = deepestCursorActiveBlockquoteDepth(
          forNodeAt: safeNodeRange,
          cursorInsideBlockquoteRanges: cursorInsideBlockquoteRanges)
        let effectiveDepth = cursorInNode ? depth : max(0, depth - parentCursorDepth)

        // Adjust listBaseIndent: if this blockquote's parent list is itself
        // inside a cursor-active blockquote, the list's continuation whitespace
        // is visible and provides the layout offset, so listBaseIndent = 0.
        let effectiveListBaseIndent: CGFloat
        if listBaseIndent > 0, parentCursorDepth > 0 {
          effectiveListBaseIndent = 0
        } else {
          effectiveListBaseIndent = listBaseIndent
        }

        // Apply blockquote attributes using effective depth.
        let blockquoteAttrs = style.blockquoteAttributes(
          depth: effectiveDepth, cursorInside: cursorInNode, baseIndent: effectiveListBaseIndent)

        if safeNodeRange.length > 0 {
          let lineRange = nsText.lineRange(for: safeNodeRange)
          styledRanges.append(
            RenderSpec.StyledRange(range: lineRange, attributes: blockquoteAttrs))
        }

        // Delimiter visibility (hidden vs revealed) for all `> ` prefixes
        applyDelimiterVisibility(
          delimiterRanges: node.delimiterRanges,
          cursorInNode: cursorInNode,
          textLength: textLength,
          style: style,
          hiddenIndexes: &hiddenIndexes,
          temporaryAttributes: &temporaryAttributes)

        // Record blockquote range for left border drawing when cursor is outside.
        // Use effective depth so the border position accounts for cursor-active parents.
        if !cursorInNode && effectiveDepth > 0 && safeNodeRange.length > 0 {
          blockquoteCharacterRanges.append(
            RenderSpec.BlockquoteRange(
              range: safeNodeRange, depth: effectiveDepth,
              listBaseIndent: effectiveListBaseIndent))
        }

        // Track cursor-inside blockquotes so child constructs can adjust indent.
        if cursorInNode {
          cursorInsideBlockquoteRanges.append(
            (range: safeNodeRange, depth: depth, listBaseIndent: effectiveListBaseIndent))
        }

      case .thematicBreak:
        // Thematic break: the entire `---`/`***`/`___` is the construct.
        // When cursor is outside: transparent text + thick strikethrough (horizontal line effect).
        // When cursor is inside: dimmed text, no strikethrough.
        let cursorInNode = cursorOverlaps(
          cursorRange, node: safeNodeRange, textLength: textLength)

        if cursorInNode {
          // Cursor inside: show raw text dimmed, no special attributes
          temporaryAttributes.append(
            RenderSpec.StyledRange(
              range: safeNodeRange,
              attributes: [.foregroundColor: style.delimiterColor]))
        } else {
          // Cursor outside: apply thematic break attributes (transparent text + strikethrough)
          if safeNodeRange.length > 0, !node.attributes.isEmpty {
            styledRanges.append(
              RenderSpec.StyledRange(range: safeNodeRange, attributes: node.attributes))
          }
        }

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
        // Adjust baseIndent when inside a cursor-active blockquote.
        if !node.attributes.isEmpty {
          let adjustedAttrs = adjustAttributesForCursorActiveBlockquote(
            node.attributes, nodeRange: safeNodeRange,
            cursorInsideBlockquoteRanges: cursorInsideBlockquoteRanges,
            nodes: nodes, style: style)
          styledRanges.append(
            RenderSpec.StyledRange(range: lineRange, attributes: adjustedAttrs))
        }

        // Delimiter is: [leading whitespace][marker char][space]
        // When directly inside a cursor-active blockquote, keep leading
        // whitespace visible (raw markdown editing). Otherwise hide it.
        let unorderedDirectBQ = isDirectlyInsideCursorActiveBlockquote(
          nodeRange: safeNodeRange,
          cursorInsideBlockquoteRanges: cursorInsideBlockquoteRanges,
          nodes: nodes)

        for delim in node.delimiterRanges {
          let safeDelim = clamp(delim, to: textLength)
          guard safeDelim.length > 0 else { continue }

          let markerCharIndex = safeNodeRange.location
          let leadingStart = safeDelim.location
          let delimEnd = safeDelim.location + safeDelim.length

          // Hide leading whitespace only when NOT in cursor-active blockquote
          if !unorderedDirectBQ, markerCharIndex > leadingStart {
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
            // Cursor outside: replace marker char with bullet,
            // hide middle chars, keep trailing space visible for spacing.
            if markerCharIndex < delimEnd {
              bulletIndexes.insert(markerCharIndex)
            }
            // Hide chars between glyph and trailing space (but keep the space)
            if markerCharIndex + 1 < delimEnd - 1 {
              hiddenIndexes.insert(integersIn: (markerCharIndex + 1)..<(delimEnd - 1))
            }
          }
        }

        if !unorderedDirectBQ {
          // Hide leading whitespace on continuation lines within this list item.
          hideContinuationWhitespace(
            in: nsText, nodeRange: safeNodeRange, textLength: textLength,
            hiddenIndexes: &hiddenIndexes)

          // Override firstLineHeadIndent on continuation lines so they align
          // with content, not the marker position.
          applyContinuationParagraphStyle(
            in: nsText, nodeRange: safeNodeRange, nodeAttributes: node.attributes,
            textLength: textLength, styledRanges: &styledRanges)
        }
      }
    }

    return RenderSpec(
      baseAttributes: style.baseAttributes,
      styledRanges: styledRanges,
      fontTraits: fontTraits,
      hiddenIndexes: hiddenIndexes,
      bulletIndexes: bulletIndexes,
      uncheckedCheckboxIndexes: uncheckedCheckboxIndexes,
      checkedCheckboxIndexes: checkedCheckboxIndexes,
      temporaryAttributes: temporaryAttributes,
      codeBlockCharacterRanges: codeBlockCharacterRanges,
      blockquoteCharacterRanges: blockquoteCharacterRanges
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

  /// Find the depth of the deepest cursor-active blockquote containing a node.
  /// Returns 0 if the node is not inside any cursor-active blockquote.
  private static func deepestCursorActiveBlockquoteDepth(
    forNodeAt nodeRange: NSRange,
    cursorInsideBlockquoteRanges: [(range: NSRange, depth: Int, listBaseIndent: CGFloat)]
  ) -> Int {
    cursorInsideBlockquoteRanges
      .filter { bq in
        bq.range.location <= nodeRange.location
          && bq.range.location + bq.range.length >= nodeRange.location + nodeRange.length
      }
      .map(\.depth)
      .max() ?? 0
  }

  /// Check if a node is *directly* inside a cursor-active blockquote (i.e., its
  /// innermost containing blockquote is cursor-active). When true, the `> ` prefix
  /// is visible on the same line, so paragraph indent should be zero and leading
  /// whitespace should remain visible (it's part of the raw markdown).
  private static func isDirectlyInsideCursorActiveBlockquote(
    nodeRange: NSRange,
    cursorInsideBlockquoteRanges: [(range: NSRange, depth: Int, listBaseIndent: CGFloat)],
    nodes: [SyntaxNode]
  ) -> Bool {
    let parentDepth = deepestCursorActiveBlockquoteDepth(
      forNodeAt: nodeRange, cursorInsideBlockquoteRanges: cursorInsideBlockquoteRanges)
    guard parentDepth > 0 else { return false }
    let innermostBQDepth = nodes
      .compactMap { n -> Int? in
        guard case .blockquote(let d, _) = n.type,
          n.range.location <= nodeRange.location,
          n.range.location + n.range.length >= nodeRange.location + nodeRange.length
        else { return nil }
        return d
      }
      .max() ?? 0
    return innermostBQDepth <= parentDepth
  }

  /// Adjust child construct attributes when inside a cursor-active blockquote.
  ///
  /// When cursor is inside a blockquote, its `> ` prefix is visible at position 0,
  /// so paragraph indentation must be adjusted:
  /// - Constructs *directly* inside the cursor-active blockquote (no intervening
  ///   non-cursor-active blockquote) get ALL indentation zeroed. The visible `> `
  ///   and any markers provide visual offset through their character widths.
  /// - Constructs inside a deeper *non*-cursor-active blockquote have only the
  ///   cursor-active parent's share subtracted, retaining the hidden inner
  ///   blockquote's indentation.
  private static func adjustAttributesForCursorActiveBlockquote(
    _ attributes: [NSAttributedString.Key: Any],
    nodeRange: NSRange,
    cursorInsideBlockquoteRanges: [(range: NSRange, depth: Int, listBaseIndent: CGFloat)],
    nodes: [SyntaxNode],
    style: MarkdownStyle
  ) -> [NSAttributedString.Key: Any] {
    // Find the deepest cursor-active blockquote containing this node (full tuple).
    let parentBQ = cursorInsideBlockquoteRanges
      .filter { bq in
        bq.range.location <= nodeRange.location
          && bq.range.location + bq.range.length >= nodeRange.location + nodeRange.length
      }
      .max(by: { $0.depth < $1.depth })
    guard let parentBQ = parentBQ else { return attributes }

    guard let existingPS = attributes[.paragraphStyle] as? NSParagraphStyle else {
      return attributes
    }

    // Find the innermost blockquote containing this node.
    let innermostBQDepth = nodes
      .compactMap { n -> Int? in
        guard case .blockquote(let d, _) = n.type,
          n.range.location <= nodeRange.location,
          n.range.location + n.range.length >= nodeRange.location + nodeRange.length
        else { return nil }
        return d
      }
      .max() ?? 0

    let newPS = NSMutableParagraphStyle()
    newPS.setParagraphStyle(existingPS)

    if innermostBQDepth <= parentBQ.depth {
      // Directly inside the cursor-active blockquote (no intervening
      // non-cursor-active blockquote). Set indentation to the parent
      // blockquote's list base indent so the visible `> ` prefix stays
      // at the correct position within its enclosing list item.
      newPS.firstLineHeadIndent = parentBQ.listBaseIndent
      newPS.headIndent = parentBQ.listBaseIndent
    } else {
      // Inside a deeper non-cursor-active blockquote: subtract only the
      // cursor-active parent's share, retaining the inner blockquote's indent.
      let indentReduction = style.blockquoteIndent * CGFloat(parentBQ.depth)
      newPS.firstLineHeadIndent = max(parentBQ.listBaseIndent, newPS.firstLineHeadIndent - indentReduction)
      newPS.headIndent = max(parentBQ.listBaseIndent, newPS.headIndent - indentReduction)
    }

    var newAttrs = attributes
    newAttrs[.paragraphStyle] = newPS.copy() as! NSParagraphStyle
    return newAttrs
  }

  /// Adjust heading attributes when inside a blockquote.
  /// The heading's own paragraph style (with indent 0) overwrites the blockquote's,
  /// so we merge the appropriate blockquote indent into the heading's paragraph style.
  /// Uses effective depth: when cursor is inside a parent blockquote, only the
  /// remaining (non-cursor-active) blockquote levels contribute indent.
  private static func adjustAttributesForBlockquoteHeading(
    _ attributes: [NSAttributedString.Key: Any],
    nodeRange: NSRange,
    cursorInsideBlockquoteRanges: [(range: NSRange, depth: Int, listBaseIndent: CGFloat)],
    nodes: [SyntaxNode],
    style: MarkdownStyle
  ) -> [NSAttributedString.Key: Any] {
    // Find the innermost containing blockquote node
    let containingBQ = nodes.last { node in
      if case .blockquote(let depth, _) = node.type {
        return node.range.location <= nodeRange.location
          && node.range.location + node.range.length >= nodeRange.location + nodeRange.length
          && depth > 0
      }
      return false
    }
    guard let bqNode = containingBQ,
      case .blockquote(let bqDepth, let bqListBaseIndent) = bqNode.type
    else {
      return attributes
    }

    // Compute effective depth: subtract cursor-active parent's depth
    let parentCursorDepth = deepestCursorActiveBlockquoteDepth(
      forNodeAt: nodeRange, cursorInsideBlockquoteRanges: cursorInsideBlockquoteRanges)
    let effectiveDepth = max(0, bqDepth - parentCursorDepth)

    guard effectiveDepth > 0 || bqListBaseIndent > 0 else {
      // Cursor is inside this blockquote (or a blockquote at same/deeper level);
      // `> ` is visible, heading indent should be 0, and no list base indent.
      return attributes
    }

    guard let existingPS = attributes[.paragraphStyle] as? NSParagraphStyle else {
      return attributes
    }
    let newPS = NSMutableParagraphStyle()
    newPS.setParagraphStyle(existingPS)

    if effectiveDepth == 0 {
      // Cursor is inside the blockquote, but there's a list base indent to apply
      newPS.firstLineHeadIndent = newPS.firstLineHeadIndent + bqListBaseIndent
      newPS.headIndent = newPS.headIndent + bqListBaseIndent
    } else {
      // Add the effective blockquote indent plus list base indent to heading
      let totalIndent = bqListBaseIndent + style.blockquoteIndent * CGFloat(effectiveDepth)
      newPS.firstLineHeadIndent = newPS.firstLineHeadIndent + totalIndent
      newPS.headIndent = newPS.headIndent + totalIndent
    }

    var newAttrs = attributes
    newAttrs[.paragraphStyle] = newPS.copy() as! NSParagraphStyle
    return newAttrs
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

  /// Apply a paragraph style override to continuation lines within a list item
  /// so that `firstLineHeadIndent` matches `headIndent`. Without this, continuation
  /// lines (whose leading whitespace is hidden) would start at the marker position
  /// instead of the content position.
  private static func applyContinuationParagraphStyle(
    in nsText: NSString,
    nodeRange: NSRange,
    nodeAttributes: [NSAttributedString.Key: Any],
    textLength: Int,
    styledRanges: inout [RenderSpec.StyledRange]
  ) {
    guard let existingStyle = nodeAttributes[.paragraphStyle] as? NSParagraphStyle else { return }
    let headIndent = existingStyle.headIndent
    guard headIndent > 0 else { return }

    let nodeEnd = min(nodeRange.location + nodeRange.length, textLength)
    var pos = nodeRange.location

    // Skip the first line (it uses firstLineHeadIndent for the marker position)
    while pos < nodeEnd {
      if nsText.character(at: pos) == UInt16(0x000A) {
        pos += 1
        break
      }
      pos += 1
    }

    // Apply overridden paragraph style to each continuation line
    while pos < nodeEnd {
      let lineStart = pos
      // Find end of this line
      while pos < nodeEnd && nsText.character(at: pos) != UInt16(0x000A) {
        pos += 1
      }
      let lineEnd = pos
      if pos < nodeEnd { pos += 1 }  // skip \n

      let lineLength = lineEnd - lineStart
      guard lineLength > 0 else { continue }

      let contStyle = NSMutableParagraphStyle()
      contStyle.setParagraphStyle(existingStyle)
      contStyle.firstLineHeadIndent = headIndent
      styledRanges.append(
        RenderSpec.StyledRange(
          range: NSRange(location: lineStart, length: lineLength),
          attributes: [.paragraphStyle: contStyle.copy() as! NSParagraphStyle]))
    }
  }

  static func clamp(_ range: NSRange, to maxLength: Int) -> NSRange {
    let start = min(range.location, maxLength)
    let length = min(range.length, maxLength - start)
    return NSRange(location: start, length: max(0, length))
  }
}
