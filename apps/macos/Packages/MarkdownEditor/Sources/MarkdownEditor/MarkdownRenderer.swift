import AppKit
import Markdown

/// Pure computation: given editor state, produce a complete rendering specification.
@MainActor
enum MarkdownRenderer {
  private struct BlockRenderContext {
    var hiddenIndent: CGFloat
    var visibleQuoteWidth: CGFloat
    /// The `hiddenIndent` at the point where the outermost visible blockquote
    /// was entered. All descendants use this for `firstLineHeadIndent` so that
    /// visible `>` characters stay vertically aligned regardless of inner nesting.
    var quoteAlignIndent: CGFloat
    var foregroundColor: NSColor
  }

  private struct RenderAccumulator {
    var styledRanges: [RenderSpec.StyledRange] = []
    var fontTraits: [RenderSpec.TraitApplication] = []
    var hiddenIndexes = IndexSet()
    var bulletIndexes = IndexSet()
    var uncheckedCheckboxIndexes = IndexSet()
    var checkedCheckboxIndexes = IndexSet()
    var collapsedNewlineIndexes = IndexSet()
    var temporaryAttributes: [RenderSpec.StyledRange] = []
    var codeBlockCharacterRanges: [RenderSpec.CodeBlockDecoration] = []
    var blockquoteCharacterRanges: [RenderSpec.BlockquoteDecoration] = []
  }

  static func render(
    state: EditorState,
    style: MarkdownStyle = .default
  ) -> RenderSpec {
    render(text: state.markdown, cursorRange: state.selection.nsRange, style: style)
  }

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
        collapsedNewlineIndexes: IndexSet(),
        temporaryAttributes: [],
        codeBlockCharacterRanges: [],
        blockquoteCharacterRanges: []
      )
    }

    let document = Document(parsing: text)
    let converter = SourceRangeConverter(string: text)
    var parser = MarkdownParser(converter: converter, style: style)
    parser.visit(document)
    guard let parsed = parser.document else {
      return RenderSpec(
        baseAttributes: style.baseAttributes,
        styledRanges: [],
        fontTraits: [],
        hiddenIndexes: IndexSet(),
        bulletIndexes: IndexSet(),
        uncheckedCheckboxIndexes: IndexSet(),
        checkedCheckboxIndexes: IndexSet(),
        collapsedNewlineIndexes: IndexSet(),
        temporaryAttributes: [],
        codeBlockCharacterRanges: [],
        blockquoteCharacterRanges: []
      )
    }

    return buildSpec(
      document: parsed,
      cursorRange: cursorRange,
      text: text,
      textLength: textLength,
      style: style
    )
  }

  static func buildSpec(
    document: MarkdownDocument,
    cursorRange: NSRange,
    text: String,
    textLength: Int,
    style: MarkdownStyle
  ) -> RenderSpec {
    let nsText = text as NSString
    var accumulator = RenderAccumulator()
    let rootContext = BlockRenderContext(
      hiddenIndent: 0,
      visibleQuoteWidth: 0,
      quoteAlignIndent: 0,
      foregroundColor: style.textColor
    )

    for (i, block) in document.blocks.enumerated() {
      renderBlock(
        block,
        context: rootContext,
        cursorRange: cursorRange,
        nsText: nsText,
        textLength: textLength,
        style: style,
        accumulator: &accumulator,
        previousSiblingKind: i > 0 ? document.blocks[i - 1].kind : nil
      )
    }

    for inlineNode in document.inlineNodes {
      applyInlineNode(
        inlineNode,
        cursorRange: cursorRange,
        textLength: textLength,
        style: style,
        accumulator: &accumulator
      )
    }

    // Collapse the first blank line in each inter-block gap so that \n\n
    // paragraph separators render identically to the old single-\n behavior.
    collapseInterBlockBlankLines(
      between: document.blocks,
      nsText: nsText,
      textLength: textLength,
      accumulator: &accumulator
    )

    return RenderSpec(
      baseAttributes: style.baseAttributes,
      styledRanges: accumulator.styledRanges,
      fontTraits: accumulator.fontTraits,
      hiddenIndexes: accumulator.hiddenIndexes,
      bulletIndexes: accumulator.bulletIndexes,
      uncheckedCheckboxIndexes: accumulator.uncheckedCheckboxIndexes,
      checkedCheckboxIndexes: accumulator.checkedCheckboxIndexes,
      collapsedNewlineIndexes: accumulator.collapsedNewlineIndexes,
      temporaryAttributes: accumulator.temporaryAttributes,
      codeBlockCharacterRanges: accumulator.codeBlockCharacterRanges,
      blockquoteCharacterRanges: accumulator.blockquoteCharacterRanges
    )
  }

  // MARK: - Block Rendering

  private static func renderBlock(
    _ block: MarkdownBlock,
    context: BlockRenderContext,
    cursorRange: NSRange,
    nsText: NSString,
    textLength: Int,
    style: MarkdownStyle,
    accumulator: inout RenderAccumulator,
    suppressParagraphStyle: Bool = false,
    previousSiblingKind: MarkdownBlockKind? = nil
  ) {
    let safeRange = clamp(block.range, to: textLength)
    guard safeRange.length > 0 else { return }

    switch block.kind {
    case .paragraph:
      guard !suppressParagraphStyle else { return }
      let paragraphRange = nsText.lineRange(for: safeRange)
      let afterContainer: Bool
      switch previousSiblingKind {
      case .unorderedList, .orderedList, .blockquote: afterContainer = true
      default: afterContainer = false
      }
      let spacingBefore = afterContainer
        ? style.paragraphSpacing + style.spacingAfterContainerBlock
        : style.paragraphSpacing
      applyParagraphStyle(
        to: paragraphRange,
        context: context,
        font: style.baseFont,
        color: context.foregroundColor,
        paragraphSpacingBefore: spacingBefore,
        paragraphSpacing: style.paragraphSpacing,
        style: style,
        accumulator: &accumulator
      )

    case .heading(let level, let contentRange, let delimiterRanges):
      renderHeading(
        range: safeRange,
        contentRange: clamp(contentRange, to: textLength),
        delimiterRanges: delimiterRanges,
        level: level,
        context: context,
        cursorRange: cursorRange,
        nsText: nsText,
        textLength: textLength,
        style: style,
        accumulator: &accumulator
      )

    case .blockquote(let prefixRanges):
      let cursorInside = cursorOverlaps(cursorRange, node: safeRange, textLength: textLength)

      var nextContext = context
      nextContext.foregroundColor = .secondaryLabelColor

      if !cursorInside {
        accumulator.blockquoteCharacterRanges.append(
          RenderSpec.BlockquoteDecoration(
            range: safeRange,
            xPosition: context.hiddenIndent + context.visibleQuoteWidth + style.blockquoteBorderLeftPadding
          ))
      }

      if context.visibleQuoteWidth == 0 {
        nextContext.quoteAlignIndent = context.hiddenIndent
      }
      nextContext.visibleQuoteWidth += style.blockquoteIndent

      // Kern > so the > glyph + kern = blockquoteIndent exactly.
      // The > glyph is always present (same kern, same pair kerning with the
      // following character) — visible when the cursor is inside, transparent
      // when outside. This guarantees identical content positioning regardless
      // of cursor location, because NSAttributedString.kern is additive with
      // font pair kerning: using the same glyph and kern in both modes
      // ensures the pair-kerning contribution is identical.
      let gtKern = style.blockquoteIndent - style.textWidth(">")
      let gtColor: NSColor = cursorInside ? style.delimiterColor : .clear
      for prefix in prefixRanges {
        let safePrefix = clamp(prefix, to: textLength)
        guard safePrefix.length > 0 else { continue }

        let gtRange = NSRange(location: safePrefix.location, length: 1)
        accumulator.temporaryAttributes.append(
          RenderSpec.StyledRange(
            range: gtRange,
            attributes: [.foregroundColor: gtColor]))
        accumulator.styledRanges.append(
          RenderSpec.StyledRange(
            range: gtRange,
            attributes: [.kern: gtKern]))

        // Hide the space after >
        if safePrefix.length > 1 {
          accumulator.hiddenIndexes.insert(
            integersIn: (safePrefix.location + 1)..<(safePrefix.location + safePrefix.length))
        }
      }

      for (i, child) in block.children.enumerated() {
        renderBlock(
          child,
          context: nextContext,
          cursorRange: cursorRange,
          nsText: nsText,
          textLength: textLength,
          style: style,
          accumulator: &accumulator,
          previousSiblingKind: i > 0 ? block.children[i - 1].kind : nil
        )
      }

      collapseInterBlockBlankLines(
        between: block.children,
        nsText: nsText,
        textLength: textLength,
        accumulator: &accumulator
      )

    case .unorderedList, .orderedList:
      for (i, child) in block.children.enumerated() {
        renderBlock(
          child,
          context: context,
          cursorRange: cursorRange,
          nsText: nsText,
          textLength: textLength,
          style: style,
          accumulator: &accumulator,
          previousSiblingKind: i > 0 ? block.children[i - 1].kind : nil
        )
      }

      collapseInterBlockBlankLines(
        between: block.children,
        nsText: nsText,
        textLength: textLength,
        accumulator: &accumulator
      )

    case .listItem(let syntax):
      renderListItem(
        block,
        syntax: syntax,
        context: context,
        cursorRange: cursorRange,
        nsText: nsText,
        textLength: textLength,
        style: style,
        accumulator: &accumulator
      )

    case .codeBlock(_, _, let openingFenceRange, let closingFenceRange):
      let cursorInside = cursorOverlaps(cursorRange, node: safeRange, textLength: textLength)
      let insideQuote = context.visibleQuoteWidth > 0
      let localInset: CGFloat = 12
      let textOrigin = context.hiddenIndent + context.visibleQuoteWidth + localInset
      let boxOrigin = context.hiddenIndent + context.visibleQuoteWidth
      let paragraphRange = nsText.lineRange(for: safeRange)

      let paragraphStyle = NSMutableParagraphStyle()
      paragraphStyle.firstLineHeadIndent = insideQuote ? context.quoteAlignIndent : textOrigin
      paragraphStyle.headIndent = textOrigin
      paragraphStyle.tailIndent = -12
      paragraphStyle.paragraphSpacing = style.codeBlockSpacing
      paragraphStyle.paragraphSpacingBefore = style.codeBlockSpacing
      paragraphStyle.lineHeightMultiple = style.lineHeightMultiple

      accumulator.styledRanges.append(
        RenderSpec.StyledRange(
          range: paragraphRange,
          attributes: [
            .paragraphStyle: paragraphStyle.copy() as! NSParagraphStyle,
          ]))

      accumulator.styledRanges.append(
        RenderSpec.StyledRange(
          range: safeRange,
          attributes: [
            .font: style.codeFont,
            .foregroundColor: context.foregroundColor,
          ]))

      // safeRange is contiguous and includes blockquote prefix characters
      // (> and whitespace) on continuation lines when inside a blockquote.
      // Always override these prefix characters to baseFont so they have
      // consistent line height whether the > is visible or hidden — hidden
      // paragraph-start characters become ZWSP .controlCharacter glyphs that
      // still participate in line height, so using codeFont here would cause
      // a vertical shift when the cursor enters/leaves the blockquote.
      // When inside a visible blockquote, also override the innermost > kern
      // to include localInset so code text stays at the same position
      // regardless of cursor location.
      let firstLineStart = nsText.lineRange(
        for: NSRange(location: safeRange.location, length: 0)).location
      let prefixLength = safeRange.location - firstLineStart
      if prefixLength > 0 {
        let end = min(safeRange.location + safeRange.length, textLength)

        if insideQuote {
          // Override innermost > kern on the first line
          applyCodeBlockGtKernOverride(
            lineStart: firstLineStart,
            prefixLength: prefixLength,
            localInset: localInset,
            textLength: textLength,
            nsText: nsText,
            style: style,
            accumulator: &accumulator
          )
        }

        var pos = safeRange.location
        // Skip to the second line within safeRange
        while pos < end {
          if nsText.character(at: pos) == UInt16(0x000A) { pos += 1; break }
          pos += 1
        }
        // For each subsequent line, override code font on the prefix portion
        // and (when visible) override innermost > kern for correct spacing
        while pos < end {
          let prefixEnd = min(pos + prefixLength, end)
          accumulator.styledRanges.append(
            RenderSpec.StyledRange(
              range: NSRange(location: pos, length: prefixEnd - pos),
              attributes: [.font: style.baseFont]))
          if insideQuote {
            applyCodeBlockGtKernOverride(
              lineStart: pos,
              prefixLength: prefixLength,
              localInset: localInset,
              textLength: textLength,
              nsText: nsText,
              style: style,
              accumulator: &accumulator
            )
          }
          // Advance to next line
          while pos < end {
            if nsText.character(at: pos) == UInt16(0x000A) { pos += 1; break }
            pos += 1
          }
        }
      }

      accumulator.codeBlockCharacterRanges.append(
        RenderSpec.CodeBlockDecoration(range: safeRange, xOrigin: boxOrigin))

      var delimiterRanges = [openingFenceRange]
      if let closingFenceRange {
        delimiterRanges.append(closingFenceRange)
      }
      applyDelimiterVisibility(
        delimiterRanges: delimiterRanges,
        cursorInNode: cursorInside,
        textLength: textLength,
        style: style,
        hiddenIndexes: &accumulator.hiddenIndexes,
        temporaryAttributes: &accumulator.temporaryAttributes
      )

    case .thematicBreak:
      // Use a content range (excluding trailing newline) for cursor detection
      // so that a cursor on the NEXT line doesn't keep the break in edit mode.
      let rangeEnd = safeRange.location + safeRange.length
      let contentEnd: Int
      if rangeEnd > safeRange.location && rangeEnd <= textLength
        && nsText.character(at: rangeEnd - 1) == UInt16(0x000A)
      {
        contentEnd = rangeEnd - 1
      } else {
        contentEnd = rangeEnd
      }
      let contentRange = NSRange(location: safeRange.location, length: contentEnd - safeRange.location)
      let cursorInside = cursorOverlaps(cursorRange, node: contentRange, textLength: textLength)
      let paragraphStyle = NSMutableParagraphStyle()
      paragraphStyle.firstLineHeadIndent = context.visibleQuoteWidth > 0 ? context.quoteAlignIndent : context.hiddenIndent + context.visibleQuoteWidth
      paragraphStyle.headIndent = context.hiddenIndent + context.visibleQuoteWidth
      paragraphStyle.paragraphSpacing = style.baseParagraphSpacing
      paragraphStyle.lineHeightMultiple = style.lineHeightMultiple

      // Always apply the paragraph style so line height is identical in both modes.
      accumulator.styledRanges.append(
        RenderSpec.StyledRange(
          range: safeRange,
          attributes: [.paragraphStyle: paragraphStyle.copy() as! NSParagraphStyle]))

      if cursorInside {
        accumulator.temporaryAttributes.append(
          RenderSpec.StyledRange(
            range: safeRange,
            attributes: [.foregroundColor: style.delimiterColor]))
      } else {
        accumulator.styledRanges.append(
          RenderSpec.StyledRange(range: safeRange, attributes: style.thematicBreakAttributes))
      }
    }
  }

  private static func renderHeading(
    range: NSRange,
    contentRange: NSRange,
    delimiterRanges: [NSRange],
    level: Int,
    context: BlockRenderContext,
    cursorRange: NSRange,
    nsText: NSString,
    textLength: Int,
    style: MarkdownStyle,
    accumulator: inout RenderAccumulator
  ) {
    let isSetext = contentRange.location == range.location && !delimiterRanges.isEmpty
    if isSetext, let delimiter = delimiterRanges.first {
      let safeDelimiter = clamp(delimiter, to: textLength)
      let delimiterText = nsText.substring(with: safeDelimiter).trimmingCharacters(in: .whitespacesAndNewlines)
      if delimiterText == "-" {
        let underlineLineRange = nsText.lineRange(
          for: NSRange(location: safeDelimiter.location + max(0, safeDelimiter.length - 1), length: 0))
        let cursorOnUnderline =
          cursorRange.location >= underlineLineRange.location
          && cursorRange.location <= underlineLineRange.location + underlineLineRange.length
        if cursorOnUnderline {
          return
        }
      }
    }

    let lineRange = nsText.lineRange(for: range)
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
    let lineContentRange = NSRange(location: lineRange.location, length: lineContentEnd - lineRange.location)
    let cursorInside = cursorOverlaps(cursorRange, node: lineContentRange, textLength: textLength)

    let paragraphStyle = NSMutableParagraphStyle()
    paragraphStyle.firstLineHeadIndent = context.visibleQuoteWidth > 0 ? context.quoteAlignIndent : context.hiddenIndent
    paragraphStyle.headIndent = context.hiddenIndent + context.visibleQuoteWidth
    paragraphStyle.paragraphSpacingBefore = level <= 2 ? style.headingSpacingBeforeMajor : style.headingSpacingBeforeMinor
    paragraphStyle.paragraphSpacing = style.headingSpacingAfter
    paragraphStyle.lineHeightMultiple = style.lineHeightMultiple

    // Apply the heading font only to the content range (excluding the trailing
    // newline). This prevents the heading's larger font metrics from bleeding
    // into the next line, which would make the cursor on the following line
    // inherit the heading's height.
    accumulator.styledRanges.append(
      RenderSpec.StyledRange(
        range: lineContentRange,
        attributes: [
          .font: style.headingFont(level: level),
          .foregroundColor: context.foregroundColor,
          .paragraphStyle: paragraphStyle.copy() as! NSParagraphStyle,
        ]))

    // When inside a blockquote, lineContentRange includes the > prefix characters.
    // Override them to baseFont so the heading's larger/bolder font doesn't
    // affect their width (which would cause horizontal shift) or render
    // them visually inconsistent with > on other blockquote lines.
    let prefixLength = range.location - lineRange.location
    if prefixLength > 0 {
      accumulator.styledRanges.append(
        RenderSpec.StyledRange(
          range: NSRange(location: lineRange.location, length: prefixLength),
          attributes: [.font: style.baseFont]))
    }

    applyDelimiterVisibility(
      delimiterRanges: delimiterRanges,
      cursorInNode: cursorInside,
      textLength: textLength,
      style: style,
      hiddenIndexes: &accumulator.hiddenIndexes,
      temporaryAttributes: &accumulator.temporaryAttributes
    )
  }

  private static func renderListItem(
    _ block: MarkdownBlock,
    syntax: ListItemSyntax,
    context: BlockRenderContext,
    cursorRange: NSRange,
    nsText: NSString,
    textLength: Int,
    style: MarkdownStyle,
    accumulator: inout RenderAccumulator
  ) {
    let safeRange = clamp(block.range, to: textLength)
    let cursorInside = cursorOverlaps(cursorRange, node: safeRange, textLength: textLength)
    let insideQuote = context.visibleQuoteWidth > 0

    let lineStart = nsText.lineRange(for: NSRange(location: safeRange.location, length: 0)).location
    let markerRawWidth = style.textWidth(syntax.markerText)
    let leadingWhitespaceWidth = renderedWidth(
      for: syntax.leadingWhitespaceRange,
      nsText: nsText,
      textLength: textLength,
      style: style)

    let outsideDisplayText: String
    let outsideMarkerWidth: CGFloat
    switch syntax.kind {
    case .unordered:
      outsideDisplayText = "\u{2022} "
      outsideMarkerWidth = style.textWidth(outsideDisplayText)
    case .checkbox(let checked):
      outsideDisplayText = checked ? "\u{2612} " : "\u{25A1} "
      outsideMarkerWidth = style.textWidth(outsideDisplayText)
    case .ordered(let widestMarkerText):
      outsideDisplayText = widestMarkerText
      outsideMarkerWidth = style.textWidth(widestMarkerText)
    }

    let markerWidth = cursorInside && !isOrderedListItem(syntax.kind) ? markerRawWidth : outsideMarkerWidth

    // When inside a visible blockquote, the `>` prefix and leading whitespace are
    // real visible characters — don't hide them and don't include their width in
    // firstLineHeadIndent. quoteAlignIndent keeps all `>` characters vertically
    // aligned. hiddenIndent still accumulates normally so child blocks (e.g. code
    // blocks) know their true visual indent for backgrounds and wrapping.
    let firstLineIndent: CGFloat
    let contentIndent: CGFloat
    let childHiddenIndent = context.hiddenIndent + leadingWhitespaceWidth + markerWidth
    if insideQuote {
      firstLineIndent = context.quoteAlignIndent
      contentIndent = context.hiddenIndent + context.visibleQuoteWidth + leadingWhitespaceWidth + markerWidth
    } else {
      firstLineIndent = context.hiddenIndent + leadingWhitespaceWidth
      contentIndent = firstLineIndent + markerWidth
    }

    // When inside a visible blockquote, keep leading whitespace visible so it
    // naturally spaces between the `>` prefix and the list marker.
    if !insideQuote, let leadingWhitespaceRange = syntax.leadingWhitespaceRange {
      insertHidden(range: clamp(leadingWhitespaceRange, to: textLength), into: &accumulator.hiddenIndexes)
    }

    switch syntax.kind {
    case .unordered:
      applyUnorderedMarker(
        markerRange: clamp(syntax.markerRange, to: textLength),
        cursorInside: cursorInside,
        style: style,
        accumulator: &accumulator
      )
    case .checkbox(let checked):
      applyCheckboxMarker(
        markerRange: clamp(syntax.markerRange, to: textLength),
        checked: checked,
        cursorInside: cursorInside,
        style: style,
        accumulator: &accumulator
      )
    case .ordered:
      break
    }

    let styledRange = listStyledRange(for: block, lineStart: lineStart)
    let paragraphStyle = NSMutableParagraphStyle()
    paragraphStyle.firstLineHeadIndent = firstLineIndent
    paragraphStyle.headIndent = contentIndent
    paragraphStyle.paragraphSpacing = style.listItemSpacing
    paragraphStyle.lineHeightMultiple = style.lineHeightMultiple

    accumulator.styledRanges.append(
      RenderSpec.StyledRange(
        range: styledRange,
        attributes: [
          .font: style.baseFont,
          .foregroundColor: context.foregroundColor,
          .paragraphStyle: paragraphStyle.copy() as! NSParagraphStyle,
        ]))

    // Scope continuation whitespace handling to the first child paragraph + gap
    // before the next sibling. This prevents the list item's marker-width kern/hide
    // from being applied to nested child blocks (blockquotes, nested lists) which
    // handle their own continuation whitespace independently.
    var childContext = context
    childContext.hiddenIndent = childHiddenIndent

    if let firstChild = block.children.first, firstChildSharesMarkerLine(firstChild, itemRange: safeRange, nsText: nsText) {
      // Compute continuation range: from the list item start to the end of the
      // first child paragraph (or next sibling). This prevents kern/hide/paragraph-
      // style overrides from bleeding into nested child blocks.
      let contEnd: Int
      if block.children.count > 1 {
        contEnd = block.children[1].range.location
      } else {
        // Single child — use its range end, stripped of trailing newlines,
        // to avoid kern/hide bleeding past this line into the next block.
        var end = firstChild.range.location + firstChild.range.length
        while end > firstChild.range.location
          && end <= textLength
          && nsText.character(at: end - 1) == UInt16(0x000A)
        {
          end -= 1
        }
        contEnd = end
      }
      let contRange = NSRange(
        location: safeRange.location,
        length: contEnd - safeRange.location)

      if insideQuote {
        kernContinuationWhitespaceInVisibleQuote(
          in: contRange,
          targetWidth: contentIndent - context.visibleQuoteWidth,
          nsText: nsText,
          textLength: textLength,
          style: style,
          accumulator: &accumulator
        )
      } else {
        hideIndentedContinuationWhitespace(
          in: contRange,
          nsText: nsText,
          textLength: textLength,
          hiddenIndexes: &accumulator.hiddenIndexes
        )
      }

      if !insideQuote, isPlainParagraphBlock(firstChild) {
        let firstChildContRange = NSRange(
          location: firstChild.range.location,
          length: contEnd - firstChild.range.location)
        applyListContinuationParagraphStyles(
          in: firstChildContRange,
          contentIndent: contentIndent,
          font: style.baseFont,
          color: context.foregroundColor,
          paragraphSpacing: style.listItemSpacing,
          style: style,
          nsText: nsText,
          textLength: textLength,
          accumulator: &accumulator
        )
      }
      let suppress = isPlainParagraphBlock(firstChild)
      renderBlock(
        firstChild,
        context: childContext,
        cursorRange: cursorRange,
        nsText: nsText,
        textLength: textLength,
        style: style,
        accumulator: &accumulator,
        suppressParagraphStyle: suppress
      )
      for child in block.children.dropFirst() {
        renderBlock(
          child,
          context: childContext,
          cursorRange: cursorRange,
          nsText: nsText,
          textLength: textLength,
          style: style,
          accumulator: &accumulator
        )
      }
    } else {
      // First child doesn't share marker line — apply kern/hide to full range.
      if insideQuote {
        kernContinuationWhitespaceInVisibleQuote(
          in: safeRange,
          targetWidth: contentIndent - context.visibleQuoteWidth,
          nsText: nsText,
          textLength: textLength,
          style: style,
          accumulator: &accumulator
        )
      } else {
        hideIndentedContinuationWhitespace(
          in: safeRange,
          nsText: nsText,
          textLength: textLength,
          hiddenIndexes: &accumulator.hiddenIndexes
        )
      }
      for child in block.children {
        renderBlock(
          child,
          context: childContext,
          cursorRange: cursorRange,
          nsText: nsText,
          textLength: textLength,
          style: style,
          accumulator: &accumulator
        )
      }
    }

    collapseInterBlockBlankLines(
      between: block.children,
      nsText: nsText,
      textLength: textLength,
      accumulator: &accumulator
    )
  }

  // MARK: - Inline Rendering

  private static func applyInlineNode(
    _ node: InlineSyntaxNode,
    cursorRange: NSRange,
    textLength: Int,
    style: MarkdownStyle,
    accumulator: inout RenderAccumulator
  ) {
    let safeContentRange = clamp(node.contentRange, to: textLength)
    let safeRange = clamp(node.range, to: textLength)
    let cursorInside = cursorOverlaps(cursorRange, node: safeRange, textLength: textLength)

    switch node.kind {
    case .strong:
      if safeContentRange.length > 0 {
        accumulator.fontTraits.append(
          RenderSpec.TraitApplication(range: safeContentRange, trait: .boldFontMask))
      }
    case .emphasis:
      if safeContentRange.length > 0 {
        accumulator.fontTraits.append(
          RenderSpec.TraitApplication(range: safeContentRange, trait: .italicFontMask))
      }
    case .inlineCode:
      if safeContentRange.length > 0 {
        accumulator.styledRanges.append(
          RenderSpec.StyledRange(range: safeContentRange, attributes: node.attributes))
      }
    case .link:
      if safeContentRange.length > 0 {
        accumulator.styledRanges.append(
          RenderSpec.StyledRange(range: safeContentRange, attributes: node.attributes))
      }
    case .image:
      if safeContentRange.length > 0 {
        accumulator.styledRanges.append(
          RenderSpec.StyledRange(range: safeContentRange, attributes: node.attributes))
        accumulator.fontTraits.append(
          RenderSpec.TraitApplication(range: safeContentRange, trait: .italicFontMask))
      }
    case .strikethrough:
      if safeContentRange.length > 0 {
        accumulator.styledRanges.append(
          RenderSpec.StyledRange(range: safeContentRange, attributes: node.attributes))
      }
    }

    applyDelimiterVisibility(
      delimiterRanges: node.delimiterRanges,
      cursorInNode: cursorInside,
      textLength: textLength,
      style: style,
      hiddenIndexes: &accumulator.hiddenIndexes,
      temporaryAttributes: &accumulator.temporaryAttributes
    )
  }

  // MARK: - Helpers

  private static func applyParagraphStyle(
    to range: NSRange,
    context: BlockRenderContext,
    font: NSFont,
    color: NSColor,
    paragraphSpacingBefore: CGFloat,
    paragraphSpacing: CGFloat,
    style: MarkdownStyle = .default,
    accumulator: inout RenderAccumulator
  ) {
    let paragraphStyle = NSMutableParagraphStyle()
    paragraphStyle.firstLineHeadIndent = context.visibleQuoteWidth > 0 ? context.quoteAlignIndent : context.hiddenIndent
    paragraphStyle.headIndent = context.hiddenIndent + context.visibleQuoteWidth
    paragraphStyle.paragraphSpacingBefore = paragraphSpacingBefore
    paragraphStyle.paragraphSpacing = paragraphSpacing
    paragraphStyle.lineHeightMultiple = style.lineHeightMultiple

    accumulator.styledRanges.append(
      RenderSpec.StyledRange(
        range: range,
        attributes: [
          .font: font,
          .foregroundColor: color,
          .paragraphStyle: paragraphStyle.copy() as! NSParagraphStyle,
        ]))
  }

  private static func listStyledRange(for block: MarkdownBlock, lineStart: Int) -> NSRange {
    // For plain paragraph children, limit to the first child's range. This
    // prevents the list item's headIndent from bleeding into sibling blocks
    // (nested blockquotes, subsequent paragraphs). Blank Shift+Return lines
    // are part of the paragraph content and already within this range.
    if let firstChild = block.children.first, isPlainParagraphBlock(firstChild) {
      let end = firstChild.range.location + firstChild.range.length
      return NSRange(location: lineStart, length: end - lineStart)
    }
    return NSRange(location: lineStart, length: block.range.location + block.range.length - lineStart)
  }

  private static func firstChildSharesMarkerLine(
    _ child: MarkdownBlock,
    itemRange: NSRange,
    nsText: NSString
  ) -> Bool {
    nsText.lineRange(for: NSRange(location: child.range.location, length: 0)).location
      == nsText.lineRange(for: NSRange(location: itemRange.location, length: 0)).location
  }

  private static func isPlainParagraphBlock(_ block: MarkdownBlock) -> Bool {
    if case .paragraph = block.kind { return true }
    return false
  }

  private static func isOrderedListItem(_ kind: ListItemKind) -> Bool {
    if case .ordered = kind { return true }
    return false
  }

  private static func applyUnorderedMarker(
    markerRange: NSRange,
    cursorInside: Bool,
    style: MarkdownStyle,
    accumulator: inout RenderAccumulator
  ) {
    guard markerRange.length > 0 else { return }
    if cursorInside {
      accumulator.temporaryAttributes.append(
        RenderSpec.StyledRange(
          range: markerRange,
          attributes: [.foregroundColor: style.delimiterColor]))
      return
    }

    accumulator.bulletIndexes.insert(markerRange.location)
    if markerRange.location + 1 < markerRange.location + markerRange.length - 1 {
      accumulator.hiddenIndexes.insert(
        integersIn: (markerRange.location + 1)..<(markerRange.location + markerRange.length - 1))
    }
  }

  private static func applyCheckboxMarker(
    markerRange: NSRange,
    checked: Bool,
    cursorInside: Bool,
    style: MarkdownStyle,
    accumulator: inout RenderAccumulator
  ) {
    guard markerRange.length > 0 else { return }
    if cursorInside {
      accumulator.temporaryAttributes.append(
        RenderSpec.StyledRange(
          range: markerRange,
          attributes: [.foregroundColor: style.delimiterColor]))
      return
    }

    if checked {
      accumulator.checkedCheckboxIndexes.insert(markerRange.location)
    } else {
      accumulator.uncheckedCheckboxIndexes.insert(markerRange.location)
    }
    if markerRange.location + 1 < markerRange.location + markerRange.length - 1 {
      accumulator.hiddenIndexes.insert(
        integersIn: (markerRange.location + 1)..<(markerRange.location + markerRange.length - 1))
    }
  }

  private static func hideIndentedContinuationWhitespace(
    in range: NSRange,
    nsText: NSString,
    textLength: Int,
    hiddenIndexes: inout IndexSet
  ) {
    let end = min(range.location + range.length, textLength)
    var pos = range.location

    while pos < end {
      if nsText.character(at: pos) == UInt16(0x000A) {
        pos += 1
        break
      }
      pos += 1
    }

    while pos < end {
      let lineStart = pos
      pos = skipQuotePrefixes(in: nsText, from: lineStart, limit: end)
      // Hide whitespace, then skip any embedded > prefixes (from nested
      // blockquotes separated by list continuation whitespace) and hide
      // whitespace after them too.
      var foundMore = true
      while foundMore {
        let wsStart = pos
        while pos < end {
          let ch = nsText.character(at: pos)
          if ch == UInt16(0x0020) || ch == UInt16(0x0009) {
            pos += 1
          } else {
            break
          }
        }
        if pos > wsStart {
          hiddenIndexes.insert(integersIn: wsStart..<pos)
        }
        // Check for another > prefix beyond the whitespace
        let before = pos
        pos = skipQuotePrefixes(in: nsText, from: pos, limit: end)
        foundMore = pos > before
      }

      while pos < end {
        if nsText.character(at: pos) == UInt16(0x000A) {
          pos += 1
          break
        }
        pos += 1
      }
    }
  }

  /// Adjusts continuation whitespace width in visible-blockquote mode.
  /// Instead of hiding whitespace (which would leave no gap between `>` and content),
  /// applies kern to the last whitespace character on each continuation line so that the
  /// total whitespace width matches `targetWidth` in pixels — compensating for proportional
  /// font differences between space characters and list marker characters.
  private static func kernContinuationWhitespaceInVisibleQuote(
    in range: NSRange,
    targetWidth: CGFloat,
    nsText: NSString,
    textLength: Int,
    style: MarkdownStyle,
    accumulator: inout RenderAccumulator
  ) {
    let end = min(range.location + range.length, textLength)
    var pos = range.location

    // Skip the first line (its whitespace is handled by the list item paragraph style)
    while pos < end {
      if nsText.character(at: pos) == UInt16(0x000A) { pos += 1; break }
      pos += 1
    }

    // Process continuation lines
    while pos < end {
      let lineStart = pos
      // Skip blockquote prefixes (> and space after each)
      var scanPos = skipQuotePrefixes(in: nsText, from: lineStart, limit: end)
      // Find ALL whitespace on this line, skipping embedded > prefixes from
      // nested blockquotes that are separated by list continuation whitespace.
      var totalWsStart = scanPos
      var lastWsEnd = scanPos
      var foundMore = true
      while foundMore {
        let wsStart = scanPos
        while scanPos < end {
          let ch = nsText.character(at: scanPos)
          if ch == UInt16(0x0020) || ch == UInt16(0x0009) {
            scanPos += 1
          } else {
            break
          }
        }
        if scanPos > wsStart {
          lastWsEnd = scanPos
        }
        let before = scanPos
        scanPos = skipQuotePrefixes(in: nsText, from: scanPos, limit: end)
        foundMore = scanPos > before
      }
      if lastWsEnd > totalWsStart {
        // Measure VISIBLE whitespace characters (spans may include > prefixes
        // and hidden spaces between them — skip hidden characters since they
        // have null glyphs with zero advance).
        var totalNaturalWidth: CGFloat = 0
        var lastWsCharIndex = totalWsStart
        var measurePos = totalWsStart
        while measurePos < lastWsEnd {
          let ch = nsText.character(at: measurePos)
          if (ch == UInt16(0x0020) || ch == UInt16(0x0009)),
            !accumulator.hiddenIndexes.contains(measurePos)
          {
            lastWsCharIndex = measurePos
            totalNaturalWidth += style.textWidth(String(Character(UnicodeScalar(ch)!)))
          }
          measurePos += 1
        }
        let kern = targetWidth - totalNaturalWidth
        accumulator.styledRanges.append(
          RenderSpec.StyledRange(
            range: NSRange(location: lastWsCharIndex, length: 1),
            attributes: [.kern: kern]))
      }
      // Advance to the next line
      pos = lineStart
      while pos < end {
        if nsText.character(at: pos) == UInt16(0x000A) { pos += 1; break }
        pos += 1
      }
    }
  }

  /// Finds the last `>` in a code block line's prefix and overrides its kern
  /// to include localInset. Uses the correct glyph width depending on whether
  /// the `>` is space-replaced (cursor outside that blockquote) or visible.
  private static func applyCodeBlockGtKernOverride(
    lineStart: Int,
    prefixLength: Int,
    localInset: CGFloat,
    textLength: Int,
    nsText: NSString,
    style: MarkdownStyle,
    accumulator: inout RenderAccumulator
  ) {
    let prefixEnd = min(lineStart + prefixLength, textLength)
    // Scan the prefix to find the last > character
    var lastGtPos: Int? = nil
    for i in lineStart..<prefixEnd {
      if nsText.character(at: i) == UInt16(0x003E) {
        lastGtPos = i
      }
    }
    guard let gtPos = lastGtPos else { return }
    let gtKernOverride = style.blockquoteIndent + localInset - style.textWidth(">")
    accumulator.styledRanges.append(
      RenderSpec.StyledRange(
        range: NSRange(location: gtPos, length: 1),
        attributes: [.kern: gtKernOverride]))
  }

  private static func skipQuotePrefixes(in nsText: NSString, from start: Int, limit: Int) -> Int {
    var pos = start
    while pos < limit {
      if nsText.character(at: pos) != UInt16(0x003E) { break }
      pos += 1
      if pos < limit, nsText.character(at: pos) == UInt16(0x0020) {
        pos += 1
      }
    }
    return pos
  }

  private static func applyDelimiterVisibility(
    delimiterRanges: [NSRange],
    cursorInNode: Bool,
    textLength: Int,
    style: MarkdownStyle,
    hiddenIndexes: inout IndexSet,
    temporaryAttributes: inout [RenderSpec.StyledRange]
  ) {
    for delimiter in delimiterRanges {
      let safeDelimiter = clamp(delimiter, to: textLength)
      guard safeDelimiter.length > 0 else { continue }
      if cursorInNode {
        temporaryAttributes.append(
          RenderSpec.StyledRange(
            range: safeDelimiter,
            attributes: [.foregroundColor: style.delimiterColor]))
      } else {
        hiddenIndexes.insert(integersIn: safeDelimiter.location..<(safeDelimiter.location + safeDelimiter.length))
      }
    }
  }

  private static func insertHidden(range: NSRange, into hiddenIndexes: inout IndexSet) {
    guard range.length > 0 else { return }
    hiddenIndexes.insert(integersIn: range.location..<(range.location + range.length))
  }

  private static func applyListContinuationParagraphStyles(
    in paragraphRange: NSRange,
    contentIndent: CGFloat,
    font: NSFont,
    color: NSColor,
    paragraphSpacing: CGFloat,
    style: MarkdownStyle,
    nsText: NSString,
    textLength: Int,
    accumulator: inout RenderAccumulator
  ) {
    let safeRange = clamp(paragraphRange, to: textLength)
    guard safeRange.length > 0 else { return }

    let paragraphEnd = safeRange.location + safeRange.length
    var lineStart = nsText.lineRange(for: NSRange(location: safeRange.location, length: 0)).location
    var isFirstLine = true

    while lineStart < paragraphEnd {
      let lineRange = nsText.lineRange(for: NSRange(location: lineStart, length: 0))
      let clampedLine = clamp(lineRange, to: paragraphEnd)
      if !isFirstLine, clampedLine.length > 0 {
        let paragraphStyle = NSMutableParagraphStyle()
        paragraphStyle.firstLineHeadIndent = contentIndent
        paragraphStyle.headIndent = contentIndent
        paragraphStyle.paragraphSpacing = paragraphSpacing
        paragraphStyle.lineHeightMultiple = style.lineHeightMultiple

        accumulator.styledRanges.append(
          RenderSpec.StyledRange(
            range: clampedLine,
            attributes: [
              .font: font,
              .foregroundColor: color,
              .paragraphStyle: paragraphStyle.copy() as! NSParagraphStyle,
            ]))
      }

      isFirstLine = false
      let nextLineStart = lineRange.location + lineRange.length
      if nextLineStart <= lineStart {
        break
      }
      lineStart = nextLineStart
    }
  }

  private static func renderedWidth(
    for range: NSRange?,
    nsText: NSString,
    textLength: Int,
    style: MarkdownStyle
  ) -> CGFloat {
    guard let range else { return 0 }
    let safeRange = clamp(range, to: textLength)
    guard safeRange.length > 0 else { return 0 }
    return style.textWidth(nsText.substring(with: safeRange))
  }

  // MARK: - Blank Line Collapsing

  /// Collapse the first blank line in gaps between sibling blocks so that
  /// `\n\n` paragraph separators render identically to the old single-`\n` behavior.
  /// Subsequent blank lines in the same gap remain visible, preserving the user's
  /// ability to create intentional visual space.
  private static func collapseInterBlockBlankLines(
    between blocks: [MarkdownBlock],
    nsText: NSString,
    textLength: Int,
    accumulator: inout RenderAccumulator
  ) {
    guard blocks.count >= 2 else { return }

    for i in 1..<blocks.count {
      let prevEnd = blocks[i - 1].range.location + blocks[i - 1].range.length
      let nextStart = blocks[i].range.location

      guard nextStart > prevEnd else { continue }

      collapseFirstBlankLine(
        in: NSRange(location: prevEnd, length: nextStart - prevEnd),
        nsText: nsText,
        textLength: textLength,
        accumulator: &accumulator
      )
    }
  }

  /// Find the first blank line (only whitespace and `>` markers) in the given range
  /// and mark its characters as collapsed newlines so the glyph delegate sets them to
  /// `.null`, causing the line fragment to collapse to zero height.
  private static func collapseFirstBlankLine(
    in range: NSRange,
    nsText: NSString,
    textLength: Int,
    accumulator: inout RenderAccumulator
  ) {
    let end = min(range.location + range.length, textLength)
    var pos = range.location

    while pos < end {
      let lineRange = nsText.lineRange(for: NSRange(location: pos, length: 0))

      // Only consider lines that start at or after the gap start.
      // Lines extending before the gap are content lines, not blank lines.
      if lineRange.location >= range.location && lineRange.length > 0 {
        let lineText = nsText.substring(with: lineRange)
        let stripped = lineText
          .replacingOccurrences(of: ">", with: "")
          .trimmingCharacters(in: .whitespacesAndNewlines)

        if stripped.isEmpty {
          // Blank line — mark all characters as collapsed so the line fragment
          // has no visible glyphs and collapses to zero height.
          for i in lineRange.location..<(lineRange.location + lineRange.length) {
            accumulator.collapsedNewlineIndexes.insert(i)
          }
          return  // Only collapse the first blank line in each gap.
        }
      }

      let nextPos = lineRange.location + lineRange.length
      if nextPos <= pos { break }
      pos = nextPos
    }
  }

  static func cursorOverlaps(
    _ cursor: NSRange,
    node: NSRange,
    textLength: Int
  ) -> Bool {
    let cursorEnd = cursor.location + cursor.length
    let nodeEnd = node.location + node.length
    if cursor.location < nodeEnd && cursorEnd > node.location {
      return true
    }
    if cursor.length == 0 {
      if cursor.location == node.location || cursor.location == nodeEnd {
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
