import AppKit

/// Applies a `RenderSpec` to an `NSTextView`. Stateless — holds no styling data.
@MainActor
enum RenderApplicator {

  /// Full application: reset all attributes and apply the complete spec.
  /// Used after text changes and on initial load.
  static func apply(_ spec: RenderSpec, to textView: NSTextView) {
    guard let textStorage = textView.textStorage,
      let layoutManager = textView.layoutManager
    else { return }

    let textLength = (textView.string as NSString).length
    guard textLength > 0 else { return }
    let fullRange = NSRange(location: 0, length: textLength)

    // Set glyph state BEFORE text storage edit so that glyph generation
    // triggered by endEditing() uses the correct hidden/bullet state.
    // This prevents the visible flash/jitter on keystroke.
    if let glyphDelegate = layoutManager.delegate as? GlyphHidingLayoutManagerDelegate {
      glyphDelegate.hiddenCharacterIndexes = spec.hiddenIndexes
      glyphDelegate.bulletCharacterIndexes = spec.bulletIndexes
      glyphDelegate.uncheckedCheckboxCharacterIndexes = spec.uncheckedCheckboxIndexes
      glyphDelegate.checkedCheckboxCharacterIndexes = spec.checkedCheckboxIndexes
    }

    // Set code block ranges for full-width background drawing.
    if let codeBlockLM = layoutManager as? CodeBlockBackgroundLayoutManager {
      codeBlockLM.codeBlockCharacterRanges = spec.codeBlockCharacterRanges
      codeBlockLM.blockquoteCharacterRanges = spec.blockquoteCharacterRanges
    }

    // Save scroll position — the full-range attribute reset below triggers
    // layout invalidation which can momentarily displace the scroll origin.
    let clipView = textView.enclosingScrollView?.contentView
    let savedOrigin = clipView?.bounds.origin

    // Apply stored attributes. endEditing() coalesces the attribute-change
    // notifications and fires layout invalidation for all affected ranges.
    // No explicit invalidateGlyphs/invalidateLayout is needed afterward —
    // setAttributes(fullRange) already marks the entire document as changed,
    // so endEditing() invalidates the full range. A redundant second
    // invalidation pass would cause a visible scroll shudder in long documents.
    textStorage.beginEditing()
    textStorage.setAttributes(spec.baseAttributes, range: fullRange)

    for styled in spec.styledRanges {
      textStorage.addAttributes(styled.attributes, range: styled.range)
    }

    for traitApp in spec.fontTraits {
      applyFontTrait(traitApp.trait, to: textStorage, in: traitApp.range)
    }

    textStorage.endEditing()

    // Restore scroll position in case the attribute reset displaced it.
    if let origin = savedOrigin, let clipView {
      clipView.setBoundsOrigin(origin)
    }

    // Apply rendering-only attributes (delimiter colors when cursor is inside)
    layoutManager.removeTemporaryAttribute(.foregroundColor, forCharacterRange: fullRange)
    for tempAttr in spec.temporaryAttributes {
      layoutManager.addTemporaryAttributes(
        tempAttr.attributes, forCharacterRange: tempAttr.range)
    }
  }

  /// Cursor-only update: applies only glyph and temporary attribute changes.
  /// More efficient than full `apply` — skips text storage attribute reset.
  static func applyCursorUpdate(
    _ spec: RenderSpec,
    previousHidden: IndexSet,
    previousBullets: IndexSet,
    previousUncheckedCheckboxes: IndexSet = IndexSet(),
    previousCheckedCheckboxes: IndexSet = IndexSet(),
    to textView: NSTextView
  ) {
    guard let layoutManager = textView.layoutManager else { return }
    let textLength = (textView.string as NSString).length
    guard textLength > 0 else { return }
    let fullRange = NSRange(location: 0, length: textLength)

    // Update glyph delegate
    if let glyphDelegate = layoutManager.delegate as? GlyphHidingLayoutManagerDelegate {
      glyphDelegate.hiddenCharacterIndexes = spec.hiddenIndexes
      glyphDelegate.bulletCharacterIndexes = spec.bulletIndexes
      glyphDelegate.uncheckedCheckboxCharacterIndexes = spec.uncheckedCheckboxIndexes
      glyphDelegate.checkedCheckboxCharacterIndexes = spec.checkedCheckboxIndexes
    }

    // Update code block ranges for full-width background drawing.
    if let codeBlockLM = layoutManager as? CodeBlockBackgroundLayoutManager {
      codeBlockLM.codeBlockCharacterRanges = spec.codeBlockCharacterRanges
      codeBlockLM.blockquoteCharacterRanges = spec.blockquoteCharacterRanges
    }

    // Save scroll position — glyph invalidation for ranges above the
    // viewport can change layout geometry, displacing the scroll origin.
    let clipView = textView.enclosingScrollView?.contentView
    let savedOrigin = clipView?.bounds.origin

    // Invalidate only the ranges that changed
    let allPrevious = previousHidden.union(previousBullets)
      .union(previousUncheckedCheckboxes).union(previousCheckedCheckboxes)
    let allNew = spec.hiddenIndexes.union(spec.bulletIndexes)
      .union(spec.uncheckedCheckboxIndexes).union(spec.checkedCheckboxIndexes)
    let changed = allPrevious.symmetricDifference(allNew)

    for range in changed.rangeView {
      let nsRange = NSRange(location: range.lowerBound, length: range.count)
      layoutManager.invalidateGlyphs(
        forCharacterRange: nsRange, changeInLength: 0, actualCharacterRange: nil)
      layoutManager.invalidateLayout(
        forCharacterRange: nsRange, actualCharacterRange: nil)
    }

    // Restore scroll position.
    if let origin = savedOrigin, let clipView {
      clipView.setBoundsOrigin(origin)
    }

    // Update temporary attributes
    layoutManager.removeTemporaryAttribute(.foregroundColor, forCharacterRange: fullRange)
    for tempAttr in spec.temporaryAttributes {
      layoutManager.addTemporaryAttributes(
        tempAttr.attributes, forCharacterRange: tempAttr.range)
    }
  }

  // MARK: - Private

  private static func applyFontTrait(
    _ trait: NSFontTraitMask, to storage: NSTextStorage, in range: NSRange
  ) {
    guard range.length > 0 else { return }
    storage.enumerateAttribute(.font, in: range, options: []) { value, attrRange, _ in
      if let font = value as? NSFont {
        let newFont = NSFontManager.shared.convert(font, toHaveTrait: trait)
        storage.addAttribute(.font, value: newFont, range: attrRange)
      }
    }
  }
}
