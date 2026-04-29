import AppKit

/// Phase 1 of the TextKit 2 migration: applies the subset of `RenderSpec` that
/// requires no custom layout manager, no glyph hiding, no custom fragments, and
/// no rendering attributes. Equivalent to `RenderApplicator` for body-text-only
/// rendering.
///
/// Deferred features (each marked TODO Phase N inline):
/// - Phase 2 (NSTextContentStorage delegate): hiddenIndexes, bulletIndexes,
///   uncheckedCheckboxIndexes, checkedCheckboxIndexes, collapsedNewlineIndexes
/// - Phase 3 (custom NSTextLayoutFragment subclasses): codeBlockCharacterRanges,
///   blockquoteCharacterRanges
/// - Phase 4 (NSTextLayoutManager.setRenderingAttributes): temporaryAttributes
@MainActor
enum TextKit2RenderApplicator {

  static func apply(_ spec: RenderSpec, to textView: NSTextView) {
    guard let textStorage = textView.textStorage else { return }

    let textLength = (textView.string as NSString).length
    guard textLength > 0 else { return }
    let fullRange = NSRange(location: 0, length: textLength)

    // TODO Phase 2: write hidden / bullet / checkbox / collapsedNewline index
    // sets into the NSTextContentStorage delegate that produces display
    // paragraphs from source ranges.

    // TODO Phase 3: write codeBlockCharacterRanges and blockquoteCharacterRanges
    // into the NSTextLayoutManager delegate that vends custom
    // NSTextLayoutFragment subclasses for those paragraphs.

    // Save scroll position — full-range attribute reset triggers layout
    // invalidation that can momentarily displace the scroll origin.
    let clipView = textView.enclosingScrollView?.contentView
    let savedOrigin = clipView?.bounds.origin

    textStorage.beginEditing()
    textStorage.setAttributes(spec.baseAttributes, range: fullRange)

    for styled in spec.styledRanges {
      textStorage.addAttributes(styled.attributes, range: styled.range)
    }

    for traitApp in spec.fontTraits {
      applyFontTrait(traitApp.trait, to: textStorage, in: traitApp.range)
    }

    textStorage.endEditing()

    if let origin = savedOrigin, let clipView {
      clipView.setBoundsOrigin(origin)
    }

    // TODO Phase 4: apply spec.temporaryAttributes via
    // NSTextLayoutManager.setRenderingAttributes(_:for:) — the cursor-driven
    // delimiter coloring path. Spike 3 confirmed the imperative setter is the
    // correct API on AppKit (the validator closure is one-shot per fragment).
  }

  /// Cursor-only update. In Phase 1 this is a near-no-op because every effect
  /// driven by cursor movement (delimiter visibility, glyph hiding, marker
  /// substitution) lives in deferred phases. Kept as a symmetric API to the
  /// TK1 applicator so the Coordinator branches stay simple.
  static func applyCursorUpdate(
    _ spec: RenderSpec,
    previousHidden: IndexSet,
    previousBullets: IndexSet,
    previousUncheckedCheckboxes: IndexSet = IndexSet(),
    previousCheckedCheckboxes: IndexSet = IndexSet(),
    previousCollapsedNewlines: IndexSet = IndexSet(),
    to textView: NSTextView
  ) {
    // TODO Phase 2: invalidate display paragraphs whose hidden / marker state
    // changed, so the content-storage delegate is re-consulted.
    // TODO Phase 4: refresh rendering attributes for delimiter coloring.
    _ = (
      spec, previousHidden, previousBullets,
      previousUncheckedCheckboxes, previousCheckedCheckboxes,
      previousCollapsedNewlines, textView
    )
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
