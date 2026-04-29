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

    // Phase 2: write the hiding/substitution index sets into the content-
    // storage delegate. The delegate is consulted again by TextKit 2 when
    // the textStorage edit below triggers paragraph rebuilds.
    if let delegate = textView.textContentStorage?.delegate
      as? TextKit2ContentStorageDelegate
    {
      delegate.hiddenIndexes = spec.hiddenIndexes
      delegate.bulletIndexes = spec.bulletIndexes
      delegate.uncheckedCheckboxIndexes = spec.uncheckedCheckboxIndexes
      delegate.checkedCheckboxIndexes = spec.checkedCheckboxIndexes
      delegate.collapsedNewlineIndexes = spec.collapsedNewlineIndexes
    }

    // Phase 3: write the per-paragraph decoration ranges (code block bg,
    // blockquote left borders) into the layout-manager delegate. The
    // delegate vends a `TextKit2LayoutFragment` per paragraph configured
    // with the matching decorations; the actual painting happens in the
    // fragment's `draw(at:in:)` override.
    if let layoutDelegate = textView.textLayoutManager?.delegate
      as? TextKit2LayoutManagerDelegate
    {
      layoutDelegate.codeBlockCharacterRanges = spec.codeBlockCharacterRanges
      layoutDelegate.blockquoteCharacterRanges = spec.blockquoteCharacterRanges
      if let containerWidth = textView.textLayoutManager?.textContainer?.size.width,
        containerWidth > 0,
        containerWidth < CGFloat.greatestFiniteMagnitude
      {
        layoutDelegate.containerWidth = containerWidth
      }
    }

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

    // Force NSTextContentStorage to re-fetch paragraphs from the delegate.
    // Attribute-only storage edits (above) don't invalidate the paragraph
    // cache, so we explicitly record an edit action covering the full
    // document range. This is the TK2 equivalent of TK1's cursor-update
    // glyph-invalidation loop and is what makes the content delegate
    // re-consulted when its index sets change without text content changing.
    if let contentStorage = textView.textContentStorage {
      contentStorage.performEditingTransaction {
        let docRange = contentStorage.documentRange
        contentStorage.recordEditAction(in: docRange, newTextRange: docRange)
      }
    }
    if let tlm = textView.textLayoutManager {
      tlm.invalidateLayout(for: tlm.documentRange)
    }

    if let origin = savedOrigin, let clipView {
      clipView.setBoundsOrigin(origin)
    }

    // TODO Phase 4: apply spec.temporaryAttributes via
    // NSTextLayoutManager.setRenderingAttributes(_:for:) — the cursor-driven
    // delimiter coloring path. Spike 3 confirmed the imperative setter is the
    // correct API on AppKit (the validator closure is one-shot per fragment).
  }

  /// Cursor-only update. The TK1 applicator avoids re-touching all attributes
  /// on cursor moves because temporary attributes set on the layout manager
  /// must survive the update. The TK2 path doesn't have that constraint until
  /// Phase 4 introduces rendering attributes, so for now we re-run the full
  /// `apply` to refresh the delegate state and trigger paragraph rebuild.
  /// Phase 4 will optimize this to invalidate only the changed paragraphs
  /// and update rendering attributes incrementally.
  static func applyCursorUpdate(
    _ spec: RenderSpec,
    previousHidden: IndexSet,
    previousBullets: IndexSet,
    previousUncheckedCheckboxes: IndexSet = IndexSet(),
    previousCheckedCheckboxes: IndexSet = IndexSet(),
    previousCollapsedNewlines: IndexSet = IndexSet(),
    to textView: NSTextView
  ) {
    _ = (
      previousHidden, previousBullets,
      previousUncheckedCheckboxes, previousCheckedCheckboxes,
      previousCollapsedNewlines
    )
    apply(spec, to: textView)
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
