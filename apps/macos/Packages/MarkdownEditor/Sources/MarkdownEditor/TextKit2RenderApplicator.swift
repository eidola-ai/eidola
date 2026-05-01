import AppKit

/// TextKit 2 renderer-side effect surface. Mirrors `RenderApplicator` for the
/// TK2 path; consumes a `RenderSpec` and writes it through three subsystems:
///
/// - Text-storage attributes (base + styled ranges + font traits) — same as
///   TK1, modifies `NSTextStorage` directly.
/// - Content-storage delegate (`TextKit2ContentStorageDelegate`) — receives
///   the hide / bullet / checkbox index sets and rebuilds display paragraphs.
/// - Layout-manager delegate (`TextKit2LayoutManagerDelegate`) — receives the
///   code-block / blockquote decoration ranges and vends configured
///   `TextKit2LayoutFragment` instances.
/// - Layout-manager rendering attributes (`setRenderingAttributes(_:for:)`)
///   — receives `temporaryAttributes` for cursor-driven delimiter coloring.
@MainActor
enum TextKit2RenderApplicator {

  static func apply(_ spec: RenderSpec, to textView: NSTextView) {
    guard let textStorage = textView.textStorage else { return }

    let textLength = (textView.string as NSString).length
    guard textLength > 0 else { return }
    let fullRange = NSRange(location: 0, length: textLength)

    let preApplySel = textView.selectedRange()
    DebugTrace.log("apply.start", [
      "text_length": textLength,
      "hidden_count": spec.hiddenIndexes.count,
      "line_break_count": spec.lineBreakIndexes.count,
      "styled_ranges": spec.styledRanges.count,
      "temp_attrs": spec.temporaryAttributes.count,
      "selection_location": preApplySel.location,
      "selection_length": preApplySel.length,
    ])

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
      delegate.lineBreakIndexes = spec.lineBreakIndexes
      delegate.blockRendererSpecs = spec.blockRendererSpecs
      delegate.textView = textView
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
      layoutDelegate.blockRendererSpecs = spec.blockRendererSpecs
      if let containerWidth = textView.textLayoutManager?.textContainer?.size.width,
        containerWidth > 0,
        containerWidth < CGFloat.greatestFiniteMagnitude
      {
        layoutDelegate.containerWidth = containerWidth
      }
    }

    // Phase 2 bridging-layer: reconcile the registry's live host table for
    // this text view against the new spec list. New ranges get fresh hosts
    // built from the registered factory; retired ranges have their hosts
    // disposed; existing hosts get `updateSpec(_:)`. This MUST happen
    // before the content-delegate paragraph rebuild below so that when the
    // delegate vends an attachment-paragraph, the corresponding host
    // exists in the registry and the attachment can resolve its view via
    // `BlockAttachment.viewProvider(for:...)`.
    BlockRendererRegistry.shared.reconcile(
      for: textView, specs: spec.blockRendererSpecs)

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
      let preInvalidateSel = textView.selectedRange()
      tlm.invalidateLayout(for: tlm.documentRange)
      let postInvalidateSel = textView.selectedRange()
      if preInvalidateSel.location != postInvalidateSel.location
        || preInvalidateSel.length != postInvalidateSel.length
      {
        DebugTrace.log("apply.invalidate.selection_shift", [
          "before_location": preInvalidateSel.location,
          "before_length": preInvalidateSel.length,
          "after_location": postInvalidateSel.location,
          "after_length": postInvalidateSel.length,
        ])
      }

      // Force eager full-document layout so every fragment reaches
      // `.layoutAvailable` before any subsequent user interaction. TK2's
      // lazy layout otherwise leaves off-viewport paragraphs at
      // `.estimatedUsageBounds`, where their height is a coarse font-metric
      // estimate. When a cursor-driven interaction (click, arrow-key
      // scroll-to-cursor, word/line jump) later forces layout for those
      // fragments, the realized heights replace the estimates and
      // `usageBoundsForTextContainer.height` snaps to the new value. The
      // enclosing `NSScrollView` reads that as a document-size change and
      // jumps the scroller thumb — the user-visible "scrollbar jump" bug.
      //
      // Cost: bounded by document size. Markdown documents the editor
      // handles are small enough (kilobytes, not megabytes) that the
      // full-document layout is well under one frame and not noticeable.
      // If a future use-case needs streaming-render of long-form content,
      // narrow this to "only the laid-out region plus a buffer" — but
      // every cursor move already runs `apply` so the cost has to stay
      // proportional to viewport, not document.
      let preBoundsHeight = tlm.usageBoundsForTextContainer.height
      tlm.ensureLayout(for: tlm.documentRange)
      let postBoundsHeight = tlm.usageBoundsForTextContainer.height
      if abs(postBoundsHeight - preBoundsHeight) > 0.5 {
        DebugTrace.log("apply.ensureLayout.bounds_changed", [
          "before_height": preBoundsHeight,
          "after_height": postBoundsHeight,
        ])
      }
    }

    if let origin = savedOrigin, let clipView {
      clipView.setBoundsOrigin(origin)
    }

    // Phase 4: apply rendering-only attributes (delimiter dimming when the
    // cursor is inside a markdown construct). Spike 3 found that AppKit's
    // `renderingAttributesValidator` closure is one-shot per fragment and
    // selection changes do not refire it; the imperative
    // `setRenderingAttributes(_:for:)` driven from this path is the correct
    // replacement for TK1's `addTemporaryAttributes`.
    //
    // We clear `.foregroundColor` over the whole document first so attrs from
    // a prior render whose ranges are no longer in the spec don't linger.
    // (`setRenderingAttributes` overwrites within the new range only.)
    if let tlm = textView.textLayoutManager,
      let storage = textView.textContentStorage
    {
      tlm.removeRenderingAttribute(.foregroundColor, for: tlm.documentRange)
      for tempAttr in spec.temporaryAttributes {
        guard
          let location = storage.location(
            storage.documentRange.location, offsetBy: tempAttr.range.location),
          let end = storage.location(location, offsetBy: tempAttr.range.length),
          let textRange = NSTextRange(location: location, end: end)
        else { continue }
        tlm.setRenderingAttributes(tempAttr.attributes, for: textRange)
      }
    }

    let postApplySel = textView.selectedRange()
    DebugTrace.log("apply.end", [
      "selection_location": postApplySel.location,
      "selection_length": postApplySel.length,
    ])
  }

  /// Cursor-only update. Re-runs the full `apply` so that index sets, fragment
  /// configuration, and rendering attributes all refresh together. The TK1
  /// applicator splits this path for performance (avoid full attribute reset
  /// on every cursor move), but on TK2 the full path is cheap enough — Spike 3
  /// measured ~62µs per cursor move with imperative rendering-attribute writes,
  /// well under a frame. A future optimization could narrow rebuild to only
  /// paragraphs whose hidden / temp-attr state changed; not warranted yet.
  static func applyCursorUpdate(
    _ spec: RenderSpec,
    previousHidden: IndexSet,
    previousBullets: IndexSet,
    previousUncheckedCheckboxes: IndexSet = IndexSet(),
    previousCheckedCheckboxes: IndexSet = IndexSet(),
    previousLineBreaks: IndexSet = IndexSet(),
    to textView: NSTextView
  ) {
    _ = (
      previousHidden, previousBullets,
      previousUncheckedCheckboxes, previousCheckedCheckboxes,
      previousLineBreaks
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
