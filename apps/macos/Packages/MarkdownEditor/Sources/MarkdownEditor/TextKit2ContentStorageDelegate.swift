import AppKit

/// `NSTextContentStorageDelegate` that produces *display* paragraphs whose
/// `attributedString` reflects the spec's hidden / bullet / checkbox index
/// sets — the TextKit 2 equivalent of the TextKit 1 glyph-hiding /
/// glyph-substitution mechanism.
///
/// **Length-matching invariant.** For every paragraph the delegate vends,
/// `displayString.length == sourceRange.length`. TK2's `NSTextLocation`
/// model assumes display offsets are source offsets: hit-test, character
/// navigation, rendering attributes, and the cursor's visual position all
/// silently break when display and source lengths diverge. Preserving the
/// invariant via substitution (rather than character removal) keeps every
/// other layer of the TK2 stack honest without per-paragraph translation
/// maps.
///
/// Substitutions:
/// - **Hidden chars** (`hiddenIndexes`) → `U+200B ZERO WIDTH SPACE`. 1-for-1
///   length, zero visual width, valid cursor landing position.
/// - **Bullet marker** (`bulletIndexes`, source `-`) → `U+2022 •`. 1-for-1.
/// - **Unchecked checkbox** (`uncheckedCheckboxIndexes`, source `[ ]`) →
///   `☐` + `U+200B` + `U+200B`. Length 3, only the `☐` is visible.
/// - **Checked checkbox** (`checkedCheckboxIndexes`, source `[x]`) →
///   `☒` + `U+200B` + `U+200B`. Length 3.
/// - **All other source chars** pass through unchanged.
///
/// The trailing `\n` of the paragraph (the paragraph separator) stays a
/// real `\n`. The single exception to the length-matching invariant is
/// paragraphs whose entire source range is in `hiddenIndexes` — those are
/// hidden via the `NSTextContentManagerDelegate.shouldEnumerate` hook
/// below, never vended at all.
///
/// `lineBreakIndexes` (soft / hard breaks identified by the AST) is NOT
/// handled at this layer. The renderer instead emits per-line paragraph
/// styles with `paragraphSpacing = 0` so soft-break-coupled source
/// paragraphs render flush against each other — preserving 1:1 source ↔
/// `NSTextParagraph` element correspondence and TK2's natural cursor
/// navigation, which an earlier U+2028-coalescing experiment broke.
///
/// Coordinates: index sets are in **document** offsets (relative to the full
/// markdown source). The delegate projects them onto each paragraph's
/// `paragraphContentRange` when constructing display paragraphs.
@MainActor
final class TextKit2ContentStorageDelegate: NSObject,
  @MainActor NSTextContentStorageDelegate,
  @MainActor NSTextContentManagerDelegate
{

  // MARK: - Spec inputs (written by TextKit2RenderApplicator)

  var hiddenIndexes: IndexSet = IndexSet()
  var bulletIndexes: IndexSet = IndexSet()
  var uncheckedCheckboxIndexes: IndexSet = IndexSet()
  var checkedCheckboxIndexes: IndexSet = IndexSet()
  /// Source offsets of `\n` characters that the AST classifies as soft /
  /// hard line breaks (i.e. mid-AST-paragraph). Currently only used by the
  /// renderer for paragraph-spacing decisions; this delegate ignores it.
  /// Kept here so the spec-write path in `TextKit2RenderApplicator.apply`
  /// has a stable target.
  var lineBreakIndexes: IndexSet = IndexSet()

  /// Phase 2 bridging-layer: block-level renderer specs. The first
  /// paragraph of every spec range is vended as a length-matched display
  /// string `[U+FFFC attachment][ZWSP × (length - 1)][\n]`. Sibling
  /// paragraphs (those whose source range is wholly inside the spec range
  /// but does not start at it) are hidden via `shouldEnumerate` so the
  /// attachment view covers their visual region.
  ///
  /// Weak-coupled to `BlockRendererRegistry`: the attachment carries a
  /// back-reference to the host the registry's reconciliation built for
  /// the same range.
  var blockRendererSpecs: [BlockRendererSpec] = []

  /// Weak link to the text view used for resolving hosts. Set by the
  /// applicator on every `apply()`. The text view owns the registry
  /// entries keyed by its identity.
  weak var textView: NSTextView?

  // MARK: - Substitution glyphs

  private static let bulletString = "\u{2022}"  // •
  private static let uncheckedCheckboxString = "\u{25A1}"  // □
  private static let checkedCheckboxString = "\u{2612}"  // ☒
  /// Used as a length-preserving stand-in for hidden source chars and as
  /// padding after multi-char glyph substitutions (checkboxes). Zero
  /// rendered width but a valid cursor landing position — so TK2's display
  /// offsets stay 1:1 with source offsets.
  private static let zeroWidthSpace = "\u{200B}"

  /// Diagnostic: incremented each time the delegate's paragraph hook is
  /// invoked. Used by tests to verify rebuilds happen after spec changes.
  private(set) var paragraphBuildCount: Int = 0

  // MARK: - NSTextContentStorageDelegate

  func textContentStorage(
    _ textContentStorage: NSTextContentStorage,
    textParagraphWith range: NSRange
  ) -> NSTextParagraph? {
    paragraphBuildCount += 1
    guard let attr = textContentStorage.attributedString else { return nil }
    guard NSMaxRange(range) <= attr.length else { return nil }

    // Phase 2 bridging-layer: if this paragraph is the FIRST paragraph of a
    // block-renderer spec range, vend an attachment-bearing display string.
    // The first paragraph hosts the U+FFFC attachment glyph; subsequent
    // paragraphs in the same spec range are hidden from layout via the
    // `shouldEnumerate` hook below — the attachment view covers their
    // visual region. Length-matching invariant holds: the attachment
    // paragraph's display string is exactly `range.length` UTF-16 units
    // (one U+FFFC + (range.length - 2) ZWSPs + one `\n`, or analogous
    // when there's no trailing newline).
    if let spec = blockRendererSpecs.first(where: { $0.range.location == range.location }) {
      if let p = buildAttachmentParagraph(
        sourceRange: range, spec: spec, attr: attr)
      {
        return p
      }
    }

    let source = attr.attributedSubstring(from: range)

    // Identify the trailing paragraph separator (a `\n`) so we can guarantee
    // it survives into the display string. NSTextParagraph requires a valid
    // paragraphSeparatorRange — dropping the trailing `\n` causes
    // `setParagraphSeparatorRange:` to crash with an out-of-bounds range.
    let srcNS = source.string as NSString
    let endsWithNewline =
      srcNS.length > 0 && srcNS.character(at: srcNS.length - 1) == UInt16(0x0A)
    let scanLength = endsWithNewline ? range.length - 1 : range.length

    let display = NSMutableAttributedString()

    // Walk one source char at a time, emitting a substitution that has the
    // SAME UTF-16 length as the source span it replaces. Multi-source-char
    // substitutions (checkboxes) consume the next `padding` chars after the
    // visible glyph and emit ZWSPs for them so total length stays equal.
    var i = 0
    while i < scanLength {
      let docIdx = range.location + i
      let oneChar = source.attributedSubstring(from: NSRange(location: i, length: 1))
      let attrs = oneChar.attributes(at: 0, effectiveRange: nil)

      if hiddenIndexes.contains(docIdx) {
        // Length-preserving stand-in. Carry the source char's attributes so
        // any layout-fragment lookups (background, paragraph style) see the
        // same per-char metadata they would have seen before substitution.
        display.append(NSAttributedString(string: Self.zeroWidthSpace, attributes: attrs))
      } else if bulletIndexes.contains(docIdx) {
        display.append(NSAttributedString(string: Self.bulletString, attributes: attrs))
      } else if uncheckedCheckboxIndexes.contains(docIdx) {
        display.append(
          NSAttributedString(string: Self.uncheckedCheckboxString, attributes: attrs))
        // Pad the remaining 2 source chars (`[ ]` is 3 chars total) with
        // ZWSPs so total display length matches the 3-char source span.
        let padCount = min(2, scanLength - i - 1)
        for j in 0..<padCount {
          let padIdx = i + 1 + j
          let padChar = source.attributedSubstring(
            from: NSRange(location: padIdx, length: 1))
          let padAttrs = padChar.attributes(at: 0, effectiveRange: nil)
          display.append(
            NSAttributedString(string: Self.zeroWidthSpace, attributes: padAttrs))
        }
        i += padCount  // advance past consumed padding chars
      } else if checkedCheckboxIndexes.contains(docIdx) {
        display.append(
          NSAttributedString(string: Self.checkedCheckboxString, attributes: attrs))
        let padCount = min(2, scanLength - i - 1)
        for j in 0..<padCount {
          let padIdx = i + 1 + j
          let padChar = source.attributedSubstring(
            from: NSRange(location: padIdx, length: 1))
          let padAttrs = padChar.attributes(at: 0, effectiveRange: nil)
          display.append(
            NSAttributedString(string: Self.zeroWidthSpace, attributes: padAttrs))
        }
        i += padCount
      } else {
        display.append(oneChar)
      }
      i += 1
    }

    if endsWithNewline {
      let nlChar = source.attributedSubstring(
        from: NSRange(location: range.length - 1, length: 1))
      display.append(nlChar)
    }

    return NSTextParagraph(attributedString: display)
  }

  /// Build the attachment-bearing display paragraph for the FIRST paragraph
  /// of a block-renderer spec range. The display string layout is:
  ///
  ///     [U+FFFC attachment] [ZWSP × (range.length - 1 - trailingNewline?1:0)] [\n]?
  ///
  /// Total UTF-16 length equals `range.length`, preserving the length-
  /// matching invariant for this paragraph.
  ///
  /// The attachment carries a back-reference to the host (resolved via
  /// `BlockRendererRegistry.shared.host(for:atSourceOffset:)`), which the
  /// view provider uses on `loadView()` to vend the renderer's view.
  /// Returns `nil` if no live host exists for the spec — in which case the
  /// caller falls through to the regular per-character substitution path.
  private func buildAttachmentParagraph(
    sourceRange range: NSRange,
    spec: BlockRendererSpec,
    attr: NSAttributedString
  ) -> NSTextParagraph? {
    guard let textView,
      let host = BlockRendererRegistry.shared.host(
        for: textView, atSourceOffset: spec.range.location)
    else { return nil }

    let source = attr.attributedSubstring(from: range)
    let srcNS = source.string as NSString
    let endsWithNewline =
      srcNS.length > 0 && srcNS.character(at: srcNS.length - 1) == UInt16(0x0A)
    let firstAttrs = source.attributes(at: 0, effectiveRange: nil)

    // Build the attachment-glyph attributes: copy the source paragraph's
    // existing attributes (head indent, font, color, etc.) but override the
    // paragraph style to pin the line containing the U+FFFC at exactly
    // `spec.reservedHeight` points tall.
    //
    // Why this is necessary: TK2 does NOT auto-grow the line containing an
    // attachment to match `attachmentBounds.height`. Without the
    // line-height pin, the line stays at the code font's natural height and
    // the attachment view's frame extends UPWARD over preceding paragraphs
    // (the attachment is anchored at the natural line's baseline). Forcing
    // `minimumLineHeight == maximumLineHeight == reservedHeight` makes TK2
    // reserve the full vertical region inside the line itself, so siblings
    // above the block don't get overlapped.
    let attachmentParaStyle: NSParagraphStyle = {
      let base = (firstAttrs[.paragraphStyle] as? NSParagraphStyle) ?? .default
      let mutable = (base.mutableCopy() as? NSMutableParagraphStyle) ?? NSMutableParagraphStyle()
      mutable.minimumLineHeight = spec.reservedHeight
      mutable.maximumLineHeight = spec.reservedHeight
      return mutable
    }()
    var attachmentAttrs = firstAttrs
    attachmentAttrs[.paragraphStyle] = attachmentParaStyle

    let display = NSMutableAttributedString()
    // Use the host's cached attachment so view-provider identity is stable
    // across the inevitable re-vends (every selection / edit triggers a
    // paragraph rebuild). A fresh `BlockAttachment` here would make AppKit
    // drop the embedded view and substitute the default placeholder icon.
    let attachment = host.ensureAttachment()
    display.append(NSAttributedString(attachment: attachment))
    // Apply the line-height-overriding attributes to the U+FFFC glyph.
    display.addAttributes(attachmentAttrs, range: NSRange(location: 0, length: 1))

    let padCount = max(0, range.length - 1 - (endsWithNewline ? 1 : 0))
    if padCount > 0 {
      display.append(
        NSAttributedString(
          string: String(repeating: Self.zeroWidthSpace, count: padCount),
          attributes: firstAttrs))
    }

    if endsWithNewline {
      // Pin the trailing newline to the same line height too — otherwise the
      // paragraph separator can establish a smaller line above/below the
      // attachment region and the layout fragment's reported height drifts
      // off `reservedHeight`.
      let nlChar = source.attributedSubstring(
        from: NSRange(location: srcNS.length - 1, length: 1))
      let nlMutable = NSMutableAttributedString(attributedString: nlChar)
      nlMutable.addAttribute(
        .paragraphStyle,
        value: attachmentParaStyle,
        range: NSRange(location: 0, length: nlMutable.length))
      display.append(nlMutable)
    }

    return NSTextParagraph(attributedString: display)
  }

  // MARK: - NSTextContentManagerDelegate

  /// Hide source paragraphs whose entire content has been absorbed into
  /// `hiddenIndexes` by the renderer's inter-block-gap logic. Without this
  /// hook the absorbed `\n`-only paragraphs still take a visible line of
  /// space because TK2 preserves their trailing newline as the paragraph
  /// separator. Returning `false` here tells `enumerateTextElements` to
  /// skip the element entirely — it contributes no layout.
  ///
  /// Also hides "sibling" paragraphs of a block-renderer spec range — i.e.
  /// paragraphs whose source range falls inside a spec range but does NOT
  /// start at the spec's `range.location`. The first paragraph of the
  /// range carries the U+FFFC attachment whose view covers the whole
  /// block's visual region; the siblings would otherwise contribute their
  /// own lines below it. This is the same "fully hidden via shouldEnumerate"
  /// exception to the length-matching invariant the inter-block-gap
  /// absorption uses.
  ///
  /// Forward / backward arrow-key motion still strides past the absorbed
  /// source chars because they're in `hiddenIndexes` (or, for sibling
  /// paragraphs, the move overrides walk source positions and the
  /// attachment paragraph's vended display covers the first source
  /// paragraph's range only — Phase 2.2 adds the in-block selection-snap).
  func textContentManager(
    _ textContentManager: NSTextContentManager,
    shouldEnumerate textElement: NSTextElement,
    options: NSTextContentManager.EnumerationOptions = []
  ) -> Bool {
    guard let storage = textContentManager as? NSTextContentStorage,
      let elementRange = textElement.elementRange
    else { return true }
    let docStart = storage.documentRange.location
    let elementOffset = storage.offset(from: docStart, to: elementRange.location)
    let elementLength = storage.offset(
      from: elementRange.location, to: elementRange.endLocation)
    guard elementLength > 0 else { return true }

    // Sibling-paragraph hide for block-renderer specs: the FIRST paragraph
    // of a spec range carries the attachment and should be enumerated;
    // every subsequent paragraph inside the same range is hidden.
    for spec in blockRendererSpecs {
      if elementOffset > spec.range.location
        && elementOffset < spec.range.location + spec.range.length
      {
        return false
      }
    }

    for i in 0..<elementLength {
      if !hiddenIndexes.contains(elementOffset + i) {
        return true
      }
    }
    return false
  }
}
