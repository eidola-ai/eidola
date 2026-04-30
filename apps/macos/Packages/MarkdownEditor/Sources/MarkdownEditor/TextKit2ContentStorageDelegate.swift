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

  // MARK: - NSTextContentManagerDelegate

  /// Hide source paragraphs whose entire content has been absorbed into
  /// `hiddenIndexes` by the renderer's inter-block-gap logic. Without this
  /// hook the absorbed `\n`-only paragraphs still take a visible line of
  /// space because TK2 preserves their trailing newline as the paragraph
  /// separator. Returning `false` here tells `enumerateTextElements` to
  /// skip the element entirely — it contributes no layout.
  ///
  /// This is the SINGLE exception to the length-matching invariant: the
  /// element isn't vended at all, so there's no displayed paragraph for
  /// the cursor to land in. Forward / backward arrow-key motion still
  /// strides past the absorbed source chars because they're in
  /// `hiddenIndexes` and the move overrides skip them.
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
    for i in 0..<elementLength {
      if !hiddenIndexes.contains(elementOffset + i) {
        return true
      }
    }
    return false
  }
}
