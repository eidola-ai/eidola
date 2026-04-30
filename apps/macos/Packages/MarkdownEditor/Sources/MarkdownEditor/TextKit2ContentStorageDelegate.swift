import AppKit

/// `NSTextContentStorageDelegate` that produces *display* paragraphs whose
/// `attributedString` differs from the source range — the TextKit 2
/// equivalent of the TextKit 1 glyph-hiding / glyph-substitution mechanism.
///
/// Source characters in `hiddenIndexes` are omitted from display. Source
/// characters in `bulletIndexes`, `uncheckedCheckboxIndexes`, and
/// `checkedCheckboxIndexes` are substituted with `•`, `☐`, and `☒`
/// respectively (preserving the source character's attributes). All other
/// characters pass through with their attributes.
///
/// `lineBreakIndexes` (soft / hard breaks identified by the AST) are NOT
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
  //
  // Mirrors GlyphHidingLayoutManagerDelegate's choices so the two paths
  // produce the same visual.

  private static let bulletString = "\u{2022}"  // •
  private static let uncheckedCheckboxString = "\u{25A1}"  // □ (BALLOT BOX is unavailable in system fonts)
  private static let checkedCheckboxString = "\u{2612}"  // ☒

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

    for i in 0..<scanLength {
      let docIdx = range.location + i

      if hiddenIndexes.contains(docIdx) {
        continue
      }

      let oneChar = source.attributedSubstring(from: NSRange(location: i, length: 1))
      let attrs = oneChar.attributes(at: 0, effectiveRange: nil)

      if bulletIndexes.contains(docIdx) {
        display.append(NSAttributedString(string: Self.bulletString, attributes: attrs))
      } else if uncheckedCheckboxIndexes.contains(docIdx) {
        display.append(
          NSAttributedString(string: Self.uncheckedCheckboxString, attributes: attrs))
      } else if checkedCheckboxIndexes.contains(docIdx) {
        display.append(
          NSAttributedString(string: Self.checkedCheckboxString, attributes: attrs))
      } else {
        display.append(oneChar)
      }
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

  // MARK: - Hit-test support

  /// Computes the hidden-prefix length (in source chars) at the start of
  /// the given paragraph source range, by walking the current
  /// `hiddenIndexes` set. Pure function over current state — no caching, so
  /// the result is always consistent with the delegate's index sets even
  /// after viewport scroll has caused TK2 to discard cached paragraphs.
  func computeHiddenPrefix(forParagraphSourceRange range: NSRange) -> Int {
    var prefix = 0
    while prefix < range.length, hiddenIndexes.contains(range.location + prefix) {
      prefix += 1
    }
    return prefix
  }
}
