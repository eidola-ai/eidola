import AppKit

/// Complete rendering specification for a markdown document at a given cursor position.
///
/// Produced by `MarkdownRenderer.render()` (pure function).
/// Consumed by `TextKit2RenderApplicator.apply()` (imperative shell).
///
/// The same `(text, cursorRange)` input must always produce an equivalent `RenderSpec`.
struct RenderSpec {
  /// Attributes applied to the full document range as a baseline.
  let baseAttributes: [NSAttributedString.Key: Any]

  /// Ranges with specific attributes, applied in order on top of base.
  let styledRanges: [StyledRange]

  /// Font traits to apply additively (so bold + italic combine correctly).
  let fontTraits: [TraitApplication]

  /// Characters whose glyphs should be suppressed (zero width via `.null` glyph property).
  let hiddenIndexes: IndexSet

  /// Characters whose glyphs should be replaced with a bullet (•).
  let bulletIndexes: IndexSet

  /// Characters whose glyphs should be replaced with an unchecked checkbox (☐ U+2610).
  let uncheckedCheckboxIndexes: IndexSet

  /// Characters whose glyphs should be replaced with a checked checkbox (☒ U+2612).
  let checkedCheckboxIndexes: IndexSet

  /// Newline characters whose glyphs should be set to `.null` so their line fragments
  /// collapse to zero height. Used for the first blank line in each inter-block gap
  /// so that `\n\n` paragraph separators render as a single visual line break.
  let collapsedNewlineIndexes: IndexSet

  /// Rendering-only attributes (e.g., dimmed delimiter color when cursor is inside a construct).
  /// Applied via `NSLayoutManager.addTemporaryAttributes` — they don't affect the text storage.
  let temporaryAttributes: [StyledRange]

  /// Character ranges that should receive full-width code block background drawing.
  /// Consumed by `TextKit2LayoutFragment` to draw backgrounds that span the
  /// entire text container width, regardless of glyph visibility.
  let codeBlockCharacterRanges: [CodeBlockDecoration]

  /// Character ranges that should receive a left border for blockquote visual indication.
  /// Each entry carries the exact x-position for the border, so drawing does not
  /// have to reconstruct nesting from partial metadata.
  let blockquoteCharacterRanges: [BlockquoteDecoration]

  struct BlockquoteDecoration {
    let range: NSRange
    let xPosition: CGFloat
  }

  struct CodeBlockDecoration {
    let range: NSRange
    let xOrigin: CGFloat
  }

  struct StyledRange {
    let range: NSRange
    let attributes: [NSAttributedString.Key: Any]
  }

  struct TraitApplication {
    let range: NSRange
    let trait: NSFontTraitMask
  }

  /// Deep equality check for testing. Standard `Equatable` doesn't work with `[Key: Any]`.
  func matches(_ other: RenderSpec) -> Bool {
    guard hiddenIndexes == other.hiddenIndexes,
      bulletIndexes == other.bulletIndexes,
      uncheckedCheckboxIndexes == other.uncheckedCheckboxIndexes,
      checkedCheckboxIndexes == other.checkedCheckboxIndexes,
      collapsedNewlineIndexes == other.collapsedNewlineIndexes,
      styledRanges.count == other.styledRanges.count,
      fontTraits.count == other.fontTraits.count,
      temporaryAttributes.count == other.temporaryAttributes.count,
      codeBlockCharacterRanges.count == other.codeBlockCharacterRanges.count,
      blockquoteCharacterRanges.count == other.blockquoteCharacterRanges.count
    else { return false }

    for (a, b) in zip(codeBlockCharacterRanges, other.codeBlockCharacterRanges) {
      guard a.range == b.range, a.xOrigin == b.xOrigin else { return false }
    }

    for (a, b) in zip(blockquoteCharacterRanges, other.blockquoteCharacterRanges) {
      guard a.range == b.range, a.xPosition == b.xPosition else { return false }
    }

    guard attrsEqual(baseAttributes, other.baseAttributes) else { return false }

    for (a, b) in zip(styledRanges, other.styledRanges) {
      guard a.range == b.range, attrsEqual(a.attributes, b.attributes) else { return false }
    }
    for (a, b) in zip(fontTraits, other.fontTraits) {
      guard a.range == b.range, a.trait == b.trait else { return false }
    }
    for (a, b) in zip(temporaryAttributes, other.temporaryAttributes) {
      guard a.range == b.range, attrsEqual(a.attributes, b.attributes) else { return false }
    }
    return true
  }
}

private func attrsEqual(
  _ a: [NSAttributedString.Key: Any],
  _ b: [NSAttributedString.Key: Any]
) -> Bool {
  guard a.count == b.count else { return false }
  for (key, valA) in a {
    guard let valB = b[key] else { return false }
    if let objA = valA as? NSObject, let objB = valB as? NSObject {
      guard objA.isEqual(objB) else { return false }
    } else {
      guard "\(valA)" == "\(valB)" else { return false }
    }
  }
  return true
}
