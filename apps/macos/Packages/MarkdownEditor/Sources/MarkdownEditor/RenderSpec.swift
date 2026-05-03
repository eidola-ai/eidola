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

  /// Source-character offsets of `\n` characters that should be displayed as
  /// `U+2028 LINE SEPARATOR` instead of a paragraph break. Populated for every
  /// `\n` that the AST identifies as a `SoftBreak` or `LineBreak` inside a
  /// single Paragraph — the renderer keeps the source verbatim and the
  /// content-storage delegate substitutes at display time so TextKit treats
  /// the break as in-paragraph.
  let lineBreakIndexes: IndexSet

  /// Rendering-only attributes (e.g., dimmed delimiter color when cursor is inside a construct).
  /// Applied via `NSLayoutManager.addTemporaryAttributes` — they don't affect the text storage.
  let temporaryAttributes: [StyledRange]

  /// Character ranges that should receive a left border for blockquote visual indication.
  /// Each entry carries the exact x-position for the border, so drawing does not
  /// have to reconstruct nesting from partial metadata.
  let blockquoteCharacterRanges: [BlockquoteDecoration]

  /// Block-level constructs that want a custom-view renderer (Phase 2
  /// bridging layer). Each entry drives `BlockRendererRegistry`
  /// reconciliation in `TextKit2RenderApplicator.apply`. As of Phase 2.2
  /// code blocks are rendered exclusively through this path — the legacy
  /// `codeBlockCharacterRanges` painting branch has been retired.
  let blockRendererSpecs: [BlockRendererSpec]

  struct BlockquoteDecoration {
    let range: NSRange
    let xPosition: CGFloat
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
      lineBreakIndexes == other.lineBreakIndexes,
      styledRanges.count == other.styledRanges.count,
      fontTraits.count == other.fontTraits.count,
      temporaryAttributes.count == other.temporaryAttributes.count,
      blockquoteCharacterRanges.count == other.blockquoteCharacterRanges.count,
      blockRendererSpecs.count == other.blockRendererSpecs.count
    else { return false }

    for (a, b) in zip(blockquoteCharacterRanges, other.blockquoteCharacterRanges) {
      guard a.range == b.range, a.xPosition == b.xPosition else { return false }
    }

    for (a, b) in zip(blockRendererSpecs, other.blockRendererSpecs) {
      guard a.range == b.range, a.blockTypeTag == b.blockTypeTag,
        a.mode == b.mode, a.reservedHeight == b.reservedHeight
      else { return false }
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
