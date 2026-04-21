import AppKit
import CoreText

/// NSLayoutManager delegate that hides characters by replacing their glyphs with a
/// zero-width space (ZWSP) and setting the `.controlCharacter` property, and optionally
/// substitutes glyphs (e.g., `-` → `•` for list bullets).
///
/// Hidden characters remain in the text storage but occupy zero layout width. We use
/// `.controlCharacter` + ZWSP rather than `.null` because `.null` glyphs do not fully
/// participate in layout — TextKit 1 miscalculates advance widths and `.backgroundColor`
/// rects when `.null` glyphs appear at line wrapping boundaries.
@MainActor
final class GlyphHidingLayoutManagerDelegate: NSObject, @preconcurrency NSLayoutManagerDelegate {
  /// Character indexes whose glyphs should be suppressed (zero width, not rendered).
  var hiddenCharacterIndexes = IndexSet()

  /// Character indexes whose glyphs should be replaced with a bullet (•).
  var bulletCharacterIndexes = IndexSet()

  /// Character indexes whose glyphs should be replaced with an unchecked checkbox (☐ U+2610).
  var uncheckedCheckboxCharacterIndexes = IndexSet()

  /// Character indexes whose glyphs should be replaced with a checked checkbox (☒ U+2612).
  var checkedCheckboxCharacterIndexes = IndexSet()

  // MARK: - Glyph Generation

  func layoutManager(
    _ layoutManager: NSLayoutManager,
    shouldGenerateGlyphs glyphs: UnsafePointer<CGGlyph>,
    properties props: UnsafePointer<NSLayoutManager.GlyphProperty>,
    characterIndexes charIndexes: UnsafePointer<Int>,
    font aFont: NSFont,
    forGlyphRange glyphRange: NSRange
  ) -> Int {
    let count = glyphRange.length
    guard count > 0 else { return 0 }

    // Quick check: does any character in this range need modification?
    var needsModification = false
    for i in 0..<count {
      let charIdx = charIndexes[i]
      if hiddenCharacterIndexes.contains(charIdx) || bulletCharacterIndexes.contains(charIdx)
        || uncheckedCheckboxCharacterIndexes.contains(charIdx)
        || checkedCheckboxCharacterIndexes.contains(charIdx)
      {
        needsModification = true
        break
      }
    }

    guard needsModification else { return 0 }

    // Look up zero-width space glyph lazily for hidden characters.
    var zwspGlyph: CGGlyph?

    let newGlyphs = UnsafeMutablePointer<CGGlyph>.allocate(capacity: count)
    let newProps = UnsafeMutablePointer<NSLayoutManager.GlyphProperty>.allocate(capacity: count)
    defer {
      newGlyphs.deallocate()
      newProps.deallocate()
    }

    // Look up replacement glyphs lazily
    var bulletGlyph: CGGlyph?
    var uncheckedCheckboxGlyph: CGGlyph?
    var checkedCheckboxGlyph: CGGlyph?

    for i in 0..<count {
      let charIdx = charIndexes[i]
      if hiddenCharacterIndexes.contains(charIdx) {
        if zwspGlyph == nil {
          var zwspChar: UniChar = 0x200B  // ZERO WIDTH SPACE
          var glyph: CGGlyph = 0
          CTFontGetGlyphsForCharacters(aFont as CTFont, &zwspChar, &glyph, 1)
          zwspGlyph = glyph
        }
        newGlyphs[i] = zwspGlyph ?? glyphs[i]
        newProps[i] = .controlCharacter
      } else if bulletCharacterIndexes.contains(charIdx) {
        // Replace with bullet glyph
        if bulletGlyph == nil {
          var bulletChar: UniChar = 0x2022  // •
          var glyph: CGGlyph = 0
          CTFontGetGlyphsForCharacters(aFont as CTFont, &bulletChar, &glyph, 1)
          bulletGlyph = glyph
        }
        newGlyphs[i] = bulletGlyph ?? glyphs[i]
        newProps[i] = props[i]
      } else if uncheckedCheckboxCharacterIndexes.contains(charIdx) {
        // Replace with unchecked checkbox glyph □ (U+25A1 WHITE SQUARE)
        // Note: U+2610 BALLOT BOX is not in the macOS system font, so we use
        // U+25A1 which is visually similar and available.
        if uncheckedCheckboxGlyph == nil {
          var checkboxChar: UniChar = 0x25A1  // □ WHITE SQUARE
          var glyph: CGGlyph = 0
          CTFontGetGlyphsForCharacters(aFont as CTFont, &checkboxChar, &glyph, 1)
          uncheckedCheckboxGlyph = glyph
        }
        newGlyphs[i] = uncheckedCheckboxGlyph ?? glyphs[i]
        newProps[i] = props[i]
      } else if checkedCheckboxCharacterIndexes.contains(charIdx) {
        // Replace with checked checkbox glyph ☒ (U+2612 BALLOT BOX WITH X)
        if checkedCheckboxGlyph == nil {
          var checkboxChar: UniChar = 0x2612  // ☒ BALLOT BOX WITH X
          var glyph: CGGlyph = 0
          CTFontGetGlyphsForCharacters(aFont as CTFont, &checkboxChar, &glyph, 1)
          checkedCheckboxGlyph = glyph
        }
        newGlyphs[i] = checkedCheckboxGlyph ?? glyphs[i]
        newProps[i] = props[i]
      } else {
        newGlyphs[i] = glyphs[i]
        newProps[i] = props[i]
      }
    }

    layoutManager.setGlyphs(
      newGlyphs, properties: newProps,
      characterIndexes: charIndexes,
      font: aFont, forGlyphRange: glyphRange
    )
    return count
  }

}
