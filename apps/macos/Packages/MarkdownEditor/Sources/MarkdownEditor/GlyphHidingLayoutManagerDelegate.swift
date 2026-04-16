import AppKit
import CoreText

/// NSLayoutManager delegate that hides characters by setting their glyph property to `.null`,
/// and optionally substitutes glyphs (e.g., `-` → `•` for list bullets).
///
/// Characters with `.null` glyphs remain in the text storage but occupy zero layout width.
/// This is the mechanism Apple recommends for inline WYSIWYG markdown editors
/// (WWDC 2018, Session 221 "TextKit Best Practices").
@MainActor
final class GlyphHidingLayoutManagerDelegate: NSObject, @preconcurrency NSLayoutManagerDelegate {
  /// Character indexes whose glyphs should be suppressed (zero width, not rendered).
  var hiddenCharacterIndexes = IndexSet()

  /// Character indexes whose glyphs should be replaced with a bullet (•).
  var bulletCharacterIndexes = IndexSet()

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
      if hiddenCharacterIndexes.contains(charIdx) || bulletCharacterIndexes.contains(charIdx) {
        needsModification = true
        break
      }
    }

    guard needsModification else { return 0 }

    // We need the text to detect paragraph boundaries.
    let text = layoutManager.textStorage?.string as NSString?

    // Look up zero-width space glyph lazily for paragraph-start hidden chars.
    var zwspGlyph: CGGlyph?

    let newGlyphs = UnsafeMutablePointer<CGGlyph>.allocate(capacity: count)
    let newProps = UnsafeMutablePointer<NSLayoutManager.GlyphProperty>.allocate(capacity: count)
    defer {
      newGlyphs.deallocate()
      newProps.deallocate()
    }

    // Look up bullet glyph lazily
    var bulletGlyph: CGGlyph?

    for i in 0..<count {
      let charIdx = charIndexes[i]
      if hiddenCharacterIndexes.contains(charIdx) {
        // For the very first hidden character at a paragraph start, use
        // `.controlCharacter` with a zero-width space glyph instead of `.null`.
        // This keeps the glyph participating in paragraph layout (so TextKit
        // correctly computes paragraphSpacingBefore/After) while rendering
        // nothing visible.
        let isParagraphStart: Bool
        if let text = text {
          if charIdx == 0 {
            isParagraphStart = true
          } else if charIdx > 0, charIdx < text.length {
            isParagraphStart = text.character(at: charIdx - 1) == 0x000A  // \n
          } else {
            isParagraphStart = false
          }
        } else {
          isParagraphStart = false
        }

        if isParagraphStart {
          if zwspGlyph == nil {
            var zwspChar: UniChar = 0x200B  // ZERO WIDTH SPACE
            var glyph: CGGlyph = 0
            CTFontGetGlyphsForCharacters(aFont as CTFont, &zwspChar, &glyph, 1)
            zwspGlyph = glyph
          }
          newGlyphs[i] = zwspGlyph ?? glyphs[i]
          newProps[i] = .controlCharacter
        } else {
          newGlyphs[i] = glyphs[i]
          newProps[i] = .null
        }
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
