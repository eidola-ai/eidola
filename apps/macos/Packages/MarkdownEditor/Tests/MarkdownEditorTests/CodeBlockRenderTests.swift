import AppKit
import Testing

@testable import MarkdownEditor

@Suite("Code Block Render Tests")
@MainActor
struct CodeBlockRenderTests {
  // MARK: - Block-renderer spec emission
  //
  // Phase 2.2 retired the per-fragment legacy painting path and the
  // bespoke fence/font/paragraph-style emission for code blocks. The
  // entire visual is now produced by the embedded `CodeBlockRenderer`
  // (an `NSScrollView { NSTextView }` subtree anchored to a
  // `BlockAttachment`). The renderer's only output for a code block is
  // a `BlockRendererSpec` covering the full source range, so this file
  // pins on that single invariant rather than the legacy
  // `codeBlockCharacterRanges` decoration / per-line styled ranges /
  // delimiter-hide index sets the pre-2.2 tests asserted on.

  @Test("Code block emits a BlockRendererSpec covering the full source range")
  func codeBlockEmitsBlockRendererSpec() {
    let text = "```\nlet x = 42\n```"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(
      spec.blockRendererSpecs.count == 1,
      "renderer should emit exactly one BlockRendererSpec for the single code block")
    let block = spec.blockRendererSpecs[0]
    #expect(block.blockTypeTag == .codeBlock, "spec must carry the .codeBlock tag")
    #expect(block.mode == .editInPlace, "code blocks are edit-in-place")
    #expect(
      block.range == NSRange(location: 0, length: (text as NSString).length),
      "spec range should cover the entire code block including fences")
  }

  @Test("Code block with language hint preserves the spec range")
  func codeBlockLanguageHintSpec() {
    let text = "```swift\nlet x = 42\n```"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(spec.blockRendererSpecs.count == 1)
    #expect(
      spec.blockRendererSpecs[0].range == NSRange(location: 0, length: (text as NSString).length),
      "language hint is part of the opening fence and stays inside the block range")
  }

  @Test("Code block reservedHeight scales with the source line count")
  func codeBlockReservedHeightGrowsWithLines() {
    let one = "```\nlet x = 1\n```"
    let three = "```\nlet x = 1\nlet y = 2\nlet z = 3\n```"
    let specOne = MarkdownRenderer.render(
      text: one, cursorRange: NSRange(location: 0, length: 0))
    let specThree = MarkdownRenderer.render(
      text: three, cursorRange: NSRange(location: 0, length: 0))

    let hOne = specOne.blockRendererSpecs.first?.reservedHeight ?? 0
    let hThree = specThree.blockRendererSpecs.first?.reservedHeight ?? 0
    #expect(hOne > 0, "single-line code block should reserve a non-zero height")
    #expect(hThree > hOne, "three-line code block should reserve more height than one-line")
  }

  // MARK: - No legacy emission
  //
  // The pre-2.2 renderer also emitted per-character `hiddenIndexes`
  // entries for code-block fences (depending on cursor position),
  // monospace `styledRanges` for the content, and `temporaryAttributes`
  // for dimmed fences when the cursor was inside the block. None of
  // those survive into the spec now: the embedded renderer paints the
  // entire block from its own `NSTextView`, so the legacy outputs would
  // be duplicated work. The tests below pin the absence as a
  // regression guard.

  @Test("Code block does not emit hidden-index entries for the source range")
  func codeBlockNoHiddenIndexesInSourceRange() {
    // Cursor outside the block. The legacy path hid the fence backticks
    // here; the new path leaves the index set untouched for the block.
    let text = "hello\n\n```\ncode\n```\n\nworld"
    let spec = MarkdownRenderer.render(
      text: text, cursorRange: NSRange(location: 0, length: 0))
    let blockRange = NSRange(location: 7, length: ("```\ncode\n```" as NSString).length)
    for i in blockRange.location..<(blockRange.location + blockRange.length) {
      #expect(
        !spec.hiddenIndexes.contains(i),
        "code-block source position \(i) must not be in hiddenIndexes; the embedded renderer owns visibility for the whole block (pre-2.2 hid the fence chars at \(i) when the cursor was outside)")
    }
  }

  @Test("Code block does not emit fence dimming temporaryAttributes")
  func codeBlockNoFenceDimming() {
    // Cursor INSIDE the code block. Pre-2.2 the renderer added a
    // `temporaryAttributes` entry per fence to dim it; post-2.2 the
    // fence visibility is the embedded view's own concern.
    let text = "```\ncode\n```"
    let spec = MarkdownRenderer.render(
      text: text, cursorRange: NSRange(location: 5, length: 0))

    let blockRange = NSRange(location: 0, length: (text as NSString).length)
    for tempAttr in spec.temporaryAttributes {
      let intersection = NSIntersectionRange(tempAttr.range, blockRange)
      #expect(
        intersection.length == 0,
        "no temporaryAttributes entry should cover the code-block source range; embedded renderer owns delimiter dimming")
    }
  }

  @Test("Code block does not emit monospace styledRanges for content")
  func codeBlockNoContentStyling() {
    // Pre-2.2 every code block produced a `.font: codeFont` styled
    // range over its content. Post-2.2 the embedded `NSTextView`
    // applies its own monospace font to the source it mirrors, so the
    // main view's styled ranges should not target the block.
    let text = "```\nlet x = 42\n```"
    let spec = MarkdownRenderer.render(
      text: text, cursorRange: NSRange(location: 0, length: 0))
    let blockRange = NSRange(location: 0, length: (text as NSString).length)

    for styled in spec.styledRanges {
      // The styled-range list also carries the document base attributes;
      // we specifically check that no entry targets the block range with
      // a code font.
      let intersection = NSIntersectionRange(styled.range, blockRange)
      if intersection.length > 0,
        let font = styled.attributes[.font] as? NSFont,
        font.fontDescriptor.symbolicTraits.contains(.monoSpace)
      {
        Issue.record(
          "found a monospace styledRange targeting the code-block source range — embedded renderer should be the only thing applying monospace to code-block content")
      }
    }
  }

  // MARK: - Block-renderer spec for blockquoted code block
  //
  // A code block nested inside a blockquote still emits exactly one
  // `BlockRendererSpec`. Blockquote prefix handling (`>` rendering, kern
  // override, indent inheritance) is independent of the code-block
  // renderer and continues to work in the surrounding paragraphs.

  @Test("Code block inside blockquote still emits a spec")
  func blockquoteCodeBlockSpec() {
    let text = "> ```js\n> let x = 42\n> ```\n\nBody"
    let spec = MarkdownRenderer.render(
      text: text, cursorRange: NSRange(location: 0, length: 0))
    #expect(
      spec.blockRendererSpecs.count == 1,
      "expected one BlockRendererSpec for the code block, even when nested inside a blockquote")
  }
}
