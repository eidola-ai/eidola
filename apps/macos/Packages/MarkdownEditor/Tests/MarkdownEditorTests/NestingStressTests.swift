import AppKit
import Foundation
import Markdown
import Testing

@testable import MarkdownEditor

/// Stress tests for deeply nested content: blockquotes inside lists, code blocks
/// inside lists, lists inside blockquotes inside lists, etc. Validates that the
/// composable indent architecture positions each construct correctly regardless
/// of nesting depth.
@Suite("Nesting Stress Tests")
@MainActor
struct NestingStressTests {
  private func flatten(_ blocks: [MarkdownBlock]) -> [MarkdownBlock] {
    blocks + blocks.flatMap { flatten($0.children) }
  }

  // MARK: - Code block inside list

  @Test("Visual: code block inside list item")
  func visualCodeBlockInsideList() {
    let markdown = """
      - Item one
        ```
        code inside list
        ```
      - Item two

      Body text
      """
    let textLen = (markdown as NSString).length
    let initial = EditorState(markdown: markdown, selection: .cursor(textLen - 2))

    let results = EditorTestHarness.run(
      name: "nesting-codeblock-in-list",
      initial: initial,
      events: [
        .setSelection(.cursor(5)),   // inside list item text
        .setSelection(.cursor(20)),  // inside code block
        .setSelection(.cursor(textLen - 2)),  // body text
      ],
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 4)
    let fm = FileManager.default
    for r in results { #expect(fm.fileExists(atPath: r.imagePath)) }
    // Outside views should match
    #expect(results[0].bitmapHash == results[3].bitmapHash)
  }

  // MARK: - Blockquote inside list

  @Test("Visual: blockquote inside list item")
  func visualBlockquoteInsideList() {
    let markdown = """
      - Item one
        > Quoted text
        > More quoted
      - Item two

      Body text
      """
    let textLen = (markdown as NSString).length
    let initial = EditorState(markdown: markdown, selection: .cursor(textLen - 2))

    let results = EditorTestHarness.run(
      name: "nesting-blockquote-in-list",
      initial: initial,
      events: [
        .setSelection(.cursor(5)),   // inside list item before blockquote
        .setSelection(.cursor(16)),  // inside blockquote content
        .setSelection(.cursor(48)),  // inside second list item
        .setSelection(.cursor(textLen - 2)),  // body text
      ],
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 5)
    let fm = FileManager.default
    for r in results { #expect(fm.fileExists(atPath: r.imagePath)) }
    #expect(results[0].bitmapHash == results[4].bitmapHash)
  }

  // MARK: - Ordered list inside blockquote inside unordered list

  @Test("Visual: ordered list inside blockquote inside list")
  func visualOrderedListInBlockquoteInList() {
    let markdown = """
      - Item one
        > 1. First
        > 2. Second
        > 3. Third
      - Item two

      Body text
      """
    let textLen = (markdown as NSString).length
    let initial = EditorState(markdown: markdown, selection: .cursor(textLen - 2))

    let results = EditorTestHarness.run(
      name: "nesting-ordered-in-bq-in-list",
      initial: initial,
      events: [
        .setSelection(.cursor(5)),   // in list item
        .setSelection(.cursor(20)),  // in ordered list inside blockquote
        .setSelection(.cursor(textLen - 2)),  // body text
      ],
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 4)
    let fm = FileManager.default
    for r in results { #expect(fm.fileExists(atPath: r.imagePath)) }
  }

  // MARK: - Code block inside blockquote

  @Test("Visual: code block inside blockquote")
  func visualCodeBlockInsideBlockquote() {
    let markdown = """
      > Some text
      > ```
      > let x = 42
      > ```
      > More text

      Body text
      """
    let textLen = (markdown as NSString).length
    let initial = EditorState(markdown: markdown, selection: .cursor(textLen - 2))

    let results = EditorTestHarness.run(
      name: "nesting-codeblock-in-blockquote",
      initial: initial,
      events: [
        .setSelection(.cursor(5)),   // in blockquote text
        .setSelection(.cursor(20)),  // in code block
        .setSelection(.cursor(textLen - 2)),  // body
      ],
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 4)
    let fm = FileManager.default
    for r in results { #expect(fm.fileExists(atPath: r.imagePath)) }
  }

  // MARK: - Deep nesting: list > blockquote > list > code block

  @Test("Visual: deep nesting list > blockquote > ordered list > code block")
  func visualDeepNesting() {
    let markdown = """
      - Outer list item
        > Blockquote inside list
        > 1. Ordered inside blockquote
        >    ```
        >    deeply nested code
        >    ```
        > 2. Another ordered item

      Body text
      """
    let textLen = (markdown as NSString).length
    let initial = EditorState(markdown: markdown, selection: .cursor(textLen - 2))

    let results = EditorTestHarness.run(
      name: "nesting-deep-list-bq-list-code",
      initial: initial,
      events: [
        .setSelection(.cursor(5)),    // outer list item
        .setSelection(.cursor(25)),   // blockquote text
        .setSelection(.cursor(55)),   // ordered list item
        .setSelection(.cursor(80)),   // code block
        .setSelection(.cursor(textLen - 2)),  // body
      ],
      size: NSSize(width: 600, height: 400))

    #expect(results.count == 6)
    let fm = FileManager.default
    for r in results { #expect(fm.fileExists(atPath: r.imagePath)) }
  }

  // MARK: - Blockquote > list > blockquote (alternating)

  @Test("Visual: alternating blockquote > list > blockquote")
  func visualAlternatingNesting() {
    let markdown = """
      > Outer blockquote
      > - List inside blockquote
      >   > Inner blockquote inside list inside blockquote
      >   > More inner text
      > - Another list item

      Body text
      """
    let textLen = (markdown as NSString).length
    let initial = EditorState(markdown: markdown, selection: .cursor(textLen - 2))

    let results = EditorTestHarness.run(
      name: "nesting-alternating-bq-list-bq",
      initial: initial,
      events: [
        .setSelection(.cursor(5)),    // outer blockquote
        .setSelection(.cursor(25)),   // list inside blockquote
        .setSelection(.cursor(40)),   // inner blockquote
        .setSelection(.cursor(textLen - 2)),  // body
      ],
      size: NSSize(width: 600, height: 400))

    #expect(results.count == 5)
    let fm = FileManager.default
    for r in results { #expect(fm.fileExists(atPath: r.imagePath)) }
  }

  // MARK: - Continuation line alignment in nested contexts

  @Test("Visual: continuation lines in ordered list inside blockquote")
  func visualContinuationInNestedOrderedList() {
    let markdown = """
      > 1. Short
      > 2. This is a longer item that should demonstrate word wrapping and continuation line alignment within a blockquote

      Body text
      """
    let textLen = (markdown as NSString).length
    let initial = EditorState(markdown: markdown, selection: .cursor(textLen - 2))

    let results = EditorTestHarness.run(
      name: "nesting-continuation-ordered-in-bq",
      initial: initial,
      events: [
        .setSelection(.cursor(10)),   // inside blockquote
        .setSelection(.cursor(textLen - 2)),  // body
      ],
      size: NSSize(width: 400, height: 300))  // narrow to force wrapping

    #expect(results.count == 3)
    let fm = FileManager.default
    for r in results { #expect(fm.fileExists(atPath: r.imagePath)) }
  }

  // MARK: - Multiple list items with various nested blocks

  @Test("Visual: comprehensive blocks nested inside lists")
  func visualComprehensiveBlocksInLists() {
    let markdown = """
      ## Blocks nested inside lists

      - Some body text
        ```
        code
        ```
      - More body text
        > A blockquote
        > with multiple lines
        > 1. one
        > 2. two
        >    1. two, one
      - More body text
      - Another list item

      Body text outside
      """
    let textLen = (markdown as NSString).length
    let initial = EditorState(markdown: markdown, selection: .cursor(textLen - 5))

    let results = EditorTestHarness.run(
      name: "nesting-comprehensive-blocks-in-lists",
      initial: initial,
      events: [
        .setSelection(.cursor(40)),   // heading area
        .setSelection(.cursor(55)),   // code block
        .setSelection(.cursor(75)),   // blockquote
        .setSelection(.cursor(120)),  // ordered list in blockquote
        .setSelection(.cursor(textLen - 5)),  // body
      ],
      size: NSSize(width: 600, height: 500))

    #expect(results.count == 6)
    let fm = FileManager.default
    for r in results { #expect(fm.fileExists(atPath: r.imagePath)) }
  }

  // MARK: - Lists nested inside blockquotes (with continuation)

  @Test("Visual: comprehensive lists nested inside blockquotes")
  func visualComprehensiveListsInBlockquotes() {
    let markdown = """
      ## Lists nested inside blockquotes

      > Body text
      > 1. one
      >    1. one, two
      >       continuation should be aligned with "one, two"
      > 2. two two two two two two two two two two two two two two two two two two two two two
      >

      Body text outside
      """
    let textLen = (markdown as NSString).length
    let initial = EditorState(markdown: markdown, selection: .cursor(textLen - 5))

    let results = EditorTestHarness.run(
      name: "nesting-comprehensive-lists-in-bq",
      initial: initial,
      events: [
        .setSelection(.cursor(45)),   // inside blockquote
        .setSelection(.cursor(65)),   // inside nested ordered list
        .setSelection(.cursor(90)),   // inside continuation
        .setSelection(.cursor(textLen - 5)),  // body
      ],
      size: NSSize(width: 400, height: 400))  // narrow to force wrapping

    #expect(results.count == 5)
    let fm = FileManager.default
    for r in results { #expect(fm.fileExists(atPath: r.imagePath)) }
  }

  @Test("Visual: reference stress test from nested lists, blockquotes, and code blocks")
  func visualReferenceStressDocument() {
    let markdown = """
      ## Ordered List
      1. Simple list item
          1. Nested item
          2. LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL. Overflow should line up.
          3. This is the first line.
             This is the second line. (They should line up.) LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL. Overflow should line up too.
      2. This contains a nested blockquote:
         > Some quoted text.
      3. This contains a nested code block:
         ```js
         let foo = "bar";
         ```


      ## Blockquote
      > Normal text.
      > 1. Nested ordered list
      >     1. With a second depth
      >     2. LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL. Overflow should line up.
      >     3. This is the first line.
      >        This is the second line. (They should line up.) LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL LLL. Overflow should line up too.
      > 
      > Containing a code block:
      > ```js
      > let foo = "bar";
      > ```
      > 
      > Time for some russian stacking dolls:
      > - unordered list
      >     - with a blockquote:
      >       > Some quoted text that contains an ordered list:
      >       > 1. one
      >       > 2. with a code block:
      >       >    ```js
      >       >    let foo = "bar";
      >       >    ```
      >     - and a nested to-do list:
      >         - [ ] Do this
      >         - [x] Do that
      """
    let textLen = (markdown as NSString).length
    let initial = EditorState(markdown: markdown, selection: .cursor(20))

    let nsText = markdown as NSString
    let blockquoteOffset = nsText.range(of: "Normal text.").location
    let allCodeMatches = try! NSRegularExpression(pattern: #"let foo = "bar";"#)
      .matches(in: markdown, range: NSRange(location: 0, length: textLen))
    let nestedCodeOffset = allCodeMatches.last?.range.location ?? max(0, textLen - 1)

    let results = EditorTestHarness.run(
      name: "nesting-reference-stress-doc",
      initial: initial,
      events: [
        .setSelection(.cursor(max(0, blockquoteOffset + 2))),
        .setSelection(.cursor(nestedCodeOffset + 4)),
        .setSelection(.cursor(20)),
      ],
      size: NSSize(width: 900, height: 1200))

    #expect(results.count == 4)
    let fm = FileManager.default
    for r in results { #expect(fm.fileExists(atPath: r.imagePath)) }
    #expect(results[0].bitmapHash == results[3].bitmapHash)
  }

  // MARK: - Parser: structural validation

  @Test("Parser: blockquote inside list is nested under a list item")
  func parserBlockquoteInsideListIsNested() {
    let text = "- Item\n  > Quote"
    let converter = SourceRangeConverter(string: text)
    var parser = MarkdownParser(converter: converter)
    let doc = Document(parsing: text)
    parser.visit(doc)

    let blocks = flatten(parser.document?.blocks ?? [])
    let bqNode = blocks.first { n in
      if case .blockquote = n.kind { return true }
      return false
    }
    #expect(bqNode != nil, "Should find blockquote node")
    if let bq = bqNode {
      let enclosingListItem = blocks.first { candidate in
        if case .listItem = candidate.kind {
          return candidate.range.location <= bq.range.location
            && candidate.range.location + candidate.range.length >= bq.range.location + bq.range.length
        }
        return false
      }
      #expect(enclosingListItem != nil, "Blockquote should be nested under a list item")
    }
  }

  @Test("Parser: code block inside list is nested under a list item")
  func parserCodeBlockInsideListIsNested() {
    let text = "- Item\n  ```\n  code\n  ```"
    let converter = SourceRangeConverter(string: text)
    var parser = MarkdownParser(converter: converter)
    let doc = Document(parsing: text)
    parser.visit(doc)

    let blocks = flatten(parser.document?.blocks ?? [])
    let cbNode = blocks.first { n in
      if case .codeBlock = n.kind { return true }
      return false
    }
    #expect(cbNode != nil, "Should find code block node")
    if let cb = cbNode {
      let enclosingListItem = blocks.first { candidate in
        if case .listItem = candidate.kind {
          return candidate.range.location <= cb.range.location
            && candidate.range.location + candidate.range.length >= cb.range.location + cb.range.length
        }
        return false
      }
      #expect(enclosingListItem != nil, "Code block should be nested under a list item")
    }
  }

  @Test("Parser: top-level blockquote has no enclosing list item")
  func parserTopLevelBlockquoteHasNoEnclosingListItem() {
    let text = "> Quote"
    let converter = SourceRangeConverter(string: text)
    var parser = MarkdownParser(converter: converter)
    let doc = Document(parsing: text)
    parser.visit(doc)

    let blocks = flatten(parser.document?.blocks ?? [])
    let bqNode = blocks.first { n in
      if case .blockquote = n.kind { return true }
      return false
    }
    #expect(bqNode != nil)
    if let bq = bqNode {
      let enclosingListItem = blocks.first { candidate in
        if case .listItem = candidate.kind {
          return candidate.range.location <= bq.range.location
            && candidate.range.location + candidate.range.length >= bq.range.location + bq.range.length
        }
        return false
      }
      #expect(enclosingListItem == nil, "Top-level blockquote should not be nested under a list item")
    }
  }

  // MARK: - Render spec validation

  @Test("Render: blockquote border inside list is offset by list indent")
  func renderBlockquoteBorderInsideListIsOffset() {
    let text = "- Item\n  > Quote\n\nBody"
    let cursorRange = NSRange(location: (text as NSString).length - 1, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(!spec.blockquoteCharacterRanges.isEmpty, "Should have blockquote border range")
    if let bqRange = spec.blockquoteCharacterRanges.first {
      #expect(bqRange.xPosition > 0, "Blockquote border inside list should have positive x position")
    }
  }

  @Test("Render: code block inside list still emits a BlockRendererSpec")
  func renderCodeBlockInsideListEmitsSpec() {
    // Pre-2.2 this test asserted that a list-nested code block produced
    // a `codeBlockCharacterRanges` decoration with `xOrigin > 0` (the
    // legacy painting path needed an x-origin to know where to start
    // its full-width background fill, indented past the list marker).
    // Post-2.2 the painting path is gone — code blocks are rendered by
    // the embedded `CodeBlockRenderer`'s own `NSTextView`, whose
    // horizontal position is determined by the attachment paragraph's
    // `firstLineHeadIndent` (set via the regular list / blockquote
    // indent machinery) rather than a per-decoration x-origin. The new
    // invariant is just that the spec is emitted at all.
    let text = "- Item\n  ```\n  code\n  ```\n\nBody"
    let cursorRange = NSRange(location: (text as NSString).length - 1, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(
      !spec.blockRendererSpecs.isEmpty,
      "list-nested code block should still emit a BlockRendererSpec for the registry to reconcile a host against")
    if let rendererSpec = spec.blockRendererSpecs.first {
      #expect(rendererSpec.blockTypeTag == .codeBlock)
      #expect(rendererSpec.mode == .editInPlace)
    }
  }

  @Test("Render: nested code block inside list does not emit hidden-index entries for fences")
  func renderNestedCodeBlockInsideListDoesNotHideFences() {
    // Pre-2.2 this test asserted that the fence chars stayed out of
    // `hiddenIndexes` ONLY when the cursor was inside the code block —
    // outside, the legacy renderer hid the backticks. Post-2.2 the
    // entire code-block range is owned by the embedded renderer, so
    // fence chars are never added to `hiddenIndexes` regardless of
    // cursor position. The new invariant is the unconditional absence.
    let text = "- Item\n  ```\n  code\n  ```\n\nBody"
    let cursorOutside = NSRange(location: (text as NSString).length - 1, length: 0)
    let cursorInside = NSRange(location: 16, length: 0)
    for cursorRange in [cursorOutside, cursorInside] {
      let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)
      for idx in [9, 10, 11, 22, 23, 24] {
        #expect(
          !spec.hiddenIndexes.contains(idx),
          "Fence character at \(idx) must not be hidden — embedded renderer owns code-block visibility regardless of cursor (cursor=\(cursorRange))")
      }
    }
  }

  @Test("Render: nested inner blockquote prefixes are hidden when cursor is outside")
  func renderNestedInnerBlockquotePrefixesHideOutside() {
    let text = """
      > Outer
      > - Item
      >   > Inner
      >   > More

      Body
      """
    let cursorRange = NSRange(location: (text as NSString).length - 2, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    let nsText = text as NSString
    let firstInner = nsText.range(of: "> Inner")
    let secondInner = nsText.range(of: "> More")
    #expect(firstInner.location != NSNotFound)
    #expect(secondInner.location != NSNotFound)

    if firstInner.location != NSNotFound {
      #expect(!spec.hiddenIndexes.contains(firstInner.location), "Inner blockquote prefix should not be hidden (transparent glyph)")
    }
    if secondInner.location != NSNotFound {
      #expect(!spec.hiddenIndexes.contains(secondInner.location), "Inner blockquote prefix should not be hidden on subsequent lines (transparent glyph)")
    }
  }

  @Test("Render: top-level code block emits exactly one BlockRendererSpec")
  func renderTopLevelCodeBlockEmitsOneSpec() {
    // Pre-2.2 this test asserted that a top-level code block produced a
    // `codeBlockCharacterRanges` decoration with `xOrigin == 0` (the
    // legacy painted background started at x=0 for top-level code).
    // Post-2.2 there is no painted background; the new invariant is
    // simply that the bridging-layer spec is emitted.
    let text = "```\ncode\n```\n\nBody"
    let cursorRange = NSRange(location: (text as NSString).length - 1, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(spec.blockRendererSpecs.count == 1)
    #expect(spec.blockRendererSpecs.first?.blockTypeTag == .codeBlock)
  }
}
