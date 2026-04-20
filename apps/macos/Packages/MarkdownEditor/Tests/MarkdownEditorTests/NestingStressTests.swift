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

  // MARK: - Parser: structural validation

  @Test("Parser: blockquote inside list has listBaseIndent > 0")
  func parserBlockquoteInsideListHasBaseIndent() {
    let text = "- Item\n  > Quote"
    let converter = SourceRangeConverter(string: text)
    var parser = MarkdownParser(converter: converter)
    let doc = Document(parsing: text)
    parser.visit(doc)

    let bqNode = parser.nodes.first { n in
      if case .blockquote = n.type { return true }
      return false
    }
    #expect(bqNode != nil, "Should find blockquote node")
    if let bq = bqNode, case .blockquote(_, let lbi) = bq.type {
      #expect(lbi > 0, "Blockquote inside list should have listBaseIndent > 0, got \(lbi)")
    }
  }

  @Test("Parser: code block inside list has listBaseIndent > 0")
  func parserCodeBlockInsideListHasBaseIndent() {
    let text = "- Item\n  ```\n  code\n  ```"
    let converter = SourceRangeConverter(string: text)
    var parser = MarkdownParser(converter: converter)
    let doc = Document(parsing: text)
    parser.visit(doc)

    let cbNode = parser.nodes.first { n in
      if case .codeBlock = n.type { return true }
      return false
    }
    #expect(cbNode != nil, "Should find code block node")
    if let cb = cbNode, case .codeBlock(_, let lbi) = cb.type {
      #expect(lbi > 0, "Code block inside list should have listBaseIndent > 0, got \(lbi)")
    }
  }

  @Test("Parser: top-level blockquote has listBaseIndent == 0")
  func parserTopLevelBlockquoteHasZeroBaseIndent() {
    let text = "> Quote"
    let converter = SourceRangeConverter(string: text)
    var parser = MarkdownParser(converter: converter)
    let doc = Document(parsing: text)
    parser.visit(doc)

    let bqNode = parser.nodes.first { n in
      if case .blockquote = n.type { return true }
      return false
    }
    #expect(bqNode != nil)
    if let bq = bqNode, case .blockquote(_, let lbi) = bq.type {
      #expect(lbi == 0, "Top-level blockquote should have listBaseIndent == 0")
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
      #expect(bqRange.listBaseIndent > 0,
        "Blockquote border inside list should have listBaseIndent > 0, got \(bqRange.listBaseIndent)")
    }
  }

  @Test("Render: code block inside list has baseIndent > 0")
  func renderCodeBlockInsideListHasBaseIndent() {
    let text = "- Item\n  ```\n  code\n  ```\n\nBody"
    let cursorRange = NSRange(location: (text as NSString).length - 1, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(!spec.codeBlockCharacterRanges.isEmpty, "Should have code block range")
    if let cbRange = spec.codeBlockCharacterRanges.first {
      #expect(cbRange.baseIndent > 0,
        "Code block inside list should have baseIndent > 0, got \(cbRange.baseIndent)")
    }
  }

  @Test("Render: top-level code block has baseIndent == 0")
  func renderTopLevelCodeBlockHasZeroBaseIndent() {
    let text = "```\ncode\n```\n\nBody"
    let cursorRange = NSRange(location: (text as NSString).length - 1, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(!spec.codeBlockCharacterRanges.isEmpty)
    if let cbRange = spec.codeBlockCharacterRanges.first {
      #expect(cbRange.baseIndent == 0, "Top-level code block should have baseIndent == 0")
    }
  }
}
