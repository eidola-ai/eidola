import AppKit
import Foundation
import Markdown
import Testing

@testable import MarkdownEditor

/// Tests for nested content inside blockquotes: lists, nested blockquotes,
/// code blocks, and inline formatting.
@Suite("Blockquote Nested Content Tests")
@MainActor
struct BlockquoteNestedContentTests {
  private func flatten(_ blocks: [MarkdownBlock]) -> [MarkdownBlock] {
    blocks + blocks.flatMap { flatten($0.children) }
  }

  // MARK: - Parser: verify correct node emission for nested content

  @Test("Parser emits blockquote and list item nodes for list inside blockquote")
  func parserEmitsListInsideBlockquote() {
    let text = "> - Item one\n> - Item two"
    let converter = SourceRangeConverter(string: text)
    var parser = MarkdownParser(converter: converter)
    let doc = Document(parsing: text)
    parser.visit(doc)
    let nodes = flatten(parser.document?.blocks ?? [])

    let hasBlockquote = nodes.contains { node in
      if case .blockquote = node.kind { return true }
      return false
    }
    #expect(hasBlockquote, "Should have a blockquote node")

    let listItems = nodes.filter { node in
      if case .listItem(let syntax) = node.kind, case .unordered = syntax.kind { return true }
      return false
    }
    #expect(listItems.count == 2, "Should have 2 list item nodes")
  }

  @Test("Parser emits outer and inner blockquote nodes")
  func parserEmitsNestedBlockquote() {
    let text = "> Outer\n> > Inner"
    let converter = SourceRangeConverter(string: text)
    var parser = MarkdownParser(converter: converter)
    let doc = Document(parsing: text)
    parser.visit(doc)
    let nodes = flatten(parser.document?.blocks ?? [])

    let blockquotes = nodes.filter { node in
      if case .blockquote = node.kind { return true }
      return false
    }
    #expect(blockquotes.count >= 2, "Should have at least 2 blockquote nodes")
  }

  @Test("Parser emits blockquote and code block nodes for code block inside blockquote")
  func parserEmitsCodeBlockInsideBlockquote() {
    let text = "> ```\n> code\n> ```"
    let converter = SourceRangeConverter(string: text)
    var parser = MarkdownParser(converter: converter)
    let doc = Document(parsing: text)
    parser.visit(doc)
    let nodes = flatten(parser.document?.blocks ?? [])

    let hasBlockquote = nodes.contains { node in
      if case .blockquote = node.kind { return true }
      return false
    }
    #expect(hasBlockquote, "Should have a blockquote node")

    let codeBlocks = nodes.filter { node in
      if case .codeBlock = node.kind { return true }
      return false
    }
    #expect(codeBlocks.count == 1, "Should have 1 code block node")
  }

  @Test("Parser emits checkbox list item nodes for checkbox list inside blockquote")
  func parserEmitsCheckboxInsideBlockquote() {
    let text = "> - [ ] Do this\n> - [x] Do that"
    let converter = SourceRangeConverter(string: text)
    var parser = MarkdownParser(converter: converter)
    let doc = Document(parsing: text)
    parser.visit(doc)
    let nodes = flatten(parser.document?.blocks ?? [])

    let checkboxItems = nodes.compactMap { node -> Bool? in
      guard case .listItem(let syntax) = node.kind,
        case .checkbox(let checked) = syntax.kind
      else {
        return nil
      }
      return checked
    }

    #expect(checkboxItems.count == 2, "Should have 2 checkbox list item nodes")
    #expect(checkboxItems == [false, true], "Should preserve unchecked and checked states")
  }

  @Test("List item delimiters inside blockquote do not overlap with blockquote delimiters")
  func listItemDelimitersDoNotOverlapBlockquote() {
    // "> - Item one\n> - Item two"
    //  0 1 2       12 13 14 15
    let text = "> - Item one\n> - Item two"
    let converter = SourceRangeConverter(string: text)
    var parser = MarkdownParser(converter: converter)
    let doc = Document(parsing: text)
    parser.visit(doc)
    let nodes = flatten(parser.document?.blocks ?? [])

    let blockquoteNode = nodes.first { node in
      if case .blockquote = node.kind { return true }
      return false
    }
    let listItemNodes = nodes.filter { node in
      if case .listItem(let syntax) = node.kind, case .unordered = syntax.kind { return true }
      return false
    }

    #expect(blockquoteNode != nil, "Should have blockquote node")
    #expect(listItemNodes.count == 2, "Should have 2 list item nodes")

    // Blockquote delimiters should cover "> " prefixes
    if let bq = blockquoteNode {
      if case .blockquote(let prefixRanges) = bq.kind {
        #expect(prefixRanges.contains { $0.location == 0 && $0.length == 2 },
        "First line > should be a blockquote delimiter")
        #expect(prefixRanges.contains { $0.location == 13 && $0.length == 2 },
        "Second line > should be a blockquote delimiter")
      } else {
        #expect(Bool(false), "Expected blockquote node")
      }
    }

    // List item delimiters should cover only "- " (not "> - ")
    for li in listItemNodes {
      if case .listItem(let syntax) = li.kind {
        let markerRange = syntax.markerRange
        #expect(markerRange.location != 0 && markerRange.location != 13,
          "List item delimiter should not overlap with blockquote > prefix")
      }
    }
  }

  @Test("Nested blockquote inner node has its own delimiter")
  func nestedBlockquoteInnerDelimiter() {
    let text = "> Outer\n> > Inner"
    let converter = SourceRangeConverter(string: text)
    var parser = MarkdownParser(converter: converter)
    let doc = Document(parsing: text)
    parser.visit(doc)
    let nodes = flatten(parser.document?.blocks ?? [])

    let blockquotes = nodes.filter { node in
      if case .blockquote = node.kind { return true }
      return false
    }
    #expect(blockquotes.count >= 2)

    // The inner blockquote should have its own delimiter for the second `> `
    let innerBQ = blockquotes.first { $0.range.location > 0 }
    #expect(innerBQ != nil, "Should have inner blockquote")
    if let inner = innerBQ {
      if case .blockquote(let prefixRanges) = inner.kind {
        #expect(!prefixRanges.isEmpty,
          "Inner blockquote should have delimiter ranges for its own > prefix")
        #expect(prefixRanges.first?.location == 10,
          "Inner delimiter should be at position 10")
      } else {
        #expect(Bool(false), "Expected blockquote node")
      }
    }
  }

  // MARK: - Render spec tests

  @Test("List inside blockquote: both > and - delimiters hidden when cursor outside")
  func bothDelimitersHiddenWhenCursorOutside() {
    let text = "> - Item one\n\nBody"
    let cursorRange = NSRange(location: 15, length: 0)  // in "Body"
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // The `>` is transparent (not hidden), space after is hidden (blockquote delimiter)
    #expect(!spec.hiddenIndexes.contains(0), "> should not be hidden (transparent glyph)")
    #expect(spec.hiddenIndexes.contains(1), "space after > should be hidden")

    // The list item `-` should be replaced by bullet
    let hasBullet = !spec.bulletIndexes.isEmpty
    let dashHidden = spec.hiddenIndexes.contains(2)
    #expect(hasBullet || dashHidden,
      "List item marker should be either bullet-replaced or hidden")
  }

  @Test("Checkbox list inside blockquote uses checkbox glyphs, not bullets")
  func checkboxInsideBlockquoteUsesCheckboxGlyphs() {
    let text = "> - [ ] Do this\n> - [x] Do that\n\nBody"
    let cursorRange = NSRange(location: text.count, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    #expect(spec.uncheckedCheckboxIndexes.contains(2), "Unchecked checkbox should render at the list marker")
    #expect(spec.checkedCheckboxIndexes.contains(18), "Checked checkbox should render at the list marker")
    #expect(!spec.bulletIndexes.contains(2), "Unchecked checkbox should not render as a bullet")
    #expect(!spec.bulletIndexes.contains(18), "Checked checkbox should not render as a bullet")
  }

  @Test("Nested blockquote: inner blockquote has deeper indentation")
  func nestedBlockquoteIndentation() {
    let text = "> Outer\n> > Inner\n\nBody"
    let cursorRange = NSRange(location: 20, length: 0)  // in "Body"
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    let indentedRanges = spec.styledRanges.filter { sr in
      if let ps = sr.attributes[.paragraphStyle] as? NSParagraphStyle {
        return ps.headIndent > 0
      }
      return false
    }
    #expect(!indentedRanges.isEmpty, "Should have indented ranges for blockquote")

    let allIndents = indentedRanges.compactMap { sr -> CGFloat? in
      (sr.attributes[.paragraphStyle] as? NSParagraphStyle)?.headIndent
    }
    let maxIndent = allIndents.max() ?? 0
    let minIndent = allIndents.min() ?? 0
    #expect(maxIndent > minIndent, "Should have different indent levels: max=\(maxIndent) min=\(minIndent)")
    #expect(maxIndent >= 40, "Inner blockquote should have at least 40pt indent, got \(maxIndent)")
  }

  @Test("List inside blockquote has blockquote-offset indentation")
  func listInsideBlockquoteIndentation() {
    let text = "> - Item one\n\nBody"
    let cursorRange = NSRange(location: 15, length: 0)  // in "Body"
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // List item should have headIndent that includes the blockquote indent
    // (firstLineHeadIndent is 0 because the transparent > glyph + kern provides indent)
    let listItemStyled = spec.styledRanges.first { sr in
      if let ps = sr.attributes[.paragraphStyle] as? NSParagraphStyle {
        return ps.headIndent >= 20  // at least blockquote indent
      }
      return false
    }
    #expect(listItemStyled != nil, "List item inside blockquote should have headIndent >= blockquote indent")
  }

  // MARK: - Visual tests

  @Test("Visual: list inside blockquote with cursor at various positions")
  func visualListInsideBlockquote() {
    let markdown = "> - Item one\n> - Item two\n\nBody text"
    let initial = EditorState(markdown: markdown, selection: .cursor(30))  // in body

    let results = EditorTestHarness.run(
      name: "blockquote-nested-list-outside",
      initial: initial,
      events: [
        .setSelection(.cursor(5)),   // inside first list item
        .setSelection(.cursor(18)),  // inside second list item
        .setSelection(.cursor(30)),  // back outside
      ],
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 4)
    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }
  }

  @Test("Visual: nested blockquote with cursor at various positions")
  func visualNestedBlockquote() {
    let markdown = "> Outer quote\n> > Inner quote\n\nBody text"
    let initial = EditorState(markdown: markdown, selection: .cursor(35))  // in body

    let results = EditorTestHarness.run(
      name: "blockquote-nested-blockquote",
      initial: initial,
      events: [
        .setSelection(.cursor(5)),   // inside outer blockquote
        .setSelection(.cursor(20)),  // inside inner blockquote
        .setSelection(.cursor(35)),  // back outside
      ],
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 4)
    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }

    // Outside (step 0 and 3) should look the same
    #expect(results[0].bitmapHash == results[3].bitmapHash,
      "Same cursor position outside should produce same visual")
  }

  @Test("Visual: code block inside blockquote")
  func visualCodeBlockInsideBlockquote() {
    let markdown = "> ```\n> let x = 42\n> ```\n\nBody text"
    let initial = EditorState(markdown: markdown, selection: .cursor(28))  // in body

    let results = EditorTestHarness.run(
      name: "blockquote-nested-codeblock",
      initial: initial,
      events: [
        .setSelection(.cursor(8)),   // inside code block
        .setSelection(.cursor(28)),  // back outside
      ],
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 3)
    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }

    // Outside views should match
    #expect(results[0].bitmapHash == results[2].bitmapHash,
      "Same cursor position outside should produce same visual")
  }

  @Test("Visual: bold and italic inside blockquote")
  func visualInlineFormattingInsideBlockquote() {
    let markdown = "> **Bold** and *italic* text\n\nBody text"
    let initial = EditorState(markdown: markdown, selection: .cursor(33))  // in body

    let results = EditorTestHarness.run(
      name: "blockquote-nested-inline",
      initial: initial,
      events: [
        .setSelection(.cursor(5)),   // inside blockquote
        .setSelection(.cursor(33)),  // back outside
      ],
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 3)
    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }
  }

  @Test("Heading inside blockquote has blockquote indentation when cursor outside")
  func headingInsideBlockquoteIndentation() {
    // "### Heading\n" inside a blockquote should get the blockquote's indent
    // when cursor is outside, not start at the left margin.
    let text = "> ### Heading\n> Body\n\nOutside"
    let cursorRange = NSRange(location: 25, length: 0)  // in "Outside"
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Find the heading styled range - it should have indentation matching blockquote
    let headingStyled = spec.styledRanges.first { sr in
      if let font = sr.attributes[.font] as? NSFont {
        return font.pointSize > 16  // heading font is larger
      }
      return false
    }
    #expect(headingStyled != nil, "Should have a heading styled range")
    if let hs = headingStyled,
       let ps = hs.attributes[.paragraphStyle] as? NSParagraphStyle {
      // firstLineHeadIndent is 0 because the transparent > glyph + kern provides indent
      #expect(ps.firstLineHeadIndent == 0,
        "Heading inside blockquote should have firstLineHeadIndent == 0 (transparent > glyph), got \(ps.firstLineHeadIndent)")
      #expect(ps.headIndent >= 20,
        "Heading inside blockquote should have headIndent >= 20, got \(ps.headIndent)")
    }
  }

  @Test("Heading inside blockquote has zero indent when cursor inside")
  func headingInsideBlockquoteZeroIndentWhenCursorInside() {
    let text = "> ### Heading\n> Body\n\nOutside"
    let cursorRange = NSRange(location: 5, length: 0)  // inside the blockquote
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // When cursor is inside, heading should have zero indent (> is visible)
    let headingStyled = spec.styledRanges.first { sr in
      if let font = sr.attributes[.font] as? NSFont {
        return font.pointSize > 16
      }
      return false
    }
    #expect(headingStyled != nil, "Should have a heading styled range")
    if let hs = headingStyled,
       let ps = hs.attributes[.paragraphStyle] as? NSParagraphStyle {
      #expect(ps.firstLineHeadIndent == 0,
        "Heading inside blockquote with cursor inside should have firstLineHeadIndent == 0, got \(ps.firstLineHeadIndent)")
    }
  }

  @Test("Visual: heading inside blockquote with cursor at various positions")
  func visualHeadingInsideBlockquote() {
    let markdown = "> ### Nested Heading\n> Regular body\n> - List item\n\nOutside text"
    let initial = EditorState(markdown: markdown, selection: .cursor(55))  // in "Outside text"

    let results = EditorTestHarness.run(
      name: "blockquote-heading-nested",
      initial: initial,
      events: [
        .setSelection(.cursor(8)),   // inside the heading
        .setSelection(.cursor(30)),  // inside regular body
        .setSelection(.cursor(55)),  // back outside
      ],
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 4)
    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }

    // Outside (step 0 and 3) should look the same
    #expect(results[0].bitmapHash == results[3].bitmapHash,
      "Same cursor position outside should produce same visual")
  }

  @Test("Visual: ordered list inside blockquote with cursor at various positions")
  func visualOrderedListInsideBlockquote() {
    // Use renumbered content: > 1. one\n>    1. one, one\n>    2. one, two\n> 2. two
    let markdown = "> 1. one\n>     1. one, one\n>     2. one, two\n> 2. two\n\nBody text"
    let textLen = (markdown as NSString).length
    let initial = EditorState(markdown: markdown, selection: .cursor(textLen - 2))  // in body

    let results = EditorTestHarness.run(
      name: "blockquote-ordered-list-nested",
      initial: initial,
      events: [
        .setSelection(.cursor(5)),   // inside first ordered item
        .setSelection(.cursor(20)),  // inside nested item
        .setSelection(.cursor(textLen - 2)),  // back outside
      ],
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 4)
    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }

    // Outside (step 0 and 3) should look the same
    #expect(results[0].bitmapHash == results[3].bitmapHash,
      "Same cursor position outside should produce same visual")
  }

  @Test("List inside blockquote: > prefix at position 0 when cursor inside")
  func listInsideBlockquotePrefixAtPosition0() {
    // When cursor is inside a blockquote, the `>` prefix and leading whitespace
    // are visible characters. firstLineHeadIndent should be 0 so the `>` starts
    // at the left edge. headIndent should be positive so wrapped content aligns
    // after the list marker.
    let text = "> - Item one\n> - Item two"
    let cursorRange = NSRange(location: 5, length: 0)  // inside first item
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    let listStyledRanges = spec.styledRanges.filter { sr in
      guard let ps = sr.attributes[.paragraphStyle] as? NSParagraphStyle else { return false }
      return ps.headIndent > 0 && sr.range.location == 0
    }

    #expect(!listStyledRanges.isEmpty, "Should find list item paragraph styling inside the blockquote")

    for sr in listStyledRanges {
      guard let ps = sr.attributes[.paragraphStyle] as? NSParagraphStyle else { continue }
      #expect(
        ps.firstLineHeadIndent == 0,
        "firstLineHeadIndent should be 0 so the visible > prefix starts at the left edge, got \(ps.firstLineHeadIndent) for range \(sr.range)")
      #expect(
        ps.headIndent > 0,
        "headIndent should be positive so wrapped content aligns after the list marker, got \(ps.headIndent) for range \(sr.range)")
    }
  }

  @Test("Visual: blockquote A-Z alphabet alignment")
  func visualBlockquoteAlphabetAlignment() {
    let lines = (UnicodeScalar("A").value...UnicodeScalar("Z").value)
      .map { "> \(String(UnicodeScalar($0)!))" }
    let markdown = lines.joined(separator: "\n") + "\n\nBody text outside"
    let textLen = (markdown as NSString).length

    // Step 0: cursor inside first line (> visible)
    let initial = EditorState(markdown: markdown, selection: .cursor(3))
    let events: [EditorEvent] = [
      .setSelection(.cursor(textLen - 3)),  // cursor in body text (> hidden)
    ]

    let results = EditorTestHarness.run(
      name: "blockquote-alphabet-alignment",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 600))

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }
  }

  @Test("Visual: comprehensive nested blockquote content")
  func visualComprehensiveNestedBlockquote() {
    let markdown = """
      > This is a blockquote
      >
      > - List item inside blockquote
      > - Another item
      >
      > > Nested blockquote
      >
      > **Bold** and *italic* inside blockquote

      Body text outside
      """
    let initial = EditorState(markdown: markdown, selection: .cursor(0))

    let textLen = (markdown as NSString).length
    let events: [EditorEvent] = [
      .setSelection(.cursor(5)),                             // inside first blockquote line
      .setSelection(.cursor(min(35, textLen))),              // inside list item
      .setSelection(.cursor(min(60, textLen))),              // inside second list item
      .setSelection(.cursor(min(80, textLen))),              // inside nested blockquote
      .setSelection(.cursor(min(110, textLen))),             // inside bold/italic line
      .setSelection(.cursor(min(textLen - 5, textLen))),     // in body text outside
    ]

    let results = EditorTestHarness.run(
      name: "blockquote-comprehensive-nested",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 500))

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }
  }

  @Test("Visual: deeply nested blockquote > list > blockquote > ordered list")
  func visualDeeplyNestedBlockquoteListOrdered() {
    let markdown = """
      > Blockquote 1
      > > Blockquote 2
      > > - List A
      > >   > Blockquote 3, paragraph 1
      > >   > 1. List B, item 1
      > >   > 2. List B, item 2
      > >   > Blockquote 3, paragraph 2
      """
    let textLen = (markdown as NSString).length
    let initial = EditorState(markdown: markdown, selection: .cursor(0))

    let events: [EditorEvent] = [
      .setSelection(.cursor(0)),                             // outside everything
      .setSelection(.cursor(17)),                            // inside bq 1
      .setSelection(.cursor(34)),                            // inside bq 2
      .setSelection(.cursor(45)),                            // inside list A
      .setSelection(.cursor(70)),                            // inside bq 3 paragraph 1
      .setSelection(.cursor(100)),                           // inside list B, item 1
      .setSelection(.cursor(125)),                           // inside list B, item 2
      .setSelection(.cursor(min(textLen - 5, textLen))),     // inside bq 3 paragraph 2
    ]

    let results = EditorTestHarness.run(
      name: "deeply-nested-bq-list-bq-ordered",
      initial: initial,
      events: events,
      size: NSSize(width: 700, height: 400))

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }
  }
}
