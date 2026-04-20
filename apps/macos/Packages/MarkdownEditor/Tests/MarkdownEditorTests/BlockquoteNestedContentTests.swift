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

  // MARK: - Parser: verify correct node emission for nested content

  @Test("Parser emits blockquote and list item nodes for list inside blockquote")
  func parserEmitsListInsideBlockquote() {
    let text = "> - Item one\n> - Item two"
    let converter = SourceRangeConverter(string: text)
    var parser = MarkdownParser(converter: converter)
    let doc = Document(parsing: text)
    parser.visit(doc)
    let nodes = parser.nodes

    let hasBlockquote = nodes.contains { node in
      if case .blockquote = node.type { return true }
      return false
    }
    #expect(hasBlockquote, "Should have a blockquote node")

    let listItems = nodes.filter { node in
      if case .unorderedListItem = node.type { return true }
      return false
    }
    #expect(listItems.count == 2, "Should have 2 list item nodes, got \(listItems.count)")
  }

  @Test("Parser emits outer and inner blockquote nodes")
  func parserEmitsNestedBlockquote() {
    let text = "> Outer\n> > Inner"
    let converter = SourceRangeConverter(string: text)
    var parser = MarkdownParser(converter: converter)
    let doc = Document(parsing: text)
    parser.visit(doc)
    let nodes = parser.nodes

    let blockquotes = nodes.filter { node in
      if case .blockquote = node.type { return true }
      return false
    }
    #expect(blockquotes.count >= 2, "Should have at least 2 blockquote nodes (outer + inner), got \(blockquotes.count)")
  }

  @Test("Parser emits blockquote and code block nodes for code block inside blockquote")
  func parserEmitsCodeBlockInsideBlockquote() {
    let text = "> ```\n> code\n> ```"
    let converter = SourceRangeConverter(string: text)
    var parser = MarkdownParser(converter: converter)
    let doc = Document(parsing: text)
    parser.visit(doc)
    let nodes = parser.nodes

    let hasBlockquote = nodes.contains { node in
      if case .blockquote = node.type { return true }
      return false
    }
    #expect(hasBlockquote, "Should have a blockquote node")

    let codeBlocks = nodes.filter { node in
      if case .codeBlock = node.type { return true }
      return false
    }
    #expect(codeBlocks.count == 1, "Should have 1 code block node, got \(codeBlocks.count)")
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
    let nodes = parser.nodes

    let blockquoteNode = nodes.first { node in
      if case .blockquote = node.type { return true }
      return false
    }
    let listItemNodes = nodes.filter { node in
      if case .unorderedListItem = node.type { return true }
      return false
    }

    #expect(blockquoteNode != nil, "Should have blockquote node")
    #expect(listItemNodes.count == 2, "Should have 2 list item nodes")

    // Blockquote delimiters should cover "> " prefixes
    if let bq = blockquoteNode {
      #expect(bq.delimiterRanges.contains { $0.location == 0 && $0.length == 2 },
        "First line > should be a blockquote delimiter")
      #expect(bq.delimiterRanges.contains { $0.location == 13 && $0.length == 2 },
        "Second line > should be a blockquote delimiter")
    }

    // List item delimiters should cover only "- " (not "> - ")
    for li in listItemNodes {
      for delim in li.delimiterRanges {
        // Delimiter should NOT start at position 0 or 13 (those are blockquote prefixes)
        #expect(delim.location != 0 && delim.location != 13,
          "List item delimiter at \(delim) should not overlap with blockquote > prefix")
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
    let nodes = parser.nodes

    let blockquotes = nodes.filter { node in
      if case .blockquote = node.type { return true }
      return false
    }
    #expect(blockquotes.count >= 2)

    // The inner blockquote should have its own delimiter for the second `> `
    let innerBQ = blockquotes.first { $0.range.location > 0 }
    #expect(innerBQ != nil, "Should have inner blockquote")
    if let inner = innerBQ {
      #expect(!inner.delimiterRanges.isEmpty,
        "Inner blockquote should have delimiter ranges for its own > prefix")
      // The inner blockquote's delimiter should be at position 10 (the second > on line 2)
      #expect(inner.delimiterRanges.first?.location == 10,
        "Inner delimiter should be at position 10")
    }
  }

  // MARK: - Render spec tests

  @Test("List inside blockquote: both > and - delimiters hidden when cursor outside")
  func bothDelimitersHiddenWhenCursorOutside() {
    let text = "> - Item one\n\nBody"
    let cursorRange = NSRange(location: 15, length: 0)  // in "Body"
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // The `> ` prefix should be hidden (blockquote delimiter)
    #expect(spec.hiddenIndexes.contains(0), "> should be hidden")
    #expect(spec.hiddenIndexes.contains(1), "space after > should be hidden")

    // The list item `-` should be replaced by bullet
    let hasBullet = !spec.bulletIndexes.isEmpty
    let dashHidden = spec.hiddenIndexes.contains(2)
    #expect(hasBullet || dashHidden,
      "List item marker should be either bullet-replaced or hidden")
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

    // List item should have indentation that includes the blockquote indent
    let listItemStyled = spec.styledRanges.first { sr in
      if let ps = sr.attributes[.paragraphStyle] as? NSParagraphStyle {
        return ps.firstLineHeadIndent >= 20  // at least blockquote indent
      }
      return false
    }
    #expect(listItemStyled != nil, "List item inside blockquote should have indentation >= blockquote indent")
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
      #expect(ps.firstLineHeadIndent >= 20,
        "Heading inside blockquote should have indentation >= 20, got \(ps.firstLineHeadIndent)")
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
    // When cursor is inside a blockquote, the list item's paragraph style should
    // have zero firstLineHeadIndent so the `> ` prefix stays at position 0.
    let text = "> - Item one\n> - Item two"
    let cursorRange = NSRange(location: 5, length: 0)  // inside first item
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // Find the list item styled range
    let listStyled = spec.styledRanges.first { sr in
      if let ps = sr.attributes[.paragraphStyle] as? NSParagraphStyle {
        // List items have non-zero headIndent or are at position 0
        return sr.range.location == 0 || sr.range.length > 10
      }
      return false
    }
    // All styled ranges for list items should have firstLineHeadIndent == 0
    // when cursor is inside the blockquote
    for sr in spec.styledRanges {
      if let ps = sr.attributes[.paragraphStyle] as? NSParagraphStyle,
         ps.firstLineHeadIndent > 0 {
        // This is a list item with indent — it should be 0 when cursor is inside
        // the blockquote (so > stays at position 0)
        #expect(false,
          "firstLineHeadIndent should be 0 when cursor is inside blockquote, got \(ps.firstLineHeadIndent) for range \(sr.range)")
      }
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
}
