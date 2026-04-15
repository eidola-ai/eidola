import Foundation
import Markdown
import Testing

@testable import MarkdownEditor

@Suite("MarkdownParser")
@MainActor
struct MarkdownParserTests {

  private func parse(_ text: String) -> [SyntaxNode] {
    let document = Document(parsing: text)
    let converter = SourceRangeConverter(string: text)
    var parser = MarkdownParser(converter: converter)
    parser.visit(document)
    return parser.nodes
  }

  @Test("Heading")
  func heading() {
    let nodes = parse("# Hello")
    let headings = nodes.filter {
      if case .heading = $0.type { return true }; return false
    }
    #expect(headings.count == 1)
    #expect(headings[0].range == NSRange(location: 0, length: 7))
    // Delimiter is "# " (2 chars)
    #expect(headings[0].delimiterRanges.count == 1)
    #expect(headings[0].delimiterRanges[0].length == 2)
  }

  @Test("Bold")
  func bold() {
    let nodes = parse("**bold**")
    let strong = nodes.filter {
      if case .strong = $0.type { return true }; return false
    }
    #expect(strong.count == 1)
    #expect(strong[0].delimiterRanges.count == 2)
    // Opening **
    #expect(strong[0].delimiterRanges[0] == NSRange(location: 0, length: 2))
    // Closing **
    #expect(strong[0].delimiterRanges[1] == NSRange(location: 6, length: 2))
    // Content: "bold"
    #expect(strong[0].contentRange == NSRange(location: 2, length: 4))
  }

  @Test("Italic")
  func italic() {
    let nodes = parse("*italic*")
    let emphasis = nodes.filter {
      if case .emphasis = $0.type { return true }; return false
    }
    #expect(emphasis.count == 1)
    #expect(emphasis[0].delimiterRanges.count == 2)
    #expect(emphasis[0].delimiterRanges[0] == NSRange(location: 0, length: 1))
    #expect(emphasis[0].delimiterRanges[1] == NSRange(location: 7, length: 1))
  }

  @Test("Inline code")
  func inlineCode() {
    let nodes = parse("`code`")
    let code = nodes.filter {
      if case .inlineCode = $0.type { return true }; return false
    }
    #expect(code.count == 1)
    #expect(code[0].contentRange == NSRange(location: 1, length: 4))
  }

  @Test("Link")
  func link() {
    let nodes = parse("[text](https://example.com)")
    let links = nodes.filter {
      if case .link = $0.type { return true }; return false
    }
    #expect(links.count == 1)
    // Content is "text" at location 1, length 4
    #expect(links[0].contentRange == NSRange(location: 1, length: 4))
    // Should have 2 delimiter ranges: [ and ](url)
    #expect(links[0].delimiterRanges.count == 2)
  }

  @Test("Strikethrough")
  func strikethrough() {
    let nodes = parse("~~struck~~")
    let struck = nodes.filter {
      if case .strikethrough = $0.type { return true }; return false
    }
    #expect(struck.count == 1)
    #expect(struck[0].delimiterRanges.count == 2)
    #expect(struck[0].delimiterRanges[0] == NSRange(location: 0, length: 2))
    #expect(struck[0].delimiterRanges[1] == NSRange(location: 8, length: 2))
  }

  @Test("Multiple constructs")
  func multipleConstruts() {
    let text = "# Title\n\nSome **bold** and *italic* text."
    let nodes = parse(text)

    let types = nodes.map { node -> String in
      switch node.type {
      case .heading: return "heading"
      case .strong: return "strong"
      case .emphasis: return "emphasis"
      default: return "other"
      }
    }

    #expect(types.contains("heading"))
    #expect(types.contains("strong"))
    #expect(types.contains("emphasis"))
  }
}
