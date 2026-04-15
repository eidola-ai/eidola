import Foundation
import Markdown
import Testing

@testable import MarkdownEditor

@Suite("SourceRangeConverter")
struct SourceRangeConverterTests {

  @Test("ASCII single line")
  func asciiSingleLine() {
    let text = "Hello, world!"
    let converter = SourceRangeConverter(string: text)

    // Line 1, column 1 = offset 0
    let offset = converter.utf16Offset(
      from: SourceLocation(line: 1, column: 1, source: nil))
    #expect(offset == 0)

    // Line 1, column 8 = offset 7
    let offset2 = converter.utf16Offset(
      from: SourceLocation(line: 1, column: 8, source: nil))
    #expect(offset2 == 7)
  }

  @Test("Multi-line")
  func multiLine() {
    let text = "Line one\nLine two\nLine three"
    let converter = SourceRangeConverter(string: text)

    // Line 2, column 1 = offset 9 ("Line one\n" = 9 chars)
    let offset = converter.utf16Offset(
      from: SourceLocation(line: 2, column: 1, source: nil))
    #expect(offset == 9)

    // Line 3, column 6 = offset 23
    let offset2 = converter.utf16Offset(
      from: SourceLocation(line: 3, column: 6, source: nil))
    #expect(offset2 == 23)
  }

  @Test("Emoji (multi-byte UTF-8)")
  func emoji() {
    // "Hi 👋 there"
    // 👋 is 4 bytes UTF-8, 2 code units UTF-16
    let text = "Hi 👋 there"
    let converter = SourceRangeConverter(string: text)

    // Column after the emoji space: "Hi 👋 " = 3 + 4 + 1 = 8 bytes UTF-8
    // In UTF-16: 3 + 2 + 1 = 6
    let offset = converter.utf16Offset(
      from: SourceLocation(line: 1, column: 9, source: nil))
    #expect(offset == 6)
  }

  @Test("nsRange from SourceRange")
  func nsRangeConversion() {
    let text = "# Hello\n\nWorld"
    let converter = SourceRangeConverter(string: text)

    let sourceRange = SourceLocation(line: 1, column: 1, source: nil)..<SourceLocation(
      line: 1, column: 8, source: nil)
    let range = converter.nsRange(from: sourceRange)
    #expect(range == NSRange(location: 0, length: 7))
  }

  @Test("Round-trip with swift-markdown parsing")
  func roundTripWithParsing() {
    let text = "This is **bold** text"
    let document = Document(parsing: text)
    let converter = SourceRangeConverter(string: text)

    // Find the Strong node
    var strongRange: NSRange?
    for block in document.children {
      for inline in block.children {
        if let strong = inline as? Strong, let sr = strong.range {
          strongRange = converter.nsRange(from: sr)
        }
      }
    }

    #expect(strongRange != nil)
    // "**bold**" starts at index 8, length 8
    #expect(strongRange == NSRange(location: 8, length: 8))
  }
}
