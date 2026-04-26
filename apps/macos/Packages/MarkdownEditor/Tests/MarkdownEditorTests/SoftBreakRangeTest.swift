import Foundation
import Testing

@testable import MarkdownEditor

@Suite("Soft Line Break Normalization")
struct SoftLineBreakNormalizationTests {

  @Test("Plain paragraph soft break normalized to space")
  func plainSoftBreak() {
    let result = EditorUpdate.normalizeSoftLineBreaks(in: "Hello\nWorld")
    #expect(result == "Hello World")
  }

  @Test("Multiple soft breaks in one paragraph")
  func multipleSoftBreaks() {
    let result = EditorUpdate.normalizeSoftLineBreaks(in: "A long\nparagraph\nwith wrapping")
    #expect(result == "A long paragraph with wrapping")
  }

  @Test("Paragraph break (blank line) preserved")
  func paragraphBreakPreserved() {
    let result = EditorUpdate.normalizeSoftLineBreaks(in: "Para one\n\nPara two")
    #expect(result == "Para one\n\nPara two")
  }

  @Test("Blockquote continuation prefix stripped with soft break")
  func blockquoteSoftBreak() {
    let result = EditorUpdate.normalizeSoftLineBreaks(in: "> Hello\n> World")
    #expect(result == "> Hello World")
  }

  @Test("List item continuation normalized")
  func listContinuation() {
    let result = EditorUpdate.normalizeSoftLineBreaks(in: "- Hello\n  World")
    #expect(result == "- Hello World")
  }

  @Test("Hard break (trailing spaces) normalized and trimmed")
  func hardBreakNormalized() {
    let result = EditorUpdate.normalizeSoftLineBreaks(in: "Hello  \nWorld")
    #expect(result == "Hello World")
  }

  @Test("Separate list items not collapsed")
  func separateListItems() {
    let result = EditorUpdate.normalizeSoftLineBreaks(in: "- Item 1\n- Item 2")
    #expect(result == "- Item 1\n- Item 2")
  }

  @Test("Heading followed by paragraph not collapsed")
  func headingThenParagraph() {
    let result = EditorUpdate.normalizeSoftLineBreaks(in: "# Title\nBody text")
    #expect(result == "# Title\nBody text")
  }

  @Test("Code block content not modified")
  func codeBlockPreserved() {
    let result = EditorUpdate.normalizeSoftLineBreaks(in: "```\nline 1\nline 2\n```")
    #expect(result == "```\nline 1\nline 2\n```")
  }

  @Test("Mixed content: paragraphs with soft breaks and block separators")
  func mixedContent() {
    let input = "First line\nof para one\n\nSecond para\nstill going\n\n- List item"
    let expected = "First line of para one\n\nSecond para still going\n\n- List item"
    let result = EditorUpdate.normalizeSoftLineBreaks(in: input)
    #expect(result == expected)
  }

  @Test("Nested blockquote soft break")
  func nestedBlockquoteSoftBreak() {
    let result = EditorUpdate.normalizeSoftLineBreaks(in: "> > Hello\n> > World")
    #expect(result == "> > Hello World")
  }

  @Test("No changes returns same string")
  func noChanges() {
    let input = "Single line paragraph"
    let result = EditorUpdate.normalizeSoftLineBreaks(in: input)
    #expect(result == input)
  }
}
