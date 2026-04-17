import AppKit
import Foundation
import Testing

@testable import MarkdownEditor

/// Tests for list bugs:
/// 1. Multi-line list items break ordinal renumbering
/// 2. Shift+Return on continuation line should produce indented continuation
/// 3. Return on continuation line should create a new list item
@Suite("List Bug Fixes")
struct ListBugFixTests {

  // MARK: - List indentation

  @Test("Nested list headIndent increases monotonically")
  func nestedHeadIndentMonotonicallyIncreases() {
    let style = MarkdownStyle.default
    var prevHead: CGFloat = 0
    for level in 1...4 {
      let attrs = style.listItemAttributes(indentLevel: level)
      let ps = attrs[.paragraphStyle] as! NSParagraphStyle
      #expect(
        ps.headIndent > prevHead,
        "headIndent at level \(level) (\(ps.headIndent)) must be > previous (\(prevHead))"
      )
      prevHead = ps.headIndent
    }
  }

  // MARK: - Bug 2: Multi-line list items break ordinal renumbering

  @Test("Multi-line ordered list item does not reset numbering")
  func multiLineOrderedListPreservesNumbering() {
    // A list with a continuation line should not break numbering.
    // Use insertText to trigger renumbering (setSelection skips it).
    let state = EditorState(
      markdown: "1. First item\n   continuation\n2. Second ite",
      selection: .cursor(44))
    let result = EditorUpdate.update(state, event: .insertText("m"))
    #expect(result.markdown == "1. First item\n   continuation\n2. Second item")
  }

  @Test("Multi-line ordered list item with multiple continuations preserves numbering")
  func multiContinuationPreservesNumbering() {
    let state = EditorState(
      markdown: "1. First\n   line 2\n   line 3\n2. Secon",
      selection: .cursor(37))
    let result = EditorUpdate.update(state, event: .insertText("d"))
    #expect(result.markdown == "1. First\n   line 2\n   line 3\n2. Second")
  }

  @Test("Blank line still resets numbering even with continuation support")
  func blankLineStillResetsNumbering() {
    let state = EditorState(
      markdown: "1. First\n\n2. Secon",
      selection: .cursor(18))
    let result = EditorUpdate.update(state, event: .insertText("d"))
    // Blank line should break the list, so "2." becomes "1."
    #expect(result.markdown == "1. First\n\n1. Second")
  }

  @Test("Non-indented non-list line resets numbering")
  func nonIndentedLineResetsNumbering() {
    let state = EditorState(
      markdown: "1. First\nplain text\n2. Secon",
      selection: .cursor(28))
    let result = EditorUpdate.update(state, event: .insertText("d"))
    // "plain text" at root indent (no leading spaces, no marker) should break the list
    #expect(result.markdown == "1. First\nplain text\n1. Second")
  }

  // MARK: - Bug 3: Shift+Return on continuation line

  @Test("Shift+Return on continuation line produces indented continuation")
  func shiftReturnOnContinuationLine() {
    // Cursor is at the end of the continuation line "  more text"
    // "- First line\n  more text" = 24 chars total, cursor at 24
    let state = EditorState(
      markdown: "- First line\n  more text",
      selection: .cursor(24))
    let result = EditorUpdate.update(state, event: .insertLineBreak)
    // Should continue with same indentation as the list item's marker width
    #expect(result.markdown == "- First line\n  more text\n  ")
    #expect(result.selection == .cursor(27))
  }

  @Test("Shift+Return on ordered list continuation line produces indented continuation")
  func shiftReturnOnOrderedContinuationLine() {
    // "1. First line\n   more text" = 26 chars total, cursor at 26
    let state = EditorState(
      markdown: "1. First line\n   more text",
      selection: .cursor(26))
    let result = EditorUpdate.update(state, event: .insertLineBreak)
    #expect(result.markdown == "1. First line\n   more text\n   ")
    #expect(result.selection == .cursor(30))
  }

  // MARK: - Bug 4: Return on continuation line creates new list item

  @Test("Return on unordered list continuation line creates new list item")
  func returnOnContinuationCreatesNewItem() {
    // "- First line\n  more text" = 24 chars total, cursor at 24
    let state = EditorState(
      markdown: "- First line\n  more text",
      selection: .cursor(24))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "- First line\n  more text\n- ")
    #expect(result.selection == .cursor(27))
  }

  @Test("Return on ordered list continuation line creates new numbered item")
  func returnOnOrderedContinuationCreatesNewItem() {
    // "1. First line\n   more text" = 26 chars total, cursor at 26
    let state = EditorState(
      markdown: "1. First line\n   more text",
      selection: .cursor(26))
    let result = EditorUpdate.update(state, event: .insertNewline)
    // Should create "2. " as the next item
    #expect(result.markdown == "1. First line\n   more text\n2. ")
    #expect(result.selection == .cursor(30))
  }

  @Test("Return in middle of continuation line splits and creates new item")
  func returnInMiddleOfContinuationSplits() {
    let state = EditorState(
      markdown: "- First line\n  more text",
      selection: .cursor(19))  // between "more" and " text"
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "- First line\n  more\n-  text")
    #expect(result.selection == .cursor(22))
  }

  @Test("Return on ordered list continuation after multi-item list gets correct number")
  func returnOnContinuationAfterMultiItemList() {
    let state = EditorState(
      markdown: "1. First\n2. Second item\n   continuation",
      selection: .cursor(39))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "1. First\n2. Second item\n   continuation\n3. ")
  }
}
