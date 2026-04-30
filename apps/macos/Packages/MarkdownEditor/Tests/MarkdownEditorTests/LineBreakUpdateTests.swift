import Foundation
import Testing

@testable import MarkdownEditor

/// State transition tests for Shift+Return (insertLineBreak) behavior.
@Suite("Line Break EditorUpdate")
struct LineBreakUpdateTests {

  // MARK: - Shift+Return in unordered list

  @Test("Shift+Return in unordered list item adds indented continuation")
  func shiftReturnUnorderedList() {
    let state = EditorState(markdown: "- Hello", selection: .cursor(7))
    let result = EditorUpdate.update(state, event: .insertLineBreak)
    #expect(result.markdown == "- Hello\n  ")  // 2 spaces = marker width "- "
    #expect(result.selection == .cursor(10))
  }

  @Test("Shift+Return in ordered list item adds indented continuation")
  func shiftReturnOrderedList() {
    let state = EditorState(markdown: "1. Hello", selection: .cursor(8))
    let result = EditorUpdate.update(state, event: .insertLineBreak)
    #expect(result.markdown == "1. Hello\n   ")  // 3 spaces = marker width "1. "
    #expect(result.selection == .cursor(12))
  }

  @Test("Shift+Return outside list is a soft break")
  func shiftReturnOutsideList() {
    // Per the editor's hybrid newline policy: Enter inserts a paragraph
    // break (`\n\n`), Shift+Enter inserts a single `\n` soft break that
    // the renderer displays as a visible in-paragraph line break.
    let state = EditorState(markdown: "Hello", selection: .cursor(5))
    let result = EditorUpdate.update(state, event: .insertLineBreak)
    #expect(result.markdown == "Hello\n")
    #expect(result.selection == .cursor(6))
  }

  @Test("Shift+Return in middle of list item content splits with indent")
  func shiftReturnMiddleOfContent() {
    let state = EditorState(markdown: "- Hello World", selection: .cursor(7))
    let result = EditorUpdate.update(state, event: .insertLineBreak)
    #expect(result.markdown == "- Hello\n   World")
    #expect(result.selection == .cursor(10))
  }

  @Test("Regular Enter still continues list (not continuation)")
  func regularEnterStillContinuesList() {
    let state = EditorState(markdown: "- Hello", selection: .cursor(7))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "- Hello\n- ")
  }

  // MARK: - Double-digit ordered list

  @Test("Shift+Return in double-digit ordered list item uses correct indent")
  func shiftReturnDoubleDigitOrderedList() {
    let state = EditorState(markdown: "10. Hello", selection: .cursor(9))
    let result = EditorUpdate.update(state, event: .insertLineBreak)
    // "10. " is 4 chars wide, so 4 spaces of indent are inserted.
    // However, renumbering normalizes standalone "10." to "1.", changing
    // the marker from 4 chars to 3 chars. The 4-space indent remains,
    // which is still valid for markdown continuation. Cursor shifts left
    // by 1 due to the number shortening.
    #expect(result.markdown == "1. Hello\n    ")
    #expect(result.selection == .cursor(13))
  }

  // MARK: - Indented list items

  @Test("Shift+Return in indented unordered list preserves full marker width")
  func shiftReturnIndentedUnorderedList() {
    let state = EditorState(markdown: "  - Hello", selection: .cursor(9))
    let result = EditorUpdate.update(state, event: .insertLineBreak)
    // "  - " is 4 chars, so 4 spaces indent
    #expect(result.markdown == "  - Hello\n    ")
    #expect(result.selection == .cursor(14))
  }

  @Test("Shift+Return in indented ordered list preserves full marker width")
  func shiftReturnIndentedOrderedList() {
    let state = EditorState(markdown: "  1. Hello", selection: .cursor(10))
    let result = EditorUpdate.update(state, event: .insertLineBreak)
    // "  1. " is 5 chars, so 5 spaces indent
    #expect(result.markdown == "  1. Hello\n     ")
    #expect(result.selection == .cursor(16))
  }

  // MARK: - Selection replacement

  @Test("Shift+Return with selection replaces selection then inserts continuation")
  func shiftReturnWithSelection() {
    let state = EditorState(
      markdown: "- Hello World", selection: .range(anchor: 7, head: 13))
    let result = EditorUpdate.update(state, event: .insertLineBreak)
    // "World" is deleted, then newline + 2 spaces inserted after "- Hello"
    #expect(result.markdown == "- Hello\n  ")
    #expect(result.selection == .cursor(10))
  }

  // MARK: - Heading (not a list)

  @Test("Shift+Return in heading is a soft break")
  func shiftReturnInHeading() {
    // Hybrid newline policy: Shift+Enter is always a soft break, even in a
    // heading. (CommonMark allows soft breaks inside ATX headings only at
    // the visual-line level; the renderer's behavior for "soft break inside
    // a heading" is governed by lineBreakIndexes the same way as body text.)
    let state = EditorState(markdown: "# Hello", selection: .cursor(7))
    let result = EditorUpdate.update(state, event: .insertLineBreak)
    #expect(result.markdown == "# Hello\n")
    #expect(result.selection == .cursor(8))
  }

  // MARK: - Empty document

  @Test("Shift+Return in empty document inserts newline")
  func shiftReturnEmptyDocument() {
    let state = EditorState()
    let result = EditorUpdate.update(state, event: .insertLineBreak)
    #expect(result.markdown == "\n")
    #expect(result.selection == .cursor(1))
  }

  // MARK: - Star and plus markers

  @Test("Shift+Return with * marker")
  func shiftReturnStarMarker() {
    let state = EditorState(markdown: "* Hello", selection: .cursor(7))
    let result = EditorUpdate.update(state, event: .insertLineBreak)
    #expect(result.markdown == "* Hello\n  ")
    #expect(result.selection == .cursor(10))
  }

  @Test("Shift+Return with + marker")
  func shiftReturnPlusMarker() {
    let state = EditorState(markdown: "+ Hello", selection: .cursor(7))
    let result = EditorUpdate.update(state, event: .insertLineBreak)
    #expect(result.markdown == "+ Hello\n  ")
    #expect(result.selection == .cursor(10))
  }

  // MARK: - Multi-line continuation sequence

  @Test("Multiple Shift+Returns build multi-line list item")
  func multipleShiftReturns() {
    var state = EditorState()
    // Type "- Line 1"
    for char in "- Line 1" {
      state = EditorUpdate.update(state, event: .insertText(String(char)))
    }
    #expect(state.markdown == "- Line 1")

    // Shift+Return to continue same item
    state = EditorUpdate.update(state, event: .insertLineBreak)
    #expect(state.markdown == "- Line 1\n  ")
    #expect(state.selection == .cursor(11))

    // Type "Line 2"
    for char in "Line 2" {
      state = EditorUpdate.update(state, event: .insertText(String(char)))
    }
    #expect(state.markdown == "- Line 1\n  Line 2")

    // Regular Enter on a continuation line creates a new list item, since
    // the continuation line "  Line 2" belongs to the parent "- " item.
    state = EditorUpdate.update(state, event: .insertNewline)
    #expect(state.markdown == "- Line 1\n  Line 2\n- ")
  }
}
