import Foundation
import Testing

@testable import MarkdownEditor

/// State transition tests for blockquote keyboard behavior.
@Suite("Blockquote EditorUpdate")
struct BlockquoteUpdateTests {

  // MARK: - Enter continues blockquote

  @Test("Enter at end of blockquote line with content continues blockquote")
  func enterContinuesBlockquote() {
    let state = EditorState(markdown: "> Hello", selection: .cursor(7))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "> Hello\n> ")
    #expect(result.selection == .cursor(10))
  }

  @Test("Enter in middle of blockquote line splits and continues")
  func enterSplitsBlockquote() {
    let state = EditorState(markdown: "> Hello World", selection: .cursor(7))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "> Hello\n>  World")
    #expect(result.selection == .cursor(10))
  }

  // MARK: - Enter on empty blockquote ends blockquote

  @Test("Enter on empty blockquote line removes prefix")
  func enterOnEmptyBlockquoteRemovesPrefix() {
    let state = EditorState(markdown: "> Hello\n> ", selection: .cursor(10))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "> Hello\n\n")
    #expect(result.selection == .cursor(8))
  }

  // MARK: - Backspace removes blockquote prefix

  @Test("Backspace right after > prefix removes entire prefix")
  func backspaceRemovesBlockquotePrefix() {
    let state = EditorState(markdown: "> Hello", selection: .cursor(2))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    #expect(result.markdown == "Hello")
    #expect(result.selection == .cursor(0))
  }

  @Test("Backspace in middle of blockquote content does not remove prefix")
  func backspaceInContentDoesNotRemovePrefix() {
    let state = EditorState(markdown: "> Hello", selection: .cursor(5))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    #expect(result.markdown == "> Helo")
    #expect(result.selection == .cursor(4))
  }

  // MARK: - Shift+Return continues blockquote

  @Test("Shift+Return in blockquote continues with > prefix")
  func shiftReturnContinuesBlockquote() {
    let state = EditorState(markdown: "> Hello", selection: .cursor(7))
    let result = EditorUpdate.update(state, event: .insertLineBreak)
    #expect(result.markdown == "> Hello\n> ")
    #expect(result.selection == .cursor(10))
  }

  @Test("Shift+Return in middle of blockquote splits with > prefix")
  func shiftReturnMiddleOfBlockquote() {
    let state = EditorState(markdown: "> Hello World", selection: .cursor(7))
    let result = EditorUpdate.update(state, event: .insertLineBreak)
    #expect(result.markdown == "> Hello\n>  World")
    #expect(result.selection == .cursor(10))
  }

  // MARK: - Multi-line blockquote

  @Test("Enter at end of second blockquote line continues")
  func enterContinuesMultiLineBlockquote() {
    let state = EditorState(markdown: "> Line one\n> Line two", selection: .cursor(21))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "> Line one\n> Line two\n> ")
    #expect(result.selection == .cursor(24))
  }

  // MARK: - No interference with list behavior

  @Test("Enter on unordered list item still continues list (not blockquote)")
  func enterOnListItemNotBlockquote() {
    let state = EditorState(markdown: "- Hello", selection: .cursor(7))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "- Hello\n- ")
    #expect(result.selection == .cursor(10))
  }

  @Test("Enter on ordered list item still continues list (not blockquote)")
  func enterOnOrderedListItemNotBlockquote() {
    let state = EditorState(markdown: "1. Hello", selection: .cursor(8))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "1. Hello\n2. ")
    #expect(result.selection == .cursor(12))
  }

  // MARK: - Typing flow

  @Test("Type multiple blockquote lines with Enter continuation")
  func typeMultipleBlockquoteLines() {
    var state = EditorState()
    // Type "> Line 1"
    for char in "> Line 1" {
      state = EditorUpdate.update(state, event: .insertText(String(char)))
    }
    #expect(state.markdown == "> Line 1")

    // Enter to continue blockquote
    state = EditorUpdate.update(state, event: .insertNewline)
    #expect(state.markdown == "> Line 1\n> ")
    #expect(state.selection == .cursor(11))

    // Type "Line 2"
    for char in "Line 2" {
      state = EditorUpdate.update(state, event: .insertText(String(char)))
    }
    #expect(state.markdown == "> Line 1\n> Line 2")

    // Enter again
    state = EditorUpdate.update(state, event: .insertNewline)
    #expect(state.markdown == "> Line 1\n> Line 2\n> ")

    // Enter on empty to end blockquote
    state = EditorUpdate.update(state, event: .insertNewline)
    #expect(state.markdown == "> Line 1\n> Line 2\n\n")
  }
}
