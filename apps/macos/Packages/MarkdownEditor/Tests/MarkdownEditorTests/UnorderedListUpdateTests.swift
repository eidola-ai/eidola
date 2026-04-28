import Foundation
import Testing

@testable import MarkdownEditor

/// State transition tests for unordered list keyboard behavior.
@Suite("Unordered List EditorUpdate")
struct UnorderedListUpdateTests {

  // MARK: - Enter continues list

  @Test("Enter at end of list item with content continues list")
  func enterContinuesList() {
    let state = EditorState(markdown: "- Hello", selection: .cursor(7))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "- Hello\n- ")
    #expect(result.selection == .cursor(10))
  }

  @Test("Enter continues list with * marker")
  func enterContinuesListStar() {
    let state = EditorState(markdown: "* Hello", selection: .cursor(7))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "* Hello\n* ")
    #expect(result.selection == .cursor(10))
  }

  @Test("Enter continues list with + marker")
  func enterContinuesListPlus() {
    let state = EditorState(markdown: "+ Hello", selection: .cursor(7))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "+ Hello\n+ ")
    #expect(result.selection == .cursor(10))
  }

  @Test("Enter in middle of list item splits text and continues list")
  func enterSplitsListItem() {
    let state = EditorState(markdown: "- Hello World", selection: .cursor(7))
    let result = EditorUpdate.update(state, event: .insertNewline)
    // The space after "Hello" remains as the first char on the new line
    #expect(result.markdown == "- Hello\n-  World")
    #expect(result.selection == .cursor(10))
  }

  @Test("Enter preserves indentation for nested list item")
  func enterPreservesIndentation() {
    let state = EditorState(markdown: "  - Hello", selection: .cursor(9))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "  - Hello\n  - ")
    #expect(result.selection == .cursor(14))
  }

  // MARK: - Enter on empty list item ends list

  @Test("Enter on empty list item removes marker")
  func enterOnEmptyListItemRemovesMarker() {
    let state = EditorState(markdown: "- Hello\n- ", selection: .cursor(10))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "- Hello\n\n")
    #expect(result.selection == .cursor(8))
  }

  @Test("Enter on empty * list item removes marker")
  func enterOnEmptyStarListItemRemovesMarker() {
    let state = EditorState(markdown: "* Hello\n* ", selection: .cursor(10))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "* Hello\n\n")
    #expect(result.selection == .cursor(8))
  }

  @Test("Enter on empty indented list item removes marker")
  func enterOnEmptyIndentedListItemRemovesMarker() {
    let state = EditorState(markdown: "- Parent\n  - ", selection: .cursor(13))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "- Parent\n\n")
    #expect(result.selection == .cursor(9))
  }

  // MARK: - Backspace removes list marker

  @Test("Backspace right after list marker removes entire marker")
  func backspaceRemovesMarker() {
    let state = EditorState(markdown: "- Hello", selection: .cursor(2))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    #expect(result.markdown == "Hello")
    #expect(result.selection == .cursor(0))
  }

  @Test("Backspace right after * marker removes entire marker")
  func backspaceRemovesStarMarker() {
    let state = EditorState(markdown: "* Hello", selection: .cursor(2))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    #expect(result.markdown == "Hello")
    #expect(result.selection == .cursor(0))
  }

  @Test("Backspace right after indented marker removes marker but preserves content")
  func backspaceRemovesIndentedMarker() {
    let state = EditorState(markdown: "  - Hello", selection: .cursor(4))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    #expect(result.markdown == "Hello")
    #expect(result.selection == .cursor(0))
  }

  @Test("Backspace in middle of list content does not remove marker")
  func backspaceInContentDoesNotRemoveMarker() {
    let state = EditorState(markdown: "- Hello", selection: .cursor(5))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    #expect(result.markdown == "- Helo")
    #expect(result.selection == .cursor(4))
  }

  @Test("Backspace at start of list item does nothing (pos 0)")
  func backspaceAtStartOfListItem() {
    let state = EditorState(markdown: "- Hello", selection: .cursor(0))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    #expect(result.markdown == "- Hello")
    #expect(result.selection == .cursor(0))
  }

  // MARK: - Multi-item list sequences

  @Test("Type multiple list items")
  func typeMultipleListItems() {
    var state = EditorState()
    // Type "- Item 1"
    for char in "- Item 1" {
      state = EditorUpdate.update(state, event: .insertText(String(char)))
    }
    #expect(state.markdown == "- Item 1")

    // Enter to continue list
    state = EditorUpdate.update(state, event: .insertNewline)
    #expect(state.markdown == "- Item 1\n- ")
    #expect(state.selection == .cursor(11))

    // Type "Item 2"
    for char in "Item 2" {
      state = EditorUpdate.update(state, event: .insertText(String(char)))
    }
    #expect(state.markdown == "- Item 1\n- Item 2")

    // Enter again
    state = EditorUpdate.update(state, event: .insertNewline)
    #expect(state.markdown == "- Item 1\n- Item 2\n- ")

    // Enter on empty to end list
    state = EditorUpdate.update(state, event: .insertNewline)
    #expect(state.markdown == "- Item 1\n- Item 2\n\n")
  }
}
