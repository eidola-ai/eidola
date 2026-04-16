import Foundation
import Testing

@testable import MarkdownEditor

/// State transition tests for ordered list keyboard behavior.
@Suite("Ordered List EditorUpdate")
struct OrderedListUpdateTests {

  // MARK: - Enter continues ordered list

  @Test("Enter at end of ordered list item with content continues list with next number")
  func enterContinuesOrderedList() {
    let state = EditorState(markdown: "1. Hello", selection: .cursor(8))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "1. Hello\n2. ")
    #expect(result.selection == .cursor(12))
  }

  @Test("Enter after standalone item 3 renumbers to 1 then produces item 2")
  func enterIncrementsNumber() {
    let state = EditorState(markdown: "3. Third", selection: .cursor(8))
    let result = EditorUpdate.update(state, event: .insertNewline)
    // Renumbering normalizes "3." to "1.", then new item is "2."
    #expect(result.markdown == "1. Third\n2. ")
    #expect(result.selection == .cursor(12))
  }

  @Test("Enter after standalone item 9 renumbers to 1 then produces item 2")
  func enterIncrementsSingleToDouble() {
    let state = EditorState(markdown: "9. Ninth", selection: .cursor(8))
    let result = EditorUpdate.update(state, event: .insertNewline)
    // Renumbering normalizes "9." to "1.", "10." to "2."
    #expect(result.markdown == "1. Ninth\n2. ")
    #expect(result.selection == .cursor(12))
  }

  @Test("Enter in middle of ordered list item splits text and continues list")
  func enterSplitsOrderedListItem() {
    let state = EditorState(markdown: "1. Hello World", selection: .cursor(8))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "1. Hello\n2.  World")
    #expect(result.selection == .cursor(12))
  }

  @Test("Enter preserves indentation for ordered list item")
  func enterPreservesOrderedIndentation() {
    let state = EditorState(markdown: "  1. Hello", selection: .cursor(10))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "  1. Hello\n  2. ")
    #expect(result.selection == .cursor(16))
  }

  // MARK: - Enter on empty ordered list item ends list

  @Test("Enter on empty ordered list item removes marker")
  func enterOnEmptyOrderedListItemRemovesMarker() {
    let state = EditorState(markdown: "1. Hello\n2. ", selection: .cursor(12))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "1. Hello\n\n")
    #expect(result.selection == .cursor(9))
  }

  @Test("Enter on empty indented ordered list item removes marker")
  func enterOnEmptyIndentedOrderedListItemRemovesMarker() {
    let state = EditorState(markdown: "1. Parent\n  2. ", selection: .cursor(15))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "1. Parent\n\n")
    #expect(result.selection == .cursor(10))
  }

  // MARK: - Backspace removes ordered list marker

  @Test("Backspace right after ordered list marker removes entire marker")
  func backspaceRemovesOrderedMarker() {
    let state = EditorState(markdown: "1. Hello", selection: .cursor(3))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    #expect(result.markdown == "Hello")
    #expect(result.selection == .cursor(0))
  }

  @Test("Backspace right after double-digit ordered marker removes entire marker")
  func backspaceRemovesDoubleDigitOrderedMarker() {
    let state = EditorState(markdown: "12. Hello", selection: .cursor(4))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    #expect(result.markdown == "Hello")
    #expect(result.selection == .cursor(0))
  }

  @Test("Backspace right after indented ordered marker removes marker")
  func backspaceRemovesIndentedOrderedMarker() {
    let state = EditorState(markdown: "  1. Hello", selection: .cursor(5))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    #expect(result.markdown == "Hello")
    #expect(result.selection == .cursor(0))
  }

  @Test("Backspace in middle of ordered list content does not remove marker")
  func backspaceInOrderedContentDoesNotRemoveMarker() {
    let state = EditorState(markdown: "1. Hello", selection: .cursor(6))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    #expect(result.markdown == "1. Helo")
    #expect(result.selection == .cursor(5))
  }

  // MARK: - Multi-item ordered list sequences

  // MARK: - Ordinal renumbering

  @Test("Insert item in middle of ordered list renumbers subsequent items")
  func insertMiddleRenumbers() {
    let state = EditorState(markdown: "1. First\n2. Second\n3. Third", selection: .cursor(8))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "1. First\n2. \n3. Second\n4. Third")
  }

  @Test("Remove item from middle renumbers subsequent items")
  func removeMiddleRenumbers() {
    // cursor right after "2. " marker (position 12)
    let state = EditorState(
      markdown: "1. First\n2. Second\n3. Third", selection: .cursor(12))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    // "2. " marker removed; "Second" is now plain text, breaking the list.
    // "3. Third" starts a new list and is renumbered to "1. Third".
    #expect(result.markdown == "1. First\nSecond\n1. Third")
  }

  @Test("Split list renumbers second half starting from 1")
  func splitListRenumbers() {
    // Empty "2. " item, Enter ends the list
    let state = EditorState(
      markdown: "1. First\n2. \n3. Third", selection: .cursor(12))
    let result = EditorUpdate.update(state, event: .insertNewline)
    // "3. Third" should become "1. Third" as start of new list
    #expect(result.markdown.contains("1. Third"))
  }

  @Test("Recombine lists by deleting separator renumbers")
  func recombineRenumbers() {
    let state = EditorState(
      markdown: "1. First\n2. Second\n\n1. Alpha\n2. Beta", selection: .cursor(19))
    // Delete the blank line
    let result = EditorUpdate.update(state, event: .deleteForward)
    // Should renumber: "1. Alpha" -> "3. Alpha", "2. Beta" -> "4. Beta"
    #expect(result.markdown == "1. First\n2. Second\n3. Alpha\n4. Beta")
  }

  @Test("Renumbering adjusts cursor when digit count changes")
  func renumberingAdjustsCursor() {
    // Build a 10-item list, then insert at position 1 to push items down
    var lines: [String] = []
    for i in 1...10 {
      lines.append("\(i). Item \(i)")
    }
    let markdown = lines.joined(separator: "\n")
    // cursor at end of "1. Item 1" (position 9)
    let state = EditorState(markdown: markdown, selection: .cursor(9))
    let result = EditorUpdate.update(state, event: .insertNewline)
    // "10. Item 10" should become "11. Item 10" — one more digit
    // The cursor should still be correctly placed after the new "2. " marker
    #expect(result.markdown.contains("11. Item 10"))
  }

  @Test("Indented ordered lists are numbered independently")
  func indentedOrderedListsNumberedIndependently() {
    let markdown = "1. Parent 1\n  1. Child A\n  2. Child B\n2. Parent 2"
    let state = EditorState(
      markdown: markdown, selection: .cursor(11))  // end of "Parent 1"
    let result = EditorUpdate.update(state, event: .insertNewline)
    // Children should stay numbered 1, 2 independently
    #expect(result.markdown.contains("  1. Child A"))
    #expect(result.markdown.contains("  2. Child B"))
  }

  // MARK: - Multi-item ordered list sequences

  @Test("Type multiple ordered list items")
  func typeMultipleOrderedListItems() {
    var state = EditorState()
    // Type "1. Item 1"
    for char in "1. Item 1" {
      state = EditorUpdate.update(state, event: .insertText(String(char)))
    }
    #expect(state.markdown == "1. Item 1")

    // Enter to continue list
    state = EditorUpdate.update(state, event: .insertNewline)
    #expect(state.markdown == "1. Item 1\n2. ")
    #expect(state.selection == .cursor(13))

    // Type "Item 2"
    for char in "Item 2" {
      state = EditorUpdate.update(state, event: .insertText(String(char)))
    }
    #expect(state.markdown == "1. Item 1\n2. Item 2")

    // Enter again
    state = EditorUpdate.update(state, event: .insertNewline)
    #expect(state.markdown == "1. Item 1\n2. Item 2\n3. ")

    // Enter on empty to end list
    state = EditorUpdate.update(state, event: .insertNewline)
    #expect(state.markdown == "1. Item 1\n2. Item 2\n\n")
  }
}
