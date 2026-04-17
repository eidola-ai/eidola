import Foundation
import Testing

@testable import MarkdownEditor

/// State transition tests for Tab (indent) and Shift+Tab (outdent) behavior.
@Suite("Indent/Outdent EditorUpdate")
struct IndentOutdentUpdateTests {

  // MARK: - Tab on unordered list items

  @Test("Tab on top-level unordered list item indents by 4 spaces")
  func tabIndentsUnordered() {
    let state = EditorState(markdown: "- Item", selection: .cursor(6))
    let result = EditorUpdate.update(state, event: .indent)
    #expect(result.markdown == "    - Item")
    #expect(result.selection == .cursor(10))
  }

  @Test("Tab on top-level unordered list item with cursor at marker")
  func tabIndentsUnorderedCursorAtMarker() {
    let state = EditorState(markdown: "- Item", selection: .cursor(2))
    let result = EditorUpdate.update(state, event: .indent)
    #expect(result.markdown == "    - Item")
    #expect(result.selection == .cursor(6))
  }

  @Test("Tab on already-indented list item adds another level")
  func tabIndentsNestedUnordered() {
    let state = EditorState(markdown: "    - Item", selection: .cursor(10))
    let result = EditorUpdate.update(state, event: .indent)
    #expect(result.markdown == "        - Item")
    #expect(result.selection == .cursor(14))
  }

  @Test("Tab on unordered list item with * marker")
  func tabIndentsStarMarker() {
    let state = EditorState(markdown: "* Item", selection: .cursor(6))
    let result = EditorUpdate.update(state, event: .indent)
    #expect(result.markdown == "    * Item")
    #expect(result.selection == .cursor(10))
  }

  @Test("Tab on unordered list item with + marker")
  func tabIndentsPlusMarker() {
    let state = EditorState(markdown: "+ Item", selection: .cursor(6))
    let result = EditorUpdate.update(state, event: .indent)
    #expect(result.markdown == "    + Item")
    #expect(result.selection == .cursor(10))
  }

  // MARK: - Shift+Tab on unordered list items

  @Test("Shift+Tab on nested unordered list item outdents by 4 spaces")
  func shiftTabOutdentsUnordered() {
    let state = EditorState(markdown: "    - Item", selection: .cursor(10))
    let result = EditorUpdate.update(state, event: .outdent)
    #expect(result.markdown == "- Item")
    #expect(result.selection == .cursor(6))
  }

  @Test("Shift+Tab at top level unordered list is no-op")
  func shiftTabAtTopLevelUnorderedIsNoop() {
    let state = EditorState(markdown: "- Item", selection: .cursor(6))
    let result = EditorUpdate.update(state, event: .outdent)
    #expect(result.markdown == "- Item")
    #expect(result.selection == .cursor(6))
  }

  @Test("Shift+Tab removes only up to 4 spaces when fewer exist")
  func shiftTabRemovesFewerSpaces() {
    let state = EditorState(markdown: "  - Item", selection: .cursor(8))
    let result = EditorUpdate.update(state, event: .outdent)
    #expect(result.markdown == "- Item")
    #expect(result.selection == .cursor(6))
  }

  @Test("Shift+Tab on double-nested item outdents one level")
  func shiftTabOutdentsOneLevel() {
    let state = EditorState(markdown: "        - Item", selection: .cursor(14))
    let result = EditorUpdate.update(state, event: .outdent)
    #expect(result.markdown == "    - Item")
    #expect(result.selection == .cursor(10))
  }

  // MARK: - Tab on ordered list items

  @Test("Tab on top-level ordered list item indents and renumbers")
  func tabIndentsOrdered() {
    let state = EditorState(markdown: "1. First\n2. Second", selection: .cursor(14))
    let result = EditorUpdate.update(state, event: .indent)
    // "2. Second" becomes "    1. Second" (renumbered as new sub-list starting at 1)
    #expect(result.markdown.contains("    1. Second"))
    // First item should remain "1. First"
    #expect(result.markdown.hasPrefix("1. First\n"))
  }

  @Test("Tab on ordered list item at start indents")
  func tabIndentsOrderedAtStart() {
    let state = EditorState(markdown: "1. Item", selection: .cursor(7))
    let result = EditorUpdate.update(state, event: .indent)
    #expect(result.markdown == "    1. Item")
    #expect(result.selection == .cursor(11))
  }

  // MARK: - Shift+Tab on ordered list items

  @Test("Shift+Tab on nested ordered list item outdents and renumbers")
  func shiftTabOutdentsOrdered() {
    let state = EditorState(markdown: "1. First\n    1. Second", selection: .cursor(21))
    let result = EditorUpdate.update(state, event: .outdent)
    // Should outdent "    1. Second" to "2. Second" (renumbered)
    #expect(result.markdown == "1. First\n2. Second")
  }

  // MARK: - Tab outside list

  @Test("Tab outside list inserts spaces")
  func tabOutsideListInsertsSpaces() {
    let state = EditorState(markdown: "Hello", selection: .cursor(5))
    let result = EditorUpdate.update(state, event: .indent)
    #expect(result.markdown == "Hello    ")
    #expect(result.selection == .cursor(9))
  }

  @Test("Tab in empty document inserts spaces")
  func tabInEmptyDocumentInsertsSpaces() {
    let state = EditorState(markdown: "", selection: .cursor(0))
    let result = EditorUpdate.update(state, event: .indent)
    #expect(result.markdown == "    ")
    #expect(result.selection == .cursor(4))
  }

  @Test("Tab in middle of plain text inserts spaces")
  func tabInMiddleOfTextInsertsSpaces() {
    let state = EditorState(markdown: "Hello World", selection: .cursor(5))
    let result = EditorUpdate.update(state, event: .indent)
    #expect(result.markdown == "Hello     World")
    #expect(result.selection == .cursor(9))
  }

  // MARK: - Shift+Tab outside list

  @Test("Shift+Tab outside list is no-op")
  func shiftTabOutsideListIsNoop() {
    let state = EditorState(markdown: "Hello", selection: .cursor(5))
    let result = EditorUpdate.update(state, event: .outdent)
    #expect(result.markdown == "Hello")
    #expect(result.selection == .cursor(5))
  }

  @Test("Shift+Tab in empty document is no-op")
  func shiftTabInEmptyDocumentIsNoop() {
    let state = EditorState(markdown: "", selection: .cursor(0))
    let result = EditorUpdate.update(state, event: .outdent)
    #expect(result.markdown == "")
    #expect(result.selection == .cursor(0))
  }

  // MARK: - Multi-item list indent/outdent

  @Test("Indent second item in multi-item unordered list")
  func indentSecondItem() {
    let state = EditorState(markdown: "- First\n- Second\n- Third", selection: .cursor(16))
    let result = EditorUpdate.update(state, event: .indent)
    #expect(result.markdown == "- First\n    - Second\n- Third")
    #expect(result.selection == .cursor(20))
  }

  @Test("Outdent nested item in multi-item unordered list")
  func outdentNestedItem() {
    let state = EditorState(markdown: "- First\n    - Second\n- Third", selection: .cursor(20))
    let result = EditorUpdate.update(state, event: .outdent)
    #expect(result.markdown == "- First\n- Second\n- Third")
    #expect(result.selection == .cursor(16))
  }

  // MARK: - Cursor position edge cases

  @Test("Indent with cursor at start of list line")
  func indentCursorAtLineStart() {
    let state = EditorState(markdown: "- Item", selection: .cursor(0))
    let result = EditorUpdate.update(state, event: .indent)
    #expect(result.markdown == "    - Item")
    #expect(result.selection == .cursor(4))
  }

  @Test("Outdent with cursor at start of indented list line")
  func outdentCursorAtLineStart() {
    let state = EditorState(markdown: "    - Item", selection: .cursor(4))
    let result = EditorUpdate.update(state, event: .outdent)
    #expect(result.markdown == "- Item")
    #expect(result.selection == .cursor(0))
  }

  // MARK: - Indent/outdent preserves other lines

  @Test("Indent only affects the line with the cursor")
  func indentOnlyAffectsCursorLine() {
    let state = EditorState(markdown: "- First\n- Second\n- Third", selection: .cursor(7))
    let result = EditorUpdate.update(state, event: .indent)
    // Only the first line ("- First") should be indented
    let lines = result.markdown.components(separatedBy: "\n")
    #expect(lines[0] == "    - First")
    #expect(lines[1] == "- Second")
    #expect(lines[2] == "- Third")
  }
}
