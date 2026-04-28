import Foundation
import Testing

@testable import MarkdownEditor

/// State transition tests for checkbox list item keyboard behavior.
@Suite("Checkbox List EditorUpdate")
struct CheckboxListUpdateTests {

  // MARK: - Enter continues checkbox list

  @Test("Enter at end of checkbox item with content continues with unchecked checkbox")
  func enterContinuesCheckboxList() {
    let state = EditorState(markdown: "- [ ] Task one", selection: .cursor(15))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "- [ ] Task one\n- [ ] ")
    #expect(result.selection == .cursor(21))
  }

  @Test("Enter at end of checked item continues with unchecked checkbox")
  func enterAfterCheckedContinuesUnchecked() {
    let state = EditorState(markdown: "- [x] Done task", selection: .cursor(15))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "- [x] Done task\n- [ ] ")
    #expect(result.selection == .cursor(22))
  }

  @Test("Enter preserves indentation for nested checkbox item")
  func enterPreservesIndentation() {
    let state = EditorState(markdown: "  - [ ] Nested", selection: .cursor(14))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "  - [ ] Nested\n  - [ ] ")
    #expect(result.selection == .cursor(23))
  }

  // MARK: - Enter on empty checkbox item ends list

  @Test("Enter on empty checkbox item removes marker")
  func enterOnEmptyCheckboxRemovesMarker() {
    let state = EditorState(markdown: "- [ ] Task\n- [ ] ", selection: .cursor(17))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "- [ ] Task\n\n")
    #expect(result.selection == .cursor(11))
  }

  @Test("Enter on empty checked checkbox item removes marker")
  func enterOnEmptyCheckedCheckboxRemovesMarker() {
    let state = EditorState(markdown: "- [x] Task\n- [x] ", selection: .cursor(17))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "- [x] Task\n\n")
    #expect(result.selection == .cursor(11))
  }

  // MARK: - Backspace removes checkbox marker

  @Test("Backspace right after checkbox marker removes entire marker")
  func backspaceRemovesCheckboxMarker() {
    let state = EditorState(markdown: "- [ ] Hello", selection: .cursor(6))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    #expect(result.markdown == "Hello")
    #expect(result.selection == .cursor(0))
  }

  @Test("Backspace right after checked marker removes entire marker")
  func backspaceRemovesCheckedMarker() {
    let state = EditorState(markdown: "- [x] Hello", selection: .cursor(6))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    #expect(result.markdown == "Hello")
    #expect(result.selection == .cursor(0))
  }

  @Test("Backspace right after indented checkbox marker removes marker")
  func backspaceRemovesIndentedCheckboxMarker() {
    let state = EditorState(markdown: "  - [ ] Hello", selection: .cursor(8))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    #expect(result.markdown == "Hello")
    #expect(result.selection == .cursor(0))
  }

  @Test("Backspace in middle of checkbox content does not remove marker")
  func backspaceInCheckboxContentDoesNotRemoveMarker() {
    let state = EditorState(markdown: "- [ ] Hello", selection: .cursor(9))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    #expect(result.markdown == "- [ ] Helo")
    #expect(result.selection == .cursor(8))
  }

  // MARK: - Multi-item checkbox list sequences

  @Test("Type multiple checkbox items")
  func typeMultipleCheckboxItems() {
    var state = EditorState()
    // Type "- [ ] Task 1"
    for char in "- [ ] Task 1" {
      state = EditorUpdate.update(state, event: .insertText(String(char)))
    }
    #expect(state.markdown == "- [ ] Task 1")

    // Enter to continue list
    state = EditorUpdate.update(state, event: .insertNewline)
    #expect(state.markdown == "- [ ] Task 1\n- [ ] ")
    #expect(state.selection == .cursor(19))

    // Type "Task 2"
    for char in "Task 2" {
      state = EditorUpdate.update(state, event: .insertText(String(char)))
    }
    #expect(state.markdown == "- [ ] Task 1\n- [ ] Task 2")

    // Enter again
    state = EditorUpdate.update(state, event: .insertNewline)
    #expect(state.markdown == "- [ ] Task 1\n- [ ] Task 2\n- [ ] ")

    // Enter on empty to end list
    state = EditorUpdate.update(state, event: .insertNewline)
    #expect(state.markdown == "- [ ] Task 1\n- [ ] Task 2\n\n")
  }

  // MARK: - Mixed list (bullets + checkboxes)

  @Test("Mixed list: regular bullet then checkbox")
  func mixedListBulletThenCheckbox() {
    var state = EditorState()
    for char in "- Regular item" {
      state = EditorUpdate.update(state, event: .insertText(String(char)))
    }
    state = EditorUpdate.update(state, event: .insertNewline)
    // Should continue as regular list
    #expect(state.markdown == "- Regular item\n- ")

    // Delete the "- " and type "- [ ] " to start a checkbox
    state = EditorUpdate.update(state, event: .deleteBackward)
    #expect(state.markdown == "- Regular item\n")
    for char in "- [ ] Checkbox item" {
      state = EditorUpdate.update(state, event: .insertText(String(char)))
    }
    #expect(state.markdown == "- Regular item\n- [ ] Checkbox item")
  }
}
