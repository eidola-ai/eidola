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

  // MARK: - Lists inside blockquotes

  @Test("Enter on list item inside blockquote continues with blockquote prefix + list marker")
  func enterContinuesListInsideBlockquote() {
    let state = EditorState(markdown: "> - Item one", selection: .cursor(12))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "> - Item one\n> - ")
    #expect(result.selection == .cursor(17))
  }

  @Test("Enter on empty list item inside blockquote removes list marker but keeps blockquote")
  func enterOnEmptyListInsideBlockquote() {
    let state = EditorState(markdown: "> - Item one\n> - ", selection: .cursor(17))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "> - Item one\n> \n")
    #expect(result.selection == .cursor(15))
  }

  @Test("Enter on empty blockquote after removing list marker ends blockquote")
  func enterOnEmptyBlockquoteAfterList() {
    let state = EditorState(markdown: "> - Item one\n> ", selection: .cursor(15))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "> - Item one\n\n")
    #expect(result.selection == .cursor(13))
  }

  @Test("Backspace after list marker inside blockquote removes marker, keeps blockquote prefix")
  func backspaceRemovesListMarkerInsideBlockquote() {
    let state = EditorState(markdown: "> - Item", selection: .cursor(4))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    #expect(result.markdown == "> Item")
    #expect(result.selection == .cursor(2))
  }

  @Test("Enter on ordered list item inside blockquote continues with incremented number")
  func enterContinuesOrderedListInsideBlockquote() {
    let state = EditorState(markdown: "> 1. First", selection: .cursor(10))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "> 1. First\n> 2. ")
    #expect(result.selection == .cursor(16))
  }

  @Test("Enter on empty ordered list inside blockquote removes marker keeps prefix")
  func enterOnEmptyOrderedListInsideBlockquote() {
    let state = EditorState(markdown: "> 1. First\n> 2. ", selection: .cursor(16))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "> 1. First\n> \n")
    #expect(result.selection == .cursor(13))
  }

  @Test("Backspace after ordered marker inside blockquote removes marker keeps prefix")
  func backspaceRemovesOrderedMarkerInsideBlockquote() {
    let state = EditorState(markdown: "> 1. Item", selection: .cursor(5))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    #expect(result.markdown == "> Item")
    #expect(result.selection == .cursor(2))
  }

  @Test("Shift+Return on list item inside blockquote creates continuation line")
  func shiftReturnListInsideBlockquote() {
    let state = EditorState(markdown: "> - Item one", selection: .cursor(12))
    let result = EditorUpdate.update(state, event: .insertLineBreak)
    // Continuation line should have blockquote prefix + indent (matching marker width)
    #expect(result.markdown.hasPrefix("> - Item one\n> "))
  }

  @Test("Enter on checkbox list inside blockquote continues with unchecked checkbox")
  func enterContinuesCheckboxInsideBlockquote() {
    let state = EditorState(markdown: "> - [x] Done", selection: .cursor(12))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "> - [x] Done\n> - [ ] ")
    #expect(result.selection == .cursor(21))
  }

  @Test("Backspace after blockquote prefix removes one level")
  func backspaceRemovesOneBlockquoteLevel() {
    let state = EditorState(markdown: "> > Inner", selection: .cursor(4))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    #expect(result.markdown == "> Inner")
    #expect(result.selection == .cursor(2))
  }

  // MARK: - Ordered list renumbering inside blockquotes

  @Test("Ordered list inside blockquote is renumbered sequentially")
  func orderedListInsideBlockquoteRenumbered() {
    // Typing "5. " inside a blockquote: should be renumbered to "1. "
    let state = EditorState(markdown: "> 5. Item", selection: .cursor(9))
    let result = EditorUpdate.postProcess(state)
    #expect(result.markdown == "> 1. Item")
  }

  @Test("Multiple ordered items inside blockquote are renumbered")
  func multipleOrderedItemsInsideBlockquoteRenumbered() {
    let state = EditorState(
      markdown: "> 1. First\n> 5. Second\n> 9. Third",
      selection: .cursor(34))
    let result = EditorUpdate.postProcess(state)
    #expect(result.markdown == "> 1. First\n> 2. Second\n> 3. Third")
  }

  @Test("Nested ordered list inside blockquote is renumbered independently")
  func nestedOrderedListInsideBlockquoteRenumbered() {
    let state = EditorState(
      markdown: "> 1. one\n>     2. one, one\n>     3. one, two\n> 4. two",
      selection: .cursor(52))
    let result = EditorUpdate.postProcess(state)
    #expect(result.markdown == "> 1. one\n>     1. one, one\n>     2. one, two\n> 2. two")
  }

  @Test("Ordered lists at different blockquote depths are renumbered independently")
  func orderedListsDifferentDepthsRenumbered() {
    let state = EditorState(
      markdown: "> 3. Outer\n> > 5. Inner\n> > 7. Inner two",
      selection: .cursor(40))
    let result = EditorUpdate.postProcess(state)
    #expect(result.markdown == "> 1. Outer\n> > 1. Inner\n> > 2. Inner two")
  }

  // MARK: - Space injection after bare `>`

  @Test("Typing a character after bare > injects a space")
  func typingAfterBareGtInjectsSpace() {
    let state = EditorState(markdown: ">", selection: .cursor(1))
    let result = EditorUpdate.update(state, event: .insertText("H"))
    #expect(result.markdown == "> H")
    #expect(result.selection == .cursor(3))
  }

  @Test("Typing a space after bare > does not double-inject")
  func typingSpaceAfterBareGtNoDoubleInject() {
    let state = EditorState(markdown: ">", selection: .cursor(1))
    let result = EditorUpdate.update(state, event: .insertText(" "))
    #expect(result.markdown == "> ")
    #expect(result.selection == .cursor(2))
  }

  @Test("Typing after nested bare > injects a space")
  func typingAfterNestedBareGtInjectsSpace() {
    let state = EditorState(markdown: "> >", selection: .cursor(3))
    let result = EditorUpdate.update(state, event: .insertText("H"))
    #expect(result.markdown == "> > H")
    #expect(result.selection == .cursor(5))
  }

  @Test("Pasting text after bare > injects a space")
  func pastingAfterBareGtInjectsSpace() {
    let state = EditorState(markdown: ">", selection: .cursor(1))
    let result = EditorUpdate.update(state, event: .paste("Hello"))
    #expect(result.markdown == "> Hello")
    #expect(result.selection == .cursor(7))
  }

  @Test("Typing in middle of blockquote content does not inject space")
  func typingInContentNoSpaceInjection() {
    let state = EditorState(markdown: "> Hello", selection: .cursor(4))
    let result = EditorUpdate.update(state, event: .insertText("x"))
    #expect(result.markdown == "> Hexllo")
    #expect(result.selection == .cursor(5))
  }

  @Test("Typing > at start of line does not inject space (not a prefix yet)")
  func typingGtAtStartNoInjection() {
    let state = EditorState(markdown: "", selection: .cursor(0))
    let result = EditorUpdate.update(state, event: .insertText(">"))
    #expect(result.markdown == ">")
    #expect(result.selection == .cursor(1))
  }

  @Test("Typing after bare > nested inside a list item injects a space")
  func typingAfterBareGtInsideListInjectsSpace() {
    let state = EditorState(markdown: "- Item\n  >", selection: .cursor(10))
    let result = EditorUpdate.update(state, event: .insertText("Q"))
    #expect(result.markdown == "- Item\n  > Q")
    #expect(result.selection == .cursor(12))
  }

  @Test("Typing after bare > with tab indent inside list injects a space")
  func typingAfterBareGtWithTabIndentInjectsSpace() {
    let state = EditorState(markdown: "- Item\n\t>", selection: .cursor(9))
    let result = EditorUpdate.update(state, event: .insertText("Q"))
    #expect(result.markdown == "- Item\n\t> Q")
    #expect(result.selection == .cursor(11))
  }

  // MARK: - Bare `>` whole-unit deletion

  @Test("Backspace after bare > deletes the entire >")
  func backspaceAfterBareGtDeletesUnit() {
    let state = EditorState(markdown: ">", selection: .cursor(1))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    #expect(result.markdown == "")
    #expect(result.selection == .cursor(0))
  }

  @Test("Backspace after bare nested > deletes one level")
  func backspaceAfterBareNestedGtDeletesOneLevel() {
    let state = EditorState(markdown: "> >", selection: .cursor(3))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    #expect(result.markdown == "> ")
    #expect(result.selection == .cursor(2))
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
