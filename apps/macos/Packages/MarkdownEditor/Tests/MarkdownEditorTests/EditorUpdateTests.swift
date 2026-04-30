import Foundation
import Testing

@testable import MarkdownEditor

/// Pure state transition tests — no rendering, no NSTextView.
/// Tests that `EditorUpdate.update(state, event) -> state` produces correct results.
@Suite("EditorUpdate")
struct EditorUpdateTests {

  // MARK: - Insert Text

  @Test("Insert single character into empty document")
  func insertCharIntoEmpty() {
    let state = EditorState()
    let result = EditorUpdate.update(state, event: .insertText("a"))
    #expect(result.markdown == "a")
    #expect(result.selection == .cursor(1))
  }

  @Test("Insert character at cursor position")
  func insertCharAtCursor() {
    let state = EditorState(markdown: "hllo", selection: .cursor(1))
    let result = EditorUpdate.update(state, event: .insertText("e"))
    #expect(result.markdown == "hello")
    #expect(result.selection == .cursor(2))
  }

  @Test("Insert text at end of document")
  func insertAtEnd() {
    let state = EditorState(markdown: "hello", selection: .cursor(5))
    let result = EditorUpdate.update(state, event: .insertText(" world"))
    #expect(result.markdown == "hello world")
    #expect(result.selection == .cursor(11))
  }

  @Test("Insert text replaces selection")
  func insertReplacesSelection() {
    let state = EditorState(markdown: "hello world", selection: .range(anchor: 5, head: 11))
    let result = EditorUpdate.update(state, event: .insertText("!"))
    #expect(result.markdown == "hello!")
    #expect(result.selection == .cursor(6))
  }

  @Test("Insert multi-character string")
  func insertMultiChar() {
    let state = EditorState(markdown: "ab", selection: .cursor(1))
    let result = EditorUpdate.update(state, event: .insertText("xyz"))
    #expect(result.markdown == "axyzb")
    #expect(result.selection == .cursor(4))
  }

  // MARK: - Insert Newline

  @Test("Insert newline into empty document")
  func insertNewlineEmpty() {
    // Per the snap-to-even-trailing-`\n` Enter policy: empty doc has 0
    // trailing `\n` (even), so Enter inserts `\n\n` (paragraph break) — the
    // user lands at the start of a fresh paragraph. Soft-break-only behavior
    // is reachable via Shift+Enter.
    let state = EditorState()
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "\n\n")
    #expect(result.selection == .cursor(2))
  }

  @Test("Insert newline at end of text")
  func insertNewlineAtEnd() {
    let state = EditorState(markdown: "hello", selection: .cursor(5))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "hello\n\n")
    #expect(result.selection == .cursor(7))
  }

  @Test("Insert newline in middle of text")
  func insertNewlineMiddle() {
    let state = EditorState(markdown: "hello world", selection: .cursor(5))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(result.markdown == "hello\n\n world")
    #expect(result.selection == .cursor(7))
  }

  // MARK: - Delete Backward

  @Test("Delete backward at start of document does nothing")
  func deleteBackwardAtStart() {
    let state = EditorState(markdown: "hello", selection: .cursor(0))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    #expect(result.markdown == "hello")
    #expect(result.selection == .cursor(0))
  }

  @Test("Delete backward removes character before cursor")
  func deleteBackwardRemovesChar() {
    let state = EditorState(markdown: "hello", selection: .cursor(5))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    #expect(result.markdown == "hell")
    #expect(result.selection == .cursor(4))
  }

  @Test("Delete backward in middle of text")
  func deleteBackwardMiddle() {
    let state = EditorState(markdown: "hello", selection: .cursor(3))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    #expect(result.markdown == "helo")
    #expect(result.selection == .cursor(2))
  }

  @Test("Delete backward with selection deletes selection")
  func deleteBackwardSelection() {
    let state = EditorState(markdown: "hello world", selection: .range(anchor: 0, head: 5))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    #expect(result.markdown == " world")
    #expect(result.selection == .cursor(0))
  }

  @Test("Delete backward removes emoji as one unit")
  func deleteBackwardEmoji() {
    let state = EditorState(markdown: "a\u{1F600}b", selection: .cursor(3))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    #expect(result.markdown == "ab")
    #expect(result.selection == .cursor(1))
  }

  // MARK: - Delete Forward

  @Test("Delete forward at end of document does nothing")
  func deleteForwardAtEnd() {
    let state = EditorState(markdown: "hello", selection: .cursor(5))
    let result = EditorUpdate.update(state, event: .deleteForward)
    #expect(result.markdown == "hello")
    #expect(result.selection == .cursor(5))
  }

  @Test("Delete forward removes character after cursor")
  func deleteForwardRemovesChar() {
    let state = EditorState(markdown: "hello", selection: .cursor(0))
    let result = EditorUpdate.update(state, event: .deleteForward)
    #expect(result.markdown == "ello")
    #expect(result.selection == .cursor(0))
  }

  @Test("Delete forward with selection deletes selection")
  func deleteForwardSelection() {
    let state = EditorState(markdown: "hello world", selection: .range(anchor: 5, head: 11))
    let result = EditorUpdate.update(state, event: .deleteForward)
    #expect(result.markdown == "hello")
    #expect(result.selection == .cursor(5))
  }

  // MARK: - Set Selection

  @Test("Set cursor position")
  func setCursor() {
    let state = EditorState(markdown: "hello", selection: .cursor(0))
    let result = EditorUpdate.update(state, event: .setSelection(.cursor(3)))
    #expect(result.markdown == "hello")
    #expect(result.selection == .cursor(3))
  }

  @Test("Set range selection")
  func setRange() {
    let state = EditorState(markdown: "hello", selection: .cursor(0))
    let result = EditorUpdate.update(state, event: .setSelection(.range(anchor: 1, head: 4)))
    #expect(result.markdown == "hello")
    #expect(result.selection == .range(anchor: 1, head: 4))
  }

  @Test("Set selection clamps to valid range")
  func setSelectionClamped() {
    let state = EditorState(markdown: "hi", selection: .cursor(0))
    let result = EditorUpdate.update(state, event: .setSelection(.cursor(100)))
    #expect(result.selection == .cursor(2))
  }

  // MARK: - Paste

  @Test("Paste at cursor")
  func pasteAtCursor() {
    let state = EditorState(markdown: "ab", selection: .cursor(1))
    let result = EditorUpdate.update(state, event: .paste("XY"))
    #expect(result.markdown == "aXYb")
    #expect(result.selection == .cursor(3))
  }

  @Test("Paste replaces selection")
  func pasteReplacesSelection() {
    let state = EditorState(markdown: "hello world", selection: .range(anchor: 0, head: 5))
    let result = EditorUpdate.update(state, event: .paste("goodbye"))
    #expect(result.markdown == "goodbye world")
    #expect(result.selection == .cursor(7))
  }

  // MARK: - Multi-step Sequences

  @Test("Type 'Hello' character by character")
  func typeHelloSequence() {
    var state = EditorState()
    for char in "Hello" {
      state = EditorUpdate.update(state, event: .insertText(String(char)))
    }
    #expect(state.markdown == "Hello")
    #expect(state.selection == .cursor(5))
  }

  @Test("Type, select all, delete, retype")
  func typeSelectDeleteRetype() {
    var state = EditorState()

    // Type "abc"
    for char in "abc" {
      state = EditorUpdate.update(state, event: .insertText(String(char)))
    }
    #expect(state.markdown == "abc")

    // Select all
    state = EditorUpdate.update(state, event: .setSelection(.range(anchor: 0, head: 3)))

    // Delete selection
    state = EditorUpdate.update(state, event: .deleteBackward)
    #expect(state.markdown == "")
    #expect(state.selection == .cursor(0))

    // Type "xyz"
    for char in "xyz" {
      state = EditorUpdate.update(state, event: .insertText(String(char)))
    }
    #expect(state.markdown == "xyz")
    #expect(state.selection == .cursor(3))
  }

  @Test("Empty document operations are safe")
  func emptyDocumentSafety() {
    let state = EditorState()
    #expect(EditorUpdate.update(state, event: .deleteBackward) == state)
    #expect(EditorUpdate.update(state, event: .deleteForward) == state)
    #expect(EditorUpdate.update(state, event: .setSelection(.cursor(0))) == state)
  }
}
