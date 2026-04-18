import Foundation
import Testing

@testable import MarkdownEditor

/// Pure state transition tests for delete variant events:
/// deleteToBeginningOfLine, deleteToEndOfLine, deleteWordBackward, deleteWordForward.
@Suite("Delete Variant Updates")
struct DeleteVariantUpdateTests {

  // MARK: - deleteToBeginningOfLine

  @Test("deleteToBeginningOfLine from middle of line")
  func deleteToBeginningOfLineMiddle() {
    let state = EditorState(markdown: "hello world", selection: .cursor(5))
    let result = EditorUpdate.update(state, event: .deleteToBeginningOfLine)
    #expect(result.markdown == " world")
    #expect(result.selection == .cursor(0))
  }

  @Test("deleteToBeginningOfLine at start of line is no-op")
  func deleteToBeginningOfLineAtStart() {
    let state = EditorState(markdown: "hello\nworld", selection: .cursor(6))
    let result = EditorUpdate.update(state, event: .deleteToBeginningOfLine)
    #expect(result.markdown == "hello\nworld")
    #expect(result.selection == .cursor(6))
  }

  @Test("deleteToBeginningOfLine at start of document is no-op")
  func deleteToBeginningOfLineAtDocStart() {
    let state = EditorState(markdown: "hello", selection: .cursor(0))
    let result = EditorUpdate.update(state, event: .deleteToBeginningOfLine)
    #expect(result.markdown == "hello")
    #expect(result.selection == .cursor(0))
  }

  @Test("deleteToBeginningOfLine on second line")
  func deleteToBeginningOfLineSecondLine() {
    let state = EditorState(markdown: "hello\nworld", selection: .cursor(9))
    let result = EditorUpdate.update(state, event: .deleteToBeginningOfLine)
    #expect(result.markdown == "hello\nld")
    #expect(result.selection == .cursor(6))
  }

  @Test("deleteToBeginningOfLine with selection deletes selection")
  func deleteToBeginningOfLineWithSelection() {
    let state = EditorState(markdown: "hello world", selection: .range(anchor: 2, head: 7))
    let result = EditorUpdate.update(state, event: .deleteToBeginningOfLine)
    #expect(result.markdown == "heorld")
    #expect(result.selection == .cursor(2))
  }

  // MARK: - deleteToEndOfLine

  @Test("deleteToEndOfLine from middle of line")
  func deleteToEndOfLineMiddle() {
    let state = EditorState(markdown: "hello world", selection: .cursor(5))
    let result = EditorUpdate.update(state, event: .deleteToEndOfLine)
    #expect(result.markdown == "hello")
    #expect(result.selection == .cursor(5))
  }

  @Test("deleteToEndOfLine at end of line is no-op")
  func deleteToEndOfLineAtEnd() {
    let state = EditorState(markdown: "hello\nworld", selection: .cursor(5))
    let result = EditorUpdate.update(state, event: .deleteToEndOfLine)
    #expect(result.markdown == "hello\nworld")
    #expect(result.selection == .cursor(5))
  }

  @Test("deleteToEndOfLine at end of document is no-op")
  func deleteToEndOfLineAtDocEnd() {
    let state = EditorState(markdown: "hello", selection: .cursor(5))
    let result = EditorUpdate.update(state, event: .deleteToEndOfLine)
    #expect(result.markdown == "hello")
    #expect(result.selection == .cursor(5))
  }

  @Test("deleteToEndOfLine preserves newline")
  func deleteToEndOfLinePreservesNewline() {
    let state = EditorState(markdown: "hello world\nsecond", selection: .cursor(5))
    let result = EditorUpdate.update(state, event: .deleteToEndOfLine)
    #expect(result.markdown == "hello\nsecond")
    #expect(result.selection == .cursor(5))
  }

  @Test("deleteToEndOfLine on first line from start")
  func deleteToEndOfLineFromStart() {
    let state = EditorState(markdown: "hello\nworld", selection: .cursor(0))
    let result = EditorUpdate.update(state, event: .deleteToEndOfLine)
    #expect(result.markdown == "\nworld")
    #expect(result.selection == .cursor(0))
  }

  @Test("deleteToEndOfLine with selection deletes selection")
  func deleteToEndOfLineWithSelection() {
    let state = EditorState(markdown: "hello world", selection: .range(anchor: 2, head: 7))
    let result = EditorUpdate.update(state, event: .deleteToEndOfLine)
    #expect(result.markdown == "heorld")
    #expect(result.selection == .cursor(2))
  }

  // MARK: - deleteWordBackward

  @Test("deleteWordBackward deletes previous word")
  func deleteWordBackwardWord() {
    let state = EditorState(markdown: "hello world", selection: .cursor(11))
    let result = EditorUpdate.update(state, event: .deleteWordBackward)
    #expect(result.markdown == "hello ")
    #expect(result.selection == .cursor(6))
  }

  @Test("deleteWordBackward skips whitespace then deletes word")
  func deleteWordBackwardSkipsWhitespace() {
    let state = EditorState(markdown: "hello   world", selection: .cursor(8))
    let result = EditorUpdate.update(state, event: .deleteWordBackward)
    // Cursor at position 8 is at 'w' in "world". Skip whitespace back (3 spaces),
    // then skip non-whitespace back ("hello"), landing at 0.
    #expect(result.markdown == "world")
    #expect(result.selection == .cursor(0))
  }

  @Test("deleteWordBackward at start of document is no-op")
  func deleteWordBackwardAtStart() {
    let state = EditorState(markdown: "hello", selection: .cursor(0))
    let result = EditorUpdate.update(state, event: .deleteWordBackward)
    #expect(result.markdown == "hello")
    #expect(result.selection == .cursor(0))
  }

  @Test("deleteWordBackward in middle of word")
  func deleteWordBackwardMiddleOfWord() {
    let state = EditorState(markdown: "hello world", selection: .cursor(8))
    let result = EditorUpdate.update(state, event: .deleteWordBackward)
    // Cursor at 8 is at 'r' in "world". No whitespace immediately before,
    // skip non-whitespace back: "wo" gets deleted, then whitespace " " gets deleted,
    // Wait, let me re-check. At pos 8 in "hello world":
    // h(0)e(1)l(2)l(3)o(4) (5)w(6)o(7)r(8)
    // char at 7 is 'o' (non-whitespace), so skip non-ws back: o,w -> pos 6
    // char at 5 is ' ' (whitespace), so skip ws back: -> pos 5
    // Actually the algorithm is: skip whitespace, THEN skip non-whitespace.
    // At pos 8, char at 7 = 'o', not whitespace, so no whitespace skipping.
    // Then skip non-ws: 'o' at 7, 'w' at 6, ' ' at 5 stops -> target = 6
    #expect(result.markdown == "hello rld")
    #expect(result.selection == .cursor(6))
  }

  @Test("deleteWordBackward with selection deletes selection")
  func deleteWordBackwardWithSelection() {
    let state = EditorState(markdown: "hello world", selection: .range(anchor: 2, head: 7))
    let result = EditorUpdate.update(state, event: .deleteWordBackward)
    #expect(result.markdown == "heorld")
    #expect(result.selection == .cursor(2))
  }

  // MARK: - deleteWordForward

  @Test("deleteWordForward deletes next word")
  func deleteWordForwardWord() {
    let state = EditorState(markdown: "hello world", selection: .cursor(0))
    let result = EditorUpdate.update(state, event: .deleteWordForward)
    // Skip non-ws forward: "hello" -> pos 5
    // Skip ws forward: " " -> pos 6
    #expect(result.markdown == "world")
    #expect(result.selection == .cursor(0))
  }

  @Test("deleteWordForward from middle of word")
  func deleteWordForwardMiddle() {
    let state = EditorState(markdown: "hello world", selection: .cursor(3))
    let result = EditorUpdate.update(state, event: .deleteWordForward)
    // At pos 3 ('l'), skip non-ws: 'l','o' -> pos 5
    // Skip ws: ' ' -> pos 6
    #expect(result.markdown == "helworld")
    #expect(result.selection == .cursor(3))
  }

  @Test("deleteWordForward at end of document is no-op")
  func deleteWordForwardAtEnd() {
    let state = EditorState(markdown: "hello", selection: .cursor(5))
    let result = EditorUpdate.update(state, event: .deleteWordForward)
    #expect(result.markdown == "hello")
    #expect(result.selection == .cursor(5))
  }

  @Test("deleteWordForward with selection deletes selection")
  func deleteWordForwardWithSelection() {
    let state = EditorState(markdown: "hello world", selection: .range(anchor: 2, head: 7))
    let result = EditorUpdate.update(state, event: .deleteWordForward)
    #expect(result.markdown == "heorld")
    #expect(result.selection == .cursor(2))
  }

  @Test("deleteWordForward skips word then trailing whitespace")
  func deleteWordForwardSkipsTrailingWhitespace() {
    let state = EditorState(markdown: "hello   world", selection: .cursor(0))
    let result = EditorUpdate.update(state, event: .deleteWordForward)
    // Skip non-ws: "hello" -> pos 5
    // Skip ws: "   " -> pos 8
    #expect(result.markdown == "world")
    #expect(result.selection == .cursor(0))
  }
}
