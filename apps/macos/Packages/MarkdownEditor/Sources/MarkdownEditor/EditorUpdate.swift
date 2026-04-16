import Foundation

/// Pure state transition function for the editor.
///
/// Given the current state and an event, produces the next state.
/// This function has no side effects — it is the core of the Elm architecture.
enum EditorUpdate {

  /// Compute the next editor state from the current state and an event.
  static func update(_ state: EditorState, event: EditorEvent) -> EditorState {
    switch event {
    case .insertText(let text):
      return handleInsertText(state, text: text)

    case .insertNewline:
      return handleInsertText(state, text: "\n")

    case .deleteBackward:
      return handleDeleteBackward(state)

    case .deleteForward:
      return handleDeleteForward(state)

    case .setSelection(let selection):
      return handleSetSelection(state, selection: selection)

    case .paste(let text):
      return handleInsertText(state, text: text)
    }
  }

  // MARK: - Event Handlers

  private static func handleInsertText(_ state: EditorState, text: String) -> EditorState {
    let nsMarkdown = state.markdown as NSString

    switch state.selection {
    case .cursor(let pos):
      let clampedPos = min(pos, nsMarkdown.length)
      let newMarkdown = nsMarkdown.replacingCharacters(
        in: NSRange(location: clampedPos, length: 0), with: text)
      let newPos = clampedPos + (text as NSString).length
      return EditorState(markdown: newMarkdown, selection: .cursor(newPos))

    case .range(let anchor, let head):
      let start = min(anchor, head)
      let end = max(anchor, head)
      let clampedStart = min(start, nsMarkdown.length)
      let clampedEnd = min(end, nsMarkdown.length)
      let replaceRange = NSRange(location: clampedStart, length: clampedEnd - clampedStart)
      let newMarkdown = nsMarkdown.replacingCharacters(in: replaceRange, with: text)
      let newPos = clampedStart + (text as NSString).length
      return EditorState(markdown: newMarkdown, selection: .cursor(newPos))
    }
  }

  private static func handleDeleteBackward(_ state: EditorState) -> EditorState {
    let nsMarkdown = state.markdown as NSString

    switch state.selection {
    case .cursor(let pos):
      guard pos > 0 else { return state }
      let clampedPos = min(pos, nsMarkdown.length)
      // Delete one character before cursor (handle multi-byte via composed character range)
      let deleteRange = nsMarkdown.rangeOfComposedCharacterSequence(at: clampedPos - 1)
      let newMarkdown = nsMarkdown.replacingCharacters(in: deleteRange, with: "")
      return EditorState(markdown: newMarkdown, selection: .cursor(deleteRange.location))

    case .range(let anchor, let head):
      // Delete the selection
      let start = min(anchor, head)
      let end = max(anchor, head)
      let clampedStart = min(start, nsMarkdown.length)
      let clampedEnd = min(end, nsMarkdown.length)
      let deleteRange = NSRange(location: clampedStart, length: clampedEnd - clampedStart)
      let newMarkdown = nsMarkdown.replacingCharacters(in: deleteRange, with: "")
      return EditorState(markdown: newMarkdown, selection: .cursor(clampedStart))
    }
  }

  private static func handleDeleteForward(_ state: EditorState) -> EditorState {
    let nsMarkdown = state.markdown as NSString

    switch state.selection {
    case .cursor(let pos):
      let clampedPos = min(pos, nsMarkdown.length)
      guard clampedPos < nsMarkdown.length else { return state }
      let deleteRange = nsMarkdown.rangeOfComposedCharacterSequence(at: clampedPos)
      let newMarkdown = nsMarkdown.replacingCharacters(in: deleteRange, with: "")
      return EditorState(markdown: newMarkdown, selection: .cursor(clampedPos))

    case .range(let anchor, let head):
      // Same as deleteBackward with a range selection
      let start = min(anchor, head)
      let end = max(anchor, head)
      let clampedStart = min(start, nsMarkdown.length)
      let clampedEnd = min(end, nsMarkdown.length)
      let deleteRange = NSRange(location: clampedStart, length: clampedEnd - clampedStart)
      let newMarkdown = nsMarkdown.replacingCharacters(in: deleteRange, with: "")
      return EditorState(markdown: newMarkdown, selection: .cursor(clampedStart))
    }
  }

  private static func handleSetSelection(_ state: EditorState, selection: Selection) -> EditorState {
    let maxPos = (state.markdown as NSString).length

    // Clamp selection to valid range
    let clamped: Selection
    switch selection {
    case .cursor(let pos):
      clamped = .cursor(min(max(0, pos), maxPos))
    case .range(let anchor, let head):
      clamped = .range(
        anchor: min(max(0, anchor), maxPos),
        head: min(max(0, head), maxPos))
    }

    return EditorState(markdown: state.markdown, selection: clamped)
  }
}
