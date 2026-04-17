import Foundation

/// A user action that produces a state transition.
///
/// Events are semantic — they describe what the user did, not low-level
/// key codes or mouse coordinates. The `EditorUpdate.update()` function
/// maps `(EditorState, EditorEvent) -> EditorState`.
public enum EditorEvent: Sendable, Equatable {
  /// Insert text at the current cursor/selection.
  /// Replaces selected text if there is a selection.
  case insertText(String)

  /// Insert a newline at the current cursor/selection.
  /// Separate from insertText("\n") because markdown-aware behavior
  /// (list continuation, etc.) hooks into this event.
  case insertNewline

  /// Insert a line break (Shift+Return). In list context, this continues
  /// the current list item on the next line with appropriate indentation.
  /// Outside a list, behaves like a plain newline.
  case insertLineBreak

  /// Delete the character before the cursor, or delete the selection.
  case deleteBackward

  /// Delete the character after the cursor, or delete the selection.
  case deleteForward

  /// Set the selection (e.g., from a mouse click or arrow key).
  case setSelection(Selection)

  /// Paste text, replacing the current selection if any.
  case paste(String)

  /// Indent the current line (Tab key). In list context, increases nesting.
  case indent

  /// Outdent the current line (Shift+Tab). In list context, decreases nesting.
  case outdent
}
