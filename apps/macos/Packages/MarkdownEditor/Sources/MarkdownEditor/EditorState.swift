import Foundation

/// The complete state of the markdown editor.
///
/// This is the "model" in the Elm architecture. All visual output is derived
/// from this state. All user interactions produce a new state via `EditorUpdate.update()`.
public struct EditorState: Sendable, Equatable {
  /// The raw markdown text.
  public var markdown: String

  /// The current selection (cursor position or range).
  public var selection: Selection

  public init(markdown: String = "", selection: Selection = .cursor(0)) {
    self.markdown = markdown
    self.selection = selection
  }
}

/// A selection within the editor text.
public enum Selection: Sendable, Equatable {
  /// An insertion point (no selected text).
  case cursor(Int)

  /// A range selection with anchor and head positions.
  /// Anchor is where the selection started, head is where it ends.
  /// Head may be before or after anchor (for left-to-right vs right-to-left selection).
  case range(anchor: Int, head: Int)

  /// The insertion point, or the head of the selection.
  public var head: Int {
    switch self {
    case .cursor(let pos): return pos
    case .range(_, let head): return head
    }
  }

  /// The NSRange representation (location + length, always non-negative length).
  public var nsRange: NSRange {
    switch self {
    case .cursor(let pos):
      return NSRange(location: pos, length: 0)
    case .range(let anchor, let head):
      let start = min(anchor, head)
      let end = max(anchor, head)
      return NSRange(location: start, length: end - start)
    }
  }

  /// Whether this is a cursor (no selection) or a range.
  public var isCursor: Bool {
    if case .cursor = self { return true }
    return false
  }

  /// The start of the selected region (or cursor position).
  public var lowerBound: Int {
    switch self {
    case .cursor(let pos): return pos
    case .range(let anchor, let head): return min(anchor, head)
    }
  }

  /// The end of the selected region (or cursor position).
  public var upperBound: Int {
    switch self {
    case .cursor(let pos): return pos
    case .range(let anchor, let head): return max(anchor, head)
    }
  }
}
