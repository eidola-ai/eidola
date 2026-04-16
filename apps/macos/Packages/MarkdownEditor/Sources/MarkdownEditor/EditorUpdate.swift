import Foundation

/// Pure state transition function for the editor.
///
/// Given the current state and an event, produces the next state.
/// This function has no side effects — it is the core of the Elm architecture.
enum EditorUpdate {

  // Regex matching an unordered list marker at the start of a line: leading
  // whitespace + one of `-`, `*`, `+` + a space.
  private static let listMarkerPattern = try! NSRegularExpression(
    pattern: #"^([ \t]*)([-*+]) $"#, options: [])

  // Regex matching a non-empty list item line (marker + at least one content char).
  private static let listItemPattern = try! NSRegularExpression(
    pattern: #"^([ \t]*[-*+] ).+"#, options: [])

  // Regex matching an empty ordered list marker: leading whitespace + digits + ". "
  private static let orderedListMarkerPattern = try! NSRegularExpression(
    pattern: #"^([ \t]*)(\d+)\. $"#, options: [])

  // Regex matching a non-empty ordered list item line (digits + ". " + content).
  private static let orderedListItemPattern = try! NSRegularExpression(
    pattern: #"^([ \t]*)(\d+)\. .+"#, options: [])

  /// Regex matching the start of an ordered list line: optional indent + digits + ". "
  private static let orderedLinePattern = try! NSRegularExpression(
    pattern: #"^([ \t]*)(\d+)\. "#, options: [])

  /// Compute the next editor state from the current state and an event.
  static func update(_ state: EditorState, event: EditorEvent) -> EditorState {
    let newState: EditorState
    switch event {
    case .insertText(let text):
      newState = handleInsertText(state, text: text)

    case .insertNewline:
      newState = handleInsertNewline(state)

    case .deleteBackward:
      newState = handleDeleteBackward(state)

    case .deleteForward:
      newState = handleDeleteForward(state)

    case .setSelection(let selection):
      return handleSetSelection(state, selection: selection)

    case .paste(let text):
      newState = handleInsertText(state, text: text)
    }

    // Post-process: renumber ordered lists after any text mutation.
    return renumberOrderedLists(in: newState)
  }

  // MARK: - Helpers

  /// Returns the line content (without trailing newline) containing the given position.
  private static func currentLine(_ nsMarkdown: NSString, at pos: Int) -> (range: NSRange, text: String) {
    let lineRange = nsMarkdown.lineRange(for: NSRange(location: pos, length: 0))
    let lineText = nsMarkdown.substring(with: lineRange)
    return (lineRange, lineText)
  }

  // MARK: - Event Handlers

  private static func handleInsertNewline(_ state: EditorState) -> EditorState {
    let nsMarkdown = state.markdown as NSString
    let pos: Int
    switch state.selection {
    case .cursor(let p): pos = min(p, nsMarkdown.length)
    case .range(let anchor, let head): pos = min(min(anchor, head), nsMarkdown.length)
    }

    let (lineRange, lineText) = currentLine(nsMarkdown, at: pos)
    let lineNS = lineText as NSString

    // Check if the current line is an empty list item (just marker, no content).
    // Pattern: optional whitespace + marker char + space + end-of-line
    let emptyMatch = listMarkerPattern.firstMatch(
      in: lineText, range: NSRange(location: 0, length: lineNS.length))
    if emptyMatch != nil {
      // Remove the marker line entirely, leave a blank line.
      let newMarkdown = nsMarkdown.replacingCharacters(in: lineRange, with: "\n")
      return EditorState(markdown: newMarkdown, selection: .cursor(lineRange.location))
    }

    // Check if the current line is an empty ordered list item.
    let emptyOrderedMatch = orderedListMarkerPattern.firstMatch(
      in: lineText, range: NSRange(location: 0, length: lineNS.length))
    if emptyOrderedMatch != nil {
      // Remove the marker line entirely, leave a blank line.
      let newMarkdown = nsMarkdown.replacingCharacters(in: lineRange, with: "\n")
      return EditorState(markdown: newMarkdown, selection: .cursor(lineRange.location))
    }

    // Check if the current line is a non-empty ordered list item.
    let orderedItemMatch = orderedListItemPattern.firstMatch(
      in: lineText, range: NSRange(location: 0, length: lineNS.length))
    if let orderedItemMatch = orderedItemMatch {
      // Extract the current number and increment it.
      let indentRange = orderedItemMatch.range(at: 1)
      let indent = lineNS.substring(with: indentRange)
      let numberRange = orderedItemMatch.range(at: 2)
      let numberStr = lineNS.substring(with: numberRange)
      let currentNumber = Int(numberStr) ?? 1
      let nextNumber = currentNumber + 1
      let prefix = "\(indent)\(nextNumber). "

      // First, handle any selected text by deleting it.
      var workMarkdown = state.markdown
      var workPos = pos
      if case .range(let anchor, let head) = state.selection {
        let start = min(anchor, head)
        let end = max(anchor, head)
        let clampedStart = min(start, nsMarkdown.length)
        let clampedEnd = min(end, nsMarkdown.length)
        let deleteRange = NSRange(location: clampedStart, length: clampedEnd - clampedStart)
        workMarkdown = nsMarkdown.replacingCharacters(in: deleteRange, with: "")
        workPos = clampedStart
      }

      let workNS = workMarkdown as NSString
      let insertion = "\n\(prefix)"
      let newMarkdown = workNS.replacingCharacters(
        in: NSRange(location: workPos, length: 0), with: insertion)
      let newPos = workPos + (insertion as NSString).length
      return EditorState(markdown: newMarkdown, selection: .cursor(newPos))
    }

    // Check if the current line is a non-empty unordered list item.
    let itemMatch = listItemPattern.firstMatch(
      in: lineText, range: NSRange(location: 0, length: lineNS.length))
    if let itemMatch = itemMatch {
      // Continue the list: insert newline + same marker prefix.
      let prefixRange = itemMatch.range(at: 1)
      let prefix = lineNS.substring(with: prefixRange)

      // First, handle any selected text by deleting it.
      var workMarkdown = state.markdown
      var workPos = pos
      if case .range(let anchor, let head) = state.selection {
        let start = min(anchor, head)
        let end = max(anchor, head)
        let clampedStart = min(start, nsMarkdown.length)
        let clampedEnd = min(end, nsMarkdown.length)
        let deleteRange = NSRange(location: clampedStart, length: clampedEnd - clampedStart)
        workMarkdown = nsMarkdown.replacingCharacters(in: deleteRange, with: "")
        workPos = clampedStart
      }

      let workNS = workMarkdown as NSString
      let insertion = "\n\(prefix)"
      let newMarkdown = workNS.replacingCharacters(
        in: NSRange(location: workPos, length: 0), with: insertion)
      let newPos = workPos + (insertion as NSString).length
      return EditorState(markdown: newMarkdown, selection: .cursor(newPos))
    }

    // Default: plain newline insertion.
    return handleInsertText(state, text: "\n")
  }

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

      // Check if cursor is right after a list marker (e.g., "- |content" or "1. |content").
      // If so, remove the entire marker instead of just one character.
      let (lineRange, lineText) = currentLine(nsMarkdown, at: clampedPos)
      let lineNS = lineText as NSString
      let posInLine = clampedPos - lineRange.location

      // Match unordered: "  - " or "- " at start of line
      let markerRegex = try! NSRegularExpression(pattern: #"^([ \t]*)([-*+]) "#, options: [])
      if let match = markerRegex.firstMatch(
        in: lineText, range: NSRange(location: 0, length: lineNS.length))
      {
        let markerEnd = match.range.location + match.range.length
        if posInLine == markerEnd {
          // Cursor is right after marker — remove the entire marker.
          let markerAbsRange = NSRange(
            location: lineRange.location + match.range.location,
            length: match.range.length)
          let newMarkdown = nsMarkdown.replacingCharacters(in: markerAbsRange, with: "")
          return EditorState(
            markdown: newMarkdown, selection: .cursor(markerAbsRange.location))
        }
      }

      // Match ordered: "  1. " or "1. " at start of line
      let orderedMarkerRegex = try! NSRegularExpression(
        pattern: #"^([ \t]*)(\d+)\. "#, options: [])
      if let match = orderedMarkerRegex.firstMatch(
        in: lineText, range: NSRange(location: 0, length: lineNS.length))
      {
        let markerEnd = match.range.location + match.range.length
        if posInLine == markerEnd {
          // Cursor is right after ordered marker — remove the entire marker.
          let markerAbsRange = NSRange(
            location: lineRange.location + match.range.location,
            length: match.range.length)
          let newMarkdown = nsMarkdown.replacingCharacters(in: markerAbsRange, with: "")
          return EditorState(
            markdown: newMarkdown, selection: .cursor(markerAbsRange.location))
        }
      }

      // Default: delete one character before cursor
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

  // MARK: - Ordered List Renumbering

  /// Scan all lines and renumber ordered list items so each contiguous run
  /// of same-indent ordered items is numbered sequentially starting from 1.
  /// Adjusts cursor position if renumbering changes character counts before it.
  private static func renumberOrderedLists(in state: EditorState) -> EditorState {
    let lines = state.markdown.components(separatedBy: "\n")
    var newLines: [String] = []

    // Track running counters per indent level.
    // Key: indent string, Value: next expected number
    var counters: [String: Int] = [:]
    // Track which indent levels had an ordered item on the previous line,
    // so we can reset counters when a gap appears.
    var activeIndents: Set<String> = []

    // We need to track cumulative character offset changes to adjust cursor.
    let cursorPos: Int
    switch state.selection {
    case .cursor(let p): cursorPos = p
    case .range(_, let head): cursorPos = head
    }
    let cursorAnchor: Int?
    switch state.selection {
    case .cursor: cursorAnchor = nil
    case .range(let anchor, _): cursorAnchor = anchor
    }

    var cursorDelta = 0
    var anchorDelta = 0
    var charOffset = 0  // running character offset into the original string

    for line in lines {
      let lineNS = line as NSString
      let match = orderedLinePattern.firstMatch(
        in: line, range: NSRange(location: 0, length: lineNS.length))

      if let match = match {
        let indent = lineNS.substring(with: match.range(at: 1))
        let oldNumberStr = lineNS.substring(with: match.range(at: 2))

        // Check if this indent was active on the previous line. If not, reset.
        if !activeIndents.contains(indent) {
          counters[indent] = 1
        }

        let correctNumber = counters[indent] ?? 1
        let correctNumberStr = String(correctNumber)
        counters[indent] = correctNumber + 1

        // Mark this indent as active. Clear indents that are "deeper" or different
        // if the current line is at a shallower indent.
        activeIndents.insert(indent)

        if correctNumberStr != oldNumberStr {
          // Replace the number
          let oldPrefix = "\(indent)\(oldNumberStr). "
          let newPrefix = "\(indent)\(correctNumberStr). "
          let rest = lineNS.substring(from: match.range.length)
          let newLine = "\(newPrefix)\(rest)"
          newLines.append(newLine)

          let lengthDiff = (newPrefix as NSString).length - (oldPrefix as NSString).length

          // Adjust cursor if it's after the number in this line (or on a later line).
          // charOffset is the start of this line in the original string.
          let oldPrefixEnd = charOffset + (oldPrefix as NSString).length
          if cursorPos >= oldPrefixEnd {
            // Cursor is after the marker on this line or on a later line
            cursorDelta += lengthDiff
          } else if cursorPos > charOffset {
            // Cursor is inside the marker on this line — move it to end of new marker
            cursorDelta += lengthDiff
          }

          if let anchor = cursorAnchor {
            if anchor >= oldPrefixEnd {
              anchorDelta += lengthDiff
            } else if anchor > charOffset {
              anchorDelta += lengthDiff
            }
          }
        } else {
          newLines.append(line)
        }
      } else {
        // Not an ordered list line — reset all counters for indents that are
        // no longer active. A blank line or non-list-item breaks all lists.
        activeIndents.removeAll()
        // Don't clear counters — they'll be reset when activeIndents doesn't contain
        // the indent on the next ordered item.
        newLines.append(line)
      }

      // Advance charOffset: line length + 1 for the \n separator
      charOffset += lineNS.length + 1
    }

    let newMarkdown = newLines.joined(separator: "\n")
    if newMarkdown == state.markdown {
      return state
    }

    let newSelection: Selection
    if let anchor = cursorAnchor {
      let newAnchor = max(0, anchor + anchorDelta)
      let newHead = max(0, cursorPos + cursorDelta)
      newSelection = .range(anchor: newAnchor, head: newHead)
    } else {
      newSelection = .cursor(max(0, cursorPos + cursorDelta))
    }

    return EditorState(markdown: newMarkdown, selection: newSelection)
  }
}
