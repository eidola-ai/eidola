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

  // Regex matching an empty checkbox list marker: leading whitespace + marker + " [ ] " or " [x] "
  private static let checkboxMarkerPattern = try! NSRegularExpression(
    pattern: #"^([ \t]*)([-*+]) \[[xX ]\] $"#, options: [])

  // Regex matching a non-empty checkbox list item line (marker + checkbox + content).
  private static let checkboxItemPattern = try! NSRegularExpression(
    pattern: #"^([ \t]*[-*+] \[[xX ]\] ).+"#, options: [])

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

  // Regex matching an empty blockquote line: just "> " with no content.
  private static let emptyBlockquotePattern = try! NSRegularExpression(
    pattern: #"^> $"#, options: [])

  // Regex matching a non-empty blockquote line: "> " followed by content.
  private static let blockquoteItemPattern = try! NSRegularExpression(
    pattern: #"^> .+"#, options: [])

  // Regex matching leading blockquote prefixes: one or more `> ` sequences.
  private static let blockquotePrefixPattern = try! NSRegularExpression(
    pattern: #"^((?:> )+)"#, options: [])

  /// Extract the blockquote prefix (e.g. `"> "`, `"> > "`) from the start of a line.
  /// Returns the prefix string and the remaining line text with the prefix stripped.
  private static func extractBlockquotePrefix(_ lineText: String) -> (prefix: String, stripped: String) {
    let lineNS = lineText as NSString
    if let match = blockquotePrefixPattern.firstMatch(
      in: lineText, range: NSRange(location: 0, length: lineNS.length))
    {
      let prefix = lineNS.substring(with: match.range(at: 1))
      let stripped = lineNS.substring(from: match.range(at: 1).length)
      return (prefix, stripped)
    }
    return ("", lineText)
  }

  /// Compute the next editor state from the current state and an event.
  static func update(_ state: EditorState, event: EditorEvent) -> EditorState {
    let newState: EditorState
    switch event {
    case .insertText(let text):
      newState = handleInsertText(state, text: text)

    case .insertNewline:
      newState = handleInsertNewline(state)

    case .insertLineBreak:
      newState = handleInsertLineBreak(state)

    case .deleteBackward:
      newState = handleDeleteBackward(state)

    case .deleteForward:
      newState = handleDeleteForward(state)

    case .deleteToBeginningOfLine:
      newState = handleDeleteToBeginningOfLine(state)

    case .deleteToEndOfLine:
      newState = handleDeleteToEndOfLine(state)

    case .deleteWordBackward:
      newState = handleDeleteWordBackward(state)

    case .deleteWordForward:
      newState = handleDeleteWordForward(state)

    case .setSelection(let selection):
      // Selection changes don't mutate text, but they can trigger setext
      // normalization if the cursor moved away from a setext underline.
      let selected = handleSetSelection(state, selection: selection)
      return normalizeSetextHeadings(in: selected)

    case .paste(let text):
      newState = handleInsertText(state, text: text)

    case .indent:
      newState = handleIndent(state)

    case .outdent:
      newState = handleOutdent(state)
    }

    // Post-process: normalize setext headings to ATX, then renumber ordered lists.
    return postProcess(newState)
  }

  /// Run post-processing (renumbering, normalization) on a state.
  /// Called externally when text was mutated outside the Elm loop
  /// (e.g., by NSTextView natively).
  static func postProcess(_ state: EditorState) -> EditorState {
    let normalized = normalizeSetextHeadings(in: state)
    return renumberOrderedLists(in: normalized)
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

    // Extract blockquote prefix (e.g. "> ", "> > ") from the current line, then
    // run list/blockquote detection on the stripped (inner) text. This way the
    // same list-continuation logic works both at top level and inside blockquotes.
    let (bqPrefix, strippedLine) = extractBlockquotePrefix(lineText)
    let strippedNS = strippedLine as NSString

    // Check if the stripped line is an empty checkbox list item (just marker + checkbox, no content).
    let emptyCheckboxMatch = checkboxMarkerPattern.firstMatch(
      in: strippedLine, range: NSRange(location: 0, length: strippedNS.length))
    if emptyCheckboxMatch != nil {
      if bqPrefix.isEmpty {
        // Remove the marker line entirely, leave a blank line.
        let newMarkdown = nsMarkdown.replacingCharacters(in: lineRange, with: "\n")
        return EditorState(markdown: newMarkdown, selection: .cursor(lineRange.location))
      } else {
        // Inside blockquote: remove list marker but keep blockquote prefix.
        let replacement = "\(bqPrefix)\n"
        let newMarkdown = nsMarkdown.replacingCharacters(in: lineRange, with: replacement)
        return EditorState(markdown: newMarkdown, selection: .cursor(lineRange.location + bqPrefix.count))
      }
    }

    // Check if the stripped line is an empty list item (just marker, no content).
    let emptyMatch = listMarkerPattern.firstMatch(
      in: strippedLine, range: NSRange(location: 0, length: strippedNS.length))
    if emptyMatch != nil {
      if bqPrefix.isEmpty {
        // Remove the marker line entirely, leave a blank line.
        let newMarkdown = nsMarkdown.replacingCharacters(in: lineRange, with: "\n")
        return EditorState(markdown: newMarkdown, selection: .cursor(lineRange.location))
      } else {
        // Inside blockquote: remove list marker but keep blockquote prefix.
        let replacement = "\(bqPrefix)\n"
        let newMarkdown = nsMarkdown.replacingCharacters(in: lineRange, with: replacement)
        return EditorState(markdown: newMarkdown, selection: .cursor(lineRange.location + bqPrefix.count))
      }
    }

    // Check if the stripped line is an empty ordered list item.
    let emptyOrderedMatch = orderedListMarkerPattern.firstMatch(
      in: strippedLine, range: NSRange(location: 0, length: strippedNS.length))
    if emptyOrderedMatch != nil {
      if bqPrefix.isEmpty {
        // Remove the marker line entirely, leave a blank line.
        let newMarkdown = nsMarkdown.replacingCharacters(in: lineRange, with: "\n")
        return EditorState(markdown: newMarkdown, selection: .cursor(lineRange.location))
      } else {
        // Inside blockquote: remove list marker but keep blockquote prefix.
        let replacement = "\(bqPrefix)\n"
        let newMarkdown = nsMarkdown.replacingCharacters(in: lineRange, with: replacement)
        return EditorState(markdown: newMarkdown, selection: .cursor(lineRange.location + bqPrefix.count))
      }
    }

    // Check if the stripped line is a non-empty ordered list item.
    let orderedItemMatch = orderedListItemPattern.firstMatch(
      in: strippedLine, range: NSRange(location: 0, length: strippedNS.length))
    if let orderedItemMatch = orderedItemMatch {
      // Extract the current number and increment it.
      let indentRange = orderedItemMatch.range(at: 1)
      let indent = strippedNS.substring(with: indentRange)
      let numberRange = orderedItemMatch.range(at: 2)
      let numberStr = strippedNS.substring(with: numberRange)
      let currentNumber = Int(numberStr) ?? 1
      let nextNumber = currentNumber + 1
      let prefix = "\(bqPrefix)\(indent)\(nextNumber). "

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

    // Check if the stripped line is a non-empty checkbox list item.
    let checkboxItemMatch = checkboxItemPattern.firstMatch(
      in: strippedLine, range: NSRange(location: 0, length: strippedNS.length))
    if let checkboxItemMatch = checkboxItemMatch {
      // Continue the list with a new unchecked checkbox.
      let fullPrefixRange = checkboxItemMatch.range(at: 1)
      let fullPrefix = strippedNS.substring(with: fullPrefixRange)
      let checkboxRegex = try! NSRegularExpression(pattern: #"^([ \t]*[-*+] )\[[xX ]\] $"#, options: [])
      let basePrefix: String
      if let baseMatch = checkboxRegex.firstMatch(
        in: fullPrefix, range: NSRange(location: 0, length: (fullPrefix as NSString).length))
      {
        basePrefix = (fullPrefix as NSString).substring(with: baseMatch.range(at: 1)) + "[ ] "
      } else {
        basePrefix = fullPrefix  // fallback
      }

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
      let insertion = "\n\(bqPrefix)\(basePrefix)"
      let newMarkdown = workNS.replacingCharacters(
        in: NSRange(location: workPos, length: 0), with: insertion)
      let newPos = workPos + (insertion as NSString).length
      return EditorState(markdown: newMarkdown, selection: .cursor(newPos))
    }

    // Check if the stripped line is a non-empty unordered list item.
    let itemMatch = listItemPattern.firstMatch(
      in: strippedLine, range: NSRange(location: 0, length: strippedNS.length))
    if let itemMatch = itemMatch {
      // Continue the list: insert newline + blockquote prefix + same marker prefix.
      let prefixRange = itemMatch.range(at: 1)
      let prefix = strippedNS.substring(with: prefixRange)

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
      let insertion = "\n\(bqPrefix)\(prefix)"
      let newMarkdown = workNS.replacingCharacters(
        in: NSRange(location: workPos, length: 0), with: insertion)
      let newPos = workPos + (insertion as NSString).length
      return EditorState(markdown: newMarkdown, selection: .cursor(newPos))
    }

    // Check if the current line is an empty blockquote (just "> " prefix(es), no content after stripping).
    // This handles both simple "> " and nested "> > " empty blockquote lines.
    if !bqPrefix.isEmpty && strippedLine.isEmpty {
      // Remove the marker line entirely, leave a blank line.
      let newMarkdown = nsMarkdown.replacingCharacters(in: lineRange, with: "\n")
      return EditorState(markdown: newMarkdown, selection: .cursor(lineRange.location))
    }

    // Check if the current line is a non-empty blockquote line (has prefix and content).
    if !bqPrefix.isEmpty && !strippedLine.isEmpty {
      // Continue the blockquote: insert newline + same blockquote prefix.
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
      let insertion = "\n\(bqPrefix)"
      let newMarkdown = workNS.replacingCharacters(
        in: NSRange(location: workPos, length: 0), with: insertion)
      let newPos = workPos + (insertion as NSString).length
      return EditorState(markdown: newMarkdown, selection: .cursor(newPos))
    }

    // Check if we're on a continuation line (indented, no marker) belonging
    // to a parent list item. If so, create a new list item.
    if let parent = findParentListItem(in: nsMarkdown, at: pos) {
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
      let newPrefix: String
      if parent.isOrdered {
        let nextNumber = parent.number + 1
        newPrefix = "\(parent.indent)\(nextNumber). "
      } else {
        newPrefix = parent.prefix
      }
      let insertion = "\n\(newPrefix)"
      let newMarkdown = workNS.replacingCharacters(
        in: NSRange(location: workPos, length: 0), with: insertion)
      let newPos = workPos + (insertion as NSString).length
      return EditorState(markdown: newMarkdown, selection: .cursor(newPos))
    }

    // Default: plain newline insertion.
    return handleInsertText(state, text: "\n")
  }

  // Regex matching any list line (unordered or ordered) to extract the full marker width.
  private static let anyListMarkerWidthPattern = try! NSRegularExpression(
    pattern: #"^([ \t]*)([-*+]|\d+\.) "#, options: [])

  private static func handleInsertLineBreak(_ state: EditorState) -> EditorState {
    let nsMarkdown = state.markdown as NSString

    // Resolve the insertion position (delete selection first if range).
    var workMarkdown = state.markdown
    var workPos: Int
    switch state.selection {
    case .cursor(let p):
      workPos = min(p, nsMarkdown.length)
    case .range(let anchor, let head):
      let start = min(anchor, head)
      let end = max(anchor, head)
      let clampedStart = min(start, nsMarkdown.length)
      let clampedEnd = min(end, nsMarkdown.length)
      let deleteRange = NSRange(location: clampedStart, length: clampedEnd - clampedStart)
      workMarkdown = nsMarkdown.replacingCharacters(in: deleteRange, with: "")
      workPos = clampedStart
    }

    let workNS = workMarkdown as NSString
    let (_, lineText) = currentLine(workNS, at: workPos)

    // Extract blockquote prefix so continuation lines inside blockquotes work.
    let (bqPrefix, strippedLine) = extractBlockquotePrefix(lineText)
    let strippedNS = strippedLine as NSString

    // Check if the stripped line starts with a checkbox list marker.
    let checkboxLinePattern = try! NSRegularExpression(
      pattern: #"^([ \t]*)([-*+]) \[[xX ]\] "#, options: [])
    if let match = checkboxLinePattern.firstMatch(
      in: strippedLine, range: NSRange(location: 0, length: strippedNS.length))
    {
      let markerWidth = match.range.length
      let indent = String(repeating: " ", count: markerWidth)
      let insertion = "\n\(bqPrefix)\(indent)"
      let newMarkdown = workNS.replacingCharacters(
        in: NSRange(location: workPos, length: 0), with: insertion)
      let newPos = workPos + (insertion as NSString).length
      return EditorState(markdown: newMarkdown, selection: .cursor(newPos))
    }

    // Check if the stripped line starts with a list marker.
    if let match = anyListMarkerWidthPattern.firstMatch(
      in: strippedLine, range: NSRange(location: 0, length: strippedNS.length))
    {
      let markerWidth = match.range.length
      let indent = String(repeating: " ", count: markerWidth)
      let insertion = "\n\(bqPrefix)\(indent)"
      let newMarkdown = workNS.replacingCharacters(
        in: NSRange(location: workPos, length: 0), with: insertion)
      let newPos = workPos + (insertion as NSString).length
      return EditorState(markdown: newMarkdown, selection: .cursor(newPos))
    }

    // Check if we're on a continuation line (indented, no marker) belonging
    // to a parent list item.
    if let parent = findParentListItem(in: workNS, at: workPos) {
      let markerWidth = parent.prefix.count
      let indent = String(repeating: " ", count: markerWidth)
      let insertion = "\n\(bqPrefix)\(indent)"
      let newMarkdown = workNS.replacingCharacters(
        in: NSRange(location: workPos, length: 0), with: insertion)
      let newPos = workPos + (insertion as NSString).length
      return EditorState(markdown: newMarkdown, selection: .cursor(newPos))
    }

    // Check if the current line is a blockquote line (has any `> ` prefix).
    if !bqPrefix.isEmpty {
      let insertion = "\n\(bqPrefix)"
      let newMarkdown = workNS.replacingCharacters(
        in: NSRange(location: workPos, length: 0), with: insertion)
      let newPos = workPos + (insertion as NSString).length
      return EditorState(markdown: newMarkdown, selection: .cursor(newPos))
    }

    // Not in a list item: plain newline.
    return handleInsertText(
      EditorState(markdown: workMarkdown, selection: .cursor(workPos)),
      text: "\n")
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
      // Inside blockquotes, strip the blockquote prefix first, then check for list markers.
      let (lineRange, lineText) = currentLine(nsMarkdown, at: clampedPos)
      let lineNS = lineText as NSString
      let posInLine = clampedPos - lineRange.location

      let (bqPrefix, strippedLine) = extractBlockquotePrefix(lineText)
      let strippedNS = strippedLine as NSString
      let posInStripped = posInLine - bqPrefix.count

      // Match checkbox: "  - [ ] " or "- [x] " at start of (stripped) line
      let checkboxMarkerRegex = try! NSRegularExpression(
        pattern: #"^([ \t]*)([-*+]) \[[xX ]\] "#, options: [])
      if let match = checkboxMarkerRegex.firstMatch(
        in: strippedLine, range: NSRange(location: 0, length: strippedNS.length))
      {
        let markerEnd = match.range.location + match.range.length
        if posInStripped == markerEnd {
          // Cursor is right after checkbox marker — remove the entire marker (keep bqPrefix).
          let markerAbsRange = NSRange(
            location: lineRange.location + bqPrefix.count + match.range.location,
            length: match.range.length)
          let newMarkdown = nsMarkdown.replacingCharacters(in: markerAbsRange, with: "")
          return EditorState(
            markdown: newMarkdown, selection: .cursor(markerAbsRange.location))
        }
      }

      // Match unordered: "  - " or "- " at start of (stripped) line
      let markerRegex = try! NSRegularExpression(pattern: #"^([ \t]*)([-*+]) "#, options: [])
      if let match = markerRegex.firstMatch(
        in: strippedLine, range: NSRange(location: 0, length: strippedNS.length))
      {
        let markerEnd = match.range.location + match.range.length
        if posInStripped == markerEnd {
          // Cursor is right after marker — remove the entire marker (keep bqPrefix).
          let markerAbsRange = NSRange(
            location: lineRange.location + bqPrefix.count + match.range.location,
            length: match.range.length)
          let newMarkdown = nsMarkdown.replacingCharacters(in: markerAbsRange, with: "")
          return EditorState(
            markdown: newMarkdown, selection: .cursor(markerAbsRange.location))
        }
      }

      // Match ordered: "  1. " or "1. " at start of (stripped) line
      let orderedMarkerRegex = try! NSRegularExpression(
        pattern: #"^([ \t]*)(\d+)\. "#, options: [])
      if let match = orderedMarkerRegex.firstMatch(
        in: strippedLine, range: NSRange(location: 0, length: strippedNS.length))
      {
        let markerEnd = match.range.location + match.range.length
        if posInStripped == markerEnd {
          // Cursor is right after ordered marker — remove the entire marker (keep bqPrefix).
          let markerAbsRange = NSRange(
            location: lineRange.location + bqPrefix.count + match.range.location,
            length: match.range.length)
          let newMarkdown = nsMarkdown.replacingCharacters(in: markerAbsRange, with: "")
          return EditorState(
            markdown: newMarkdown, selection: .cursor(markerAbsRange.location))
        }
      }

      // Match blockquote: "> " at start of line (handles removing one level of blockquote)
      let blockquoteMarkerRegex = try! NSRegularExpression(pattern: #"^((?:> )*> )"#, options: [])
      if let match = blockquoteMarkerRegex.firstMatch(
        in: lineText, range: NSRange(location: 0, length: lineNS.length))
      {
        let fullPrefixEnd = match.range.location + match.range.length
        if posInLine == fullPrefixEnd {
          // Cursor is right after the last "> " — remove just the last "> " (2 chars).
          let lastBqStart = fullPrefixEnd - 2
          let markerAbsRange = NSRange(
            location: lineRange.location + lastBqStart,
            length: 2)
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

  private static func handleDeleteToBeginningOfLine(_ state: EditorState) -> EditorState {
    let nsMarkdown = state.markdown as NSString

    // If there's a selection, delete it (same as deleteBackward with selection).
    if case .range = state.selection {
      return handleDeleteBackward(state)
    }

    guard case .cursor(let pos) = state.selection else { return state }
    let clampedPos = min(pos, nsMarkdown.length)
    guard clampedPos > 0 else { return state }

    let lineRange = nsMarkdown.lineRange(for: NSRange(location: clampedPos, length: 0))
    let lineStart = lineRange.location

    // If cursor is already at the start of the line, no-op.
    guard clampedPos > lineStart else { return state }

    let deleteRange = NSRange(location: lineStart, length: clampedPos - lineStart)
    let newMarkdown = nsMarkdown.replacingCharacters(in: deleteRange, with: "")
    return EditorState(markdown: newMarkdown, selection: .cursor(lineStart))
  }

  private static func handleDeleteToEndOfLine(_ state: EditorState) -> EditorState {
    let nsMarkdown = state.markdown as NSString

    // If there's a selection, delete it.
    if case .range = state.selection {
      return handleDeleteBackward(state)
    }

    guard case .cursor(let pos) = state.selection else { return state }
    let clampedPos = min(pos, nsMarkdown.length)
    guard clampedPos < nsMarkdown.length else { return state }

    let lineRange = nsMarkdown.lineRange(for: NSRange(location: clampedPos, length: 0))
    let lineEnd = lineRange.location + lineRange.length

    // Exclude the trailing \n from deletion (delete to end of line content, not the newline).
    let contentEnd: Int
    if lineEnd > lineRange.location
      && lineEnd <= nsMarkdown.length
      && nsMarkdown.character(at: lineEnd - 1) == UInt16(0x000A)
    {
      contentEnd = lineEnd - 1
    } else {
      contentEnd = lineEnd
    }

    // If cursor is already at the end of the line content, no-op.
    guard clampedPos < contentEnd else { return state }

    let deleteRange = NSRange(location: clampedPos, length: contentEnd - clampedPos)
    let newMarkdown = nsMarkdown.replacingCharacters(in: deleteRange, with: "")
    return EditorState(markdown: newMarkdown, selection: .cursor(clampedPos))
  }

  private static func handleDeleteWordBackward(_ state: EditorState) -> EditorState {
    let nsMarkdown = state.markdown as NSString

    // If there's a selection, delete it.
    if case .range = state.selection {
      return handleDeleteBackward(state)
    }

    guard case .cursor(let pos) = state.selection else { return state }
    let clampedPos = min(pos, nsMarkdown.length)
    guard clampedPos > 0 else { return state }

    // Skip whitespace backwards, then skip non-whitespace backwards.
    var target = clampedPos
    while target > 0 {
      let ch = Character(UnicodeScalar(nsMarkdown.character(at: target - 1))!)
      if ch.isWhitespace { target -= 1 } else { break }
    }
    while target > 0 {
      let ch = Character(UnicodeScalar(nsMarkdown.character(at: target - 1))!)
      if !ch.isWhitespace { target -= 1 } else { break }
    }

    let deleteRange = NSRange(location: target, length: clampedPos - target)
    let newMarkdown = nsMarkdown.replacingCharacters(in: deleteRange, with: "")
    return EditorState(markdown: newMarkdown, selection: .cursor(target))
  }

  private static func handleDeleteWordForward(_ state: EditorState) -> EditorState {
    let nsMarkdown = state.markdown as NSString

    // If there's a selection, delete it.
    if case .range = state.selection {
      return handleDeleteBackward(state)
    }

    guard case .cursor(let pos) = state.selection else { return state }
    let clampedPos = min(pos, nsMarkdown.length)
    guard clampedPos < nsMarkdown.length else { return state }

    // Skip non-whitespace forward, then skip whitespace forward.
    var target = clampedPos
    while target < nsMarkdown.length {
      let ch = Character(UnicodeScalar(nsMarkdown.character(at: target))!)
      if !ch.isWhitespace { target += 1 } else { break }
    }
    while target < nsMarkdown.length {
      let ch = Character(UnicodeScalar(nsMarkdown.character(at: target))!)
      if ch.isWhitespace { target += 1 } else { break }
    }

    let deleteRange = NSRange(location: clampedPos, length: target - clampedPos)
    let newMarkdown = nsMarkdown.replacingCharacters(in: deleteRange, with: "")
    return EditorState(markdown: newMarkdown, selection: .cursor(clampedPos))
  }

  /// Regex matching any list marker at the start of a line (unordered or ordered).
  private static let anyListLinePattern = try! NSRegularExpression(
    pattern: #"^([ \t]*)([-*+]|\d+\.) "#, options: [])

  private static func handleIndent(_ state: EditorState) -> EditorState {
    let nsMarkdown = state.markdown as NSString
    let pos: Int
    switch state.selection {
    case .cursor(let p): pos = min(p, nsMarkdown.length)
    case .range(_, let head): pos = min(head, nsMarkdown.length)
    }

    let (lineRange, lineText) = currentLine(nsMarkdown, at: pos)

    // Strip blockquote prefix for list detection.
    let (bqPrefix, strippedLine) = extractBlockquotePrefix(lineText)
    let strippedNS = strippedLine as NSString

    // Check if the (stripped) line is a list item line
    let listMatch = anyListLinePattern.firstMatch(
      in: strippedLine, range: NSRange(location: 0, length: strippedNS.length))

    if listMatch != nil {
      // On a list item line: add 4 spaces before the marker (after blockquote prefix)
      let indent = "    "
      let insertPos = lineRange.location + bqPrefix.count
      let newMarkdown = nsMarkdown.replacingCharacters(
        in: NSRange(location: insertPos, length: 0), with: indent)
      let newPos = pos + 4
      return EditorState(markdown: newMarkdown, selection: .cursor(newPos))
    }

    // Check if we're on a continuation line belonging to a list item
    if findParentListItem(in: nsMarkdown, at: pos) != nil {
      let indent = "    "
      let insertPos = lineRange.location + bqPrefix.count
      let newMarkdown = nsMarkdown.replacingCharacters(
        in: NSRange(location: insertPos, length: 0), with: indent)
      let newPos = pos + 4
      return EditorState(markdown: newMarkdown, selection: .cursor(newPos))
    }

    // Not on a list line: insert 4 spaces at cursor position
    return handleInsertText(state, text: "    ")
  }

  private static func handleOutdent(_ state: EditorState) -> EditorState {
    let nsMarkdown = state.markdown as NSString
    let pos: Int
    switch state.selection {
    case .cursor(let p): pos = min(p, nsMarkdown.length)
    case .range(_, let head): pos = min(head, nsMarkdown.length)
    }

    let (lineRange, lineText) = currentLine(nsMarkdown, at: pos)

    // Strip blockquote prefix for list detection.
    let (bqPrefix, strippedLine) = extractBlockquotePrefix(lineText)
    let strippedNS = strippedLine as NSString

    // Check if the (stripped) line is a list item line
    let listMatch = anyListLinePattern.firstMatch(
      in: strippedLine, range: NSRange(location: 0, length: strippedNS.length))

    if listMatch != nil {
      // Count leading spaces after blockquote prefix
      let leadingSpaces = strippedLine.prefix(while: { $0 == " " })
      let spacesToRemove = min(leadingSpaces.count, 4)
      guard spacesToRemove > 0 else { return state }

      let removeStart = lineRange.location + bqPrefix.count
      let removeRange = NSRange(location: removeStart, length: spacesToRemove)
      let newMarkdown = nsMarkdown.replacingCharacters(in: removeRange, with: "")
      let newPos = max(removeStart, pos - spacesToRemove)
      return EditorState(markdown: newMarkdown, selection: .cursor(newPos))
    }

    // Check if we're on a continuation line belonging to a list item
    if findParentListItem(in: nsMarkdown, at: pos) != nil {
      let leadingSpaces = strippedLine.prefix(while: { $0 == " " })
      let spacesToRemove = min(leadingSpaces.count, 4)
      guard spacesToRemove > 0 else { return state }

      let removeStart = lineRange.location + bqPrefix.count
      let removeRange = NSRange(location: removeStart, length: spacesToRemove)
      let newMarkdown = nsMarkdown.replacingCharacters(in: removeRange, with: "")
      let newPos = max(removeStart, pos - spacesToRemove)
      return EditorState(markdown: newMarkdown, selection: .cursor(newPos))
    }

    // Not on a list line: do nothing
    return state
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

  // MARK: - Continuation Line Detection

  /// Information about a parent list item found by walking backwards from a continuation line.
  private struct ParentListItem {
    /// The full marker prefix (e.g. "- ", "  - ", "1. ", "  1. ")
    let prefix: String
    /// Whether the parent is an ordered list item.
    let isOrdered: Bool
    /// The number of the ordered list item (only meaningful if isOrdered).
    let number: Int
    /// The leading indent string.
    let indent: String
    /// The marker character for unordered lists (e.g. "-", "*", "+").
    let markerChar: String
  }

  /// Regex matching any list marker line to extract components.
  private static let anyListMarkerPattern = try! NSRegularExpression(
    pattern: #"^([ \t]*)([-*+]) "#, options: [])
  private static let anyOrderedMarkerPattern = try! NSRegularExpression(
    pattern: #"^([ \t]*)(\d+)\. "#, options: [])

  /// Walk backwards from the current line to find the parent list item for a continuation line.
  /// Returns nil if the current line is not a continuation line.
  private static func findParentListItem(
    in nsMarkdown: NSString, at pos: Int
  ) -> ParentListItem? {
    let (lineRange, lineText) = currentLine(nsMarkdown, at: pos)
    let lineNS = lineText as NSString

    // Check if the current line is itself a list marker line (not a continuation)
    if anyListMarkerWidthPattern.firstMatch(
      in: lineText, range: NSRange(location: 0, length: lineNS.length)) != nil
    {
      return nil
    }

    // Check if the current line starts with whitespace (potential continuation)
    let leadingSpaces = lineText.prefix(while: { $0 == " " || $0 == "\t" })
    guard !leadingSpaces.isEmpty else { return nil }

    // Walk backwards through previous lines
    var searchPos = lineRange.location
    while searchPos > 0 {
      // Move to previous line
      let prevLineEnd = searchPos - 1
      let (prevLineRange, prevLineText) = currentLine(nsMarkdown, at: max(0, prevLineEnd))
      let prevNS = prevLineText as NSString

      // Check for unordered list marker
      if let match = anyListMarkerPattern.firstMatch(
        in: prevLineText, range: NSRange(location: 0, length: prevNS.length))
      {
        let indent = prevNS.substring(with: match.range(at: 1))
        let marker = prevNS.substring(with: match.range(at: 2))
        let prefix = "\(indent)\(marker) "
        // Verify our line's indentation matches or exceeds the marker width
        if leadingSpaces.count >= prefix.count {
          return ParentListItem(
            prefix: prefix, isOrdered: false, number: 0,
            indent: indent, markerChar: marker)
        }
        return nil
      }

      // Check for ordered list marker
      if let match = anyOrderedMarkerPattern.firstMatch(
        in: prevLineText, range: NSRange(location: 0, length: prevNS.length))
      {
        let indent = prevNS.substring(with: match.range(at: 1))
        let numberStr = prevNS.substring(with: match.range(at: 2))
        let number = Int(numberStr) ?? 1
        let prefix = "\(indent)\(numberStr). "
        // Verify our line's indentation matches or exceeds the marker width
        if leadingSpaces.count >= prefix.count {
          return ParentListItem(
            prefix: prefix, isOrdered: true, number: number,
            indent: indent, markerChar: "")
        }
        return nil
      }

      // Check if this previous line is also a continuation (starts with whitespace)
      let prevLeading = prevLineText.prefix(while: { $0 == " " || $0 == "\t" })
      guard !prevLeading.isEmpty else { return nil }

      // Continue walking backwards
      searchPos = prevLineRange.location
    }

    return nil
  }

  // MARK: - Ordered List Renumbering

  /// Scan all lines and renumber ordered list items so each contiguous run
  // MARK: - Setext Heading Normalization

  /// Setext heading pattern: a non-empty line followed by a line of only `=` or `-` (1+).
  private static let setextPattern = try! NSRegularExpression(
    pattern: #"^(.+)\n(=+|-+)[ \t]*$"#, options: [.anchorsMatchLines])

  /// Pattern matching a line that is purely `=` or `-` characters (with optional trailing whitespace).
  /// Used to detect if the cursor is on a setext underline.
  private static let setextUnderlinePattern = try! NSRegularExpression(
    pattern: #"^(=+|-+)[ \t]*$"#, options: [])

  /// Convert setext-style headings to ATX format (`# heading`).
  ///
  /// Normalizes when the cursor is NOT on a setext underline line. This means:
  /// - While typing `=` or `-` after text, no heading conversion occurs (cursor is on the underline)
  /// - As soon as the cursor moves away (click, arrow, Enter, etc.), the setext heading
  ///   is converted to ATX format and the underline disappears
  private static func normalizeSetextHeadings(in state: EditorState) -> EditorState {
    let text = state.markdown
    let nsText = text as NSString
    let cursorPos = state.selection.head

    // Check if cursor is currently on a setext underline line — if so, skip normalization.
    if cursorPos <= nsText.length {
      let cursorLineRange = nsText.lineRange(for: NSRange(location: cursorPos, length: 0))
      let cursorLine = nsText.substring(with: cursorLineRange)
        .trimmingCharacters(in: .newlines)
      if !cursorLine.isEmpty,
        setextUnderlinePattern.firstMatch(
          in: cursorLine, range: NSRange(location: 0, length: (cursorLine as NSString).length)) != nil
      {
        return state
      }
    }

    let matches = setextPattern.matches(in: text, range: NSRange(location: 0, length: nsText.length))
    guard !matches.isEmpty else { return state }

    var result = text
    var cursorOffset = 0

    // Process matches in reverse to preserve ranges
    for match in matches.reversed() {
      let fullRange = match.range
      let titleRange = match.range(at: 1)
      let underlineRange = match.range(at: 2)

      let title = nsText.substring(with: titleRange)
      let underline = nsText.substring(with: underlineRange)
      let level = underline.hasPrefix("=") ? 1 : 2
      let prefix = String(repeating: "#", count: level)
      let replacement = "\(prefix) \(title)"

      let nsResult = result as NSString
      result = nsResult.replacingCharacters(in: fullRange, with: replacement)

      // Adjust cursor if it's after this match
      let lengthDelta = (replacement as NSString).length - fullRange.length
      if cursorPos > fullRange.location + fullRange.length {
        cursorOffset += lengthDelta
      } else if cursorPos > fullRange.location {
        // Cursor is inside the setext heading — place it in the ATX heading content
        let posInMatch = cursorPos - fullRange.location
        let newPos = fullRange.location + min(posInMatch, (replacement as NSString).length)
        cursorOffset += newPos - cursorPos
      }
    }

    let newCursorPos = max(0, cursorPos + cursorOffset)
    return EditorState(
      markdown: result,
      selection: .cursor(min(newCursorPos, (result as NSString).length)))
  }

  // MARK: - Ordered List Renumbering

  /// of same-indent ordered items is numbered sequentially starting from 1.
  /// Adjusts cursor position if renumbering changes character counts before it.
  private static func renumberOrderedLists(in state: EditorState) -> EditorState {
    let lines = state.markdown.components(separatedBy: "\n")
    var newLines: [String] = []

    // Track running counters per (blockquote prefix + indent level).
    // Key: bqPrefix + indent, Value: next expected number.
    // Scoping by blockquote prefix ensures lists at the top level,
    // inside `> `, and inside `> > ` are renumbered independently.
    var counters: [String: Int] = [:]
    var activeIndents: Set<String> = []

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
    var charOffset = 0

    var lastMarkerWidth: [String: Int] = [:]

    for line in lines {
      let lineNS = line as NSString

      // Strip blockquote prefix before matching ordered list patterns.
      let (bqPrefix, strippedLine) = extractBlockquotePrefix(line)
      let strippedNS = strippedLine as NSString
      let match = orderedLinePattern.firstMatch(
        in: strippedLine, range: NSRange(location: 0, length: strippedNS.length))

      if let match = match {
        let indent = strippedNS.substring(with: match.range(at: 1))
        let oldNumberStr = strippedNS.substring(with: match.range(at: 2))
        let scopeKey = "\(bqPrefix)\(indent)"

        if !activeIndents.contains(scopeKey) {
          counters[scopeKey] = 1
        }

        let correctNumber = counters[scopeKey] ?? 1
        let correctNumberStr = String(correctNumber)
        counters[scopeKey] = correctNumber + 1

        activeIndents.insert(scopeKey)
        lastMarkerWidth[scopeKey] = match.range.length

        if correctNumberStr != oldNumberStr {
          let oldPrefix = "\(indent)\(oldNumberStr). "
          let newPrefix = "\(indent)\(correctNumberStr). "
          let rest = strippedNS.substring(from: match.range.length)
          let newLine = "\(bqPrefix)\(newPrefix)\(rest)"
          newLines.append(newLine)

          let lengthDiff = (newPrefix as NSString).length - (oldPrefix as NSString).length

          let oldFullPrefixEnd = charOffset + (bqPrefix as NSString).length
            + (oldPrefix as NSString).length
          if cursorPos >= oldFullPrefixEnd {
            cursorDelta += lengthDiff
          } else if cursorPos > charOffset {
            cursorDelta += lengthDiff
          }

          if let anchor = cursorAnchor {
            if anchor >= oldFullPrefixEnd {
              anchorDelta += lengthDiff
            } else if anchor > charOffset {
              anchorDelta += lengthDiff
            }
          }
        } else {
          newLines.append(line)
        }
      } else {
        // Not an ordered list line. Check for continuation lines using
        // the stripped content (after removing blockquote prefix).
        let isContinuation: Bool
        if !activeIndents.isEmpty, !strippedLine.isEmpty {
          let leadingSpaces = strippedLine.prefix(while: { $0 == " " || $0 == "\t" })
          if !leadingSpaces.isEmpty {
            isContinuation = activeIndents.contains(where: { key in
              guard key.hasPrefix(bqPrefix),
                let mw = lastMarkerWidth[key]
              else { return false }
              return leadingSpaces.count >= mw
            })
          } else {
            isContinuation = false
          }
        } else {
          isContinuation = false
        }

        if !isContinuation {
          // Clear only the scope entries matching this blockquote prefix.
          // Lines at a different prefix level don't break other scopes.
          let keysToRemove = activeIndents.filter { $0.hasPrefix(bqPrefix) }
          activeIndents.subtract(keysToRemove)
          for key in keysToRemove { lastMarkerWidth.removeValue(forKey: key) }
        }
        newLines.append(line)
      }

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
