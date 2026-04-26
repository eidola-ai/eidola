import Foundation
import Markdown

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

  // MARK: - Nested Prefix Parser

  /// A single structural component in a line's nesting prefix.
  private enum PrefixComponent: Equatable {
    /// A blockquote marker: `> ` or bare `>` (1-2 characters).
    case blockquote
    /// An unordered list marker: `- `, `* `, or `+ ` (2 characters).
    case unordered(marker: Character)
    /// An ordered list marker: `1. `, `12. `, etc. (number + ". ").
    case ordered(number: Int)
    /// A checkbox following an unordered marker: `[ ] ` or `[x] ` (4 characters).
    case checkbox(checked: Bool)
    /// Whitespace indentation between structural markers.
    case indent(String)

    /// Whether this component is a list marker (unordered or ordered).
    var isListMarker: Bool {
      switch self {
      case .unordered, .ordered: return true
      default: return false
      }
    }

    /// The character width of this component.
    var charWidth: Int {
      switch self {
      case .blockquote: return 2  // "> "
      case .unordered: return 2   // "- "
      case .ordered(let n): return "\(n). ".count
      case .checkbox: return 4    // "[ ] "
      case .indent(let s): return s.count
      }
    }
  }

  /// The fully parsed prefix of a markdown line.
  private struct ParsedPrefix {
    /// The sequence of nesting components, outermost first.
    let components: [PrefixComponent]
    /// Number of characters consumed by the prefix.
    let prefixLength: Int
    /// The remaining content after the prefix (may include trailing newline).
    let content: String
  }

  /// Parse a line's prefix into nested structural components.
  ///
  /// Scans left-to-right, greedily matching blockquote markers, list markers,
  /// checkboxes, and structural indentation. Stops when no structural marker
  /// follows (the remainder is content).
  private static func parsePrefix(_ lineText: String) -> ParsedPrefix {
    let ns = lineText as NSString
    var components: [PrefixComponent] = []
    var pos = 0

    while pos < ns.length {
      let ch = ns.character(at: pos)

      // 1. Blockquote: `>` followed by optional space
      if ch == 0x003E {  // >
        pos += 1
        if pos < ns.length && ns.character(at: pos) == 0x0020 {
          pos += 1
        }
        components.append(.blockquote)
        continue
      }

      // 2. Whitespace — only consume as structural indent if followed by a marker
      if ch == 0x0020 || ch == 0x0009 {
        let wsStart = pos
        while pos < ns.length {
          let c = ns.character(at: pos)
          guard c == 0x0020 || c == 0x0009 else { break }
          pos += 1
        }
        if pos < ns.length && isStructuralMarkerStart(ns.character(at: pos)) {
          components.append(.indent(ns.substring(with: NSRange(location: wsStart, length: pos - wsStart))))
          continue
        }
        // Not structural — backtrack
        pos = wsStart
        break
      }

      // 3. Unordered list marker: `-`, `*`, `+` followed by space
      if (ch == 0x002D || ch == 0x002A || ch == 0x002B)
        && pos + 1 < ns.length && ns.character(at: pos + 1) == 0x0020
      {
        let marker = Character(UnicodeScalar(ch)!)
        pos += 2
        components.append(.unordered(marker: marker))

        // Check for checkbox: `[x] ` or `[ ] ` or `[X] `
        if pos + 3 < ns.length
          && ns.character(at: pos) == 0x005B       // [
          && ns.character(at: pos + 2) == 0x005D   // ]
          && ns.character(at: pos + 3) == 0x0020   // space
        {
          let check = ns.character(at: pos + 1)
          if check == 0x0020 || check == 0x0078 || check == 0x0058 {
            components.append(.checkbox(checked: check != 0x0020))
            pos += 4
          }
        }
        continue
      }

      // 4. Ordered list marker: digits followed by `. `
      if ch >= 0x0030 && ch <= 0x0039 {
        let numStart = pos
        while pos < ns.length && ns.character(at: pos) >= 0x0030 && ns.character(at: pos) <= 0x0039 {
          pos += 1
        }
        if pos > numStart
          && pos + 1 < ns.length
          && ns.character(at: pos) == 0x002E       // .
          && ns.character(at: pos + 1) == 0x0020   // space
        {
          let numStr = ns.substring(with: NSRange(location: numStart, length: pos - numStart))
          components.append(.ordered(number: Int(numStr) ?? 1))
          pos += 2
          continue
        }
        // Not a valid ordered marker — backtrack
        pos = numStart
        break
      }

      // 5. Nothing matched — content starts here
      break
    }

    let content = ns.substring(from: pos)
    return ParsedPrefix(components: components, prefixLength: pos, content: content)
  }

  /// Check if a position is inside a fenced code block by counting fence lines
  /// (lines starting with ``` or ~~~) before it. An odd count means inside.
  private static func isInsideFencedCodeBlock(_ nsMarkdown: NSString, at pos: Int) -> Bool {
    var fenceCount = 0
    var searchPos = 0

    while searchPos < nsMarkdown.length {
      let lineRange = nsMarkdown.lineRange(for: NSRange(location: searchPos, length: 0))
      let lineText = nsMarkdown.substring(with: lineRange)
        .trimmingCharacters(in: .whitespacesAndNewlines)
      // Strip blockquote prefixes for nested code blocks inside `> `
      let stripped = lineText.drop(while: { $0 == ">" || $0 == " " || $0 == "\t" })

      if stripped.hasPrefix("```") || stripped.hasPrefix("~~~") {
        fenceCount += 1
      }

      let nextPos = lineRange.location + lineRange.length
      if nextPos <= searchPos { break }
      // Stop after processing the line that contains the cursor.
      if nextPos > pos { break }
      searchPos = nextPos
    }

    return fenceCount % 2 == 1
  }

  /// Check if a character can start a structural marker (blockquote, list, ordered).
  private static func isStructuralMarkerStart(_ ch: unichar) -> Bool {
    ch == 0x003E       // >
      || ch == 0x002D  // -
      || ch == 0x002A  // *
      || ch == 0x002B  // +
      || (ch >= 0x0030 && ch <= 0x0039)  // digit
  }

  /// Rebuild a prefix string from components in their original textual form.
  private static func rebuildPrefix(_ components: [PrefixComponent]) -> String {
    var result = ""
    for comp in components {
      switch comp {
      case .blockquote: result += "> "
      case .unordered(let ch): result += "\(ch) "
      case .ordered(let num): result += "\(num). "
      case .checkbox(let checked): result += checked ? "[x] " : "[ ] "
      case .indent(let s): result += s
      }
    }
    return result
  }

  /// Build a continuation prefix for Enter: outer list markers become equal-width
  /// spaces, the innermost list marker is continued (incremented/repeated).
  /// Blockquotes and indents are preserved.
  ///
  /// - Parameters:
  ///   - components: All prefix components.
  ///   - innermostListIdx: Index of the innermost list marker in `components`.
  /// - Returns: The prefix string for the new line.
  private static func buildContinuationPrefix(
    _ components: [PrefixComponent],
    innermostListIdx: Int
  ) -> String {
    var result = ""
    for (i, comp) in components.enumerated() {
      if i == innermostListIdx {
        // Continue the innermost marker
        switch comp {
        case .unordered(let ch):
          result += "\(ch) "
          // Check for checkbox following this marker
          if i + 1 < components.count, case .checkbox = components[i + 1] {
            result += "[ ] "
          }
        case .ordered(let num):
          result += "\(num + 1). "
        default:
          result += String(repeating: " ", count: comp.charWidth)
        }
      } else if i == innermostListIdx + 1, case .checkbox = comp {
        // Already handled above with the unordered marker
        continue
      } else {
        // Outer components: convert list markers to spaces, keep everything else
        switch comp {
        case .blockquote: result += "> "
        case .indent(let s): result += s
        case .unordered:
          result += "  "
          // If followed by checkbox, also convert that to spaces
          if i + 1 < components.count, case .checkbox = components[i + 1] {
            // Handled in the next iteration
          }
        case .ordered(let num):
          result += String(repeating: " ", count: "\(num). ".count)
        case .checkbox:
          // Check if the preceding unordered is also being converted to spaces
          if i > 0, case .unordered = components[i - 1],
            i - 1 != innermostListIdx
          {
            result += "    "  // 4 spaces for "[ ] "
          }
          // If preceding unordered IS the innermost, this was handled above
        }
      }
    }
    return result
  }

  /// Build a full continuation prefix for Shift+Enter: ALL list markers become
  /// equal-width spaces. Blockquotes and indents are preserved.
  private static func buildFullContinuationPrefix(_ components: [PrefixComponent]) -> String {
    var result = ""
    for comp in components {
      switch comp {
      case .blockquote: result += "> "
      case .indent(let s): result += s
      case .unordered: result += "  "
      case .ordered(let num): result += String(repeating: " ", count: "\(num). ".count)
      case .checkbox: result += "    "
      }
    }
    return result
  }

  /// Check if inserting text at `posInLine` in `lineText` requires a space injection
  /// after a bare blockquote `>`. Returns true when the text up to `posInLine` forms
  /// a valid structural prefix ending with `>` (no trailing space).
  /// This is used by both `handleInsertText` and the coordinator's `shouldChangeTextIn`.
  static func needsBlockquoteSpaceInjection(lineText: String, posInLine: Int) -> Bool {
    guard posInLine > 0 else { return false }
    let prefixText = (lineText as NSString).substring(to: posInLine)
    guard (prefixText as NSString).character(at: posInLine - 1) == 0x003E else { return false }
    // Parse the prefix up to the cursor — if the parser consumes it entirely
    // and the last component is a blockquote, the `>` is structural and needs a space.
    let parsed = parsePrefix(prefixText)
    return parsed.prefixLength == posInLine
      && !parsed.components.isEmpty
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

    // Parse the full nesting prefix (blockquotes, list markers, indentation).
    let parsed = parsePrefix(lineText)
    let contentText = parsed.content.trimmingCharacters(in: CharacterSet.newlines)
    let contentIsEmpty = contentText.allSatisfy { $0 == " " || $0 == "\t" }

    // If the prefix has structural components, use prefix-aware logic.
    if !parsed.components.isEmpty {

      // Find the innermost list marker and the innermost non-indent component.
      let innermostListIdx = parsed.components.lastIndex(where: { $0.isListMarker })
      let innermostStructuralIdx = parsed.components.lastIndex(where: {
        if case .indent = $0 { return false }
        if case .checkbox = $0 { return false }
        return true
      })

      if contentIsEmpty {
        // Empty content — unwind the innermost structural component.
        // Prefer unwinding a list marker over a blockquote; if the innermost
        // structural is a blockquote, unwind that.
        let removeIdx: Int
        if let listIdx = innermostListIdx, let structIdx = innermostStructuralIdx {
          removeIdx = max(listIdx, structIdx)
        } else if let idx = innermostListIdx ?? innermostStructuralIdx {
          removeIdx = idx
        } else {
          // Only indent components — plain newline
          return handleInsertText(state, text: "\n")
        }

        // Remove the component at removeIdx (and a following checkbox, if any).
        // Also remove a preceding indent if the removed component is a list marker,
        // since indentation is part of the list nesting level.
        var trimIdx = removeIdx
        if parsed.components[removeIdx].isListMarker
          && trimIdx > 0, case .indent = parsed.components[trimIdx - 1]
        {
          trimIdx -= 1
        }
        let remaining = Array(parsed.components[..<trimIdx])
        let unwoundPrefix = rebuildPrefix(remaining)

        if unwoundPrefix.isEmpty {
          // No remaining structure — blank line.
          let newMarkdown = nsMarkdown.replacingCharacters(in: lineRange, with: "\n")
          return EditorState(markdown: newMarkdown, selection: .cursor(lineRange.location))
        } else {
          let replacement = "\(unwoundPrefix)\n"
          let newMarkdown = nsMarkdown.replacingCharacters(in: lineRange, with: replacement)
          return EditorState(
            markdown: newMarkdown,
            selection: .cursor(lineRange.location + unwoundPrefix.count))
        }
      }

      // Non-empty content — continue the innermost list, or the prefix structure.
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
      let insertion: String

      if let listIdx = innermostListIdx {
        // Continue the innermost list marker with continuation prefix for outer layers.
        insertion = "\n" + buildContinuationPrefix(parsed.components, innermostListIdx: listIdx)
      } else {
        // No list marker — continue the structural prefix (blockquotes/indents).
        insertion = "\n" + rebuildPrefix(parsed.components)
      }

      let newMarkdown = workNS.replacingCharacters(
        in: NSRange(location: workPos, length: 0), with: insertion)
      let newPos = workPos + (insertion as NSString).length
      return EditorState(markdown: newMarkdown, selection: .cursor(newPos))
    }

    // No structural prefix — check if we're on a continuation line belonging
    // to a parent list item. If so, create a new list item.
    if let parent = findParentListItem(in: nsMarkdown, at: pos) {
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

    // Default: paragraph break or blank line.
    // Inside a fenced code block, always use single \n (literal newlines).
    // In paragraph context, insert \n\n so the raw markdown uses a blank line
    // to separate paragraphs (per CommonMark). On an already-blank line, insert
    // a single \n so consecutive Returns produce visible empty paragraphs.
    if isInsideFencedCodeBlock(nsMarkdown, at: pos) {
      return handleInsertText(state, text: "\n")
    }
    let (_, defaultLineText) = currentLine(nsMarkdown, at: pos)
    let defaultLineContent = defaultLineText.trimmingCharacters(in: .newlines)
    let onBlankLine: Bool
    if case .cursor = state.selection {
      onBlankLine = defaultLineContent.isEmpty
    } else {
      onBlankLine = false
    }
    return handleInsertText(state, text: onBlankLine ? "\n" : "\n\n")
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

    // Parse the full nesting prefix for correct continuation indentation.
    let parsed = parsePrefix(lineText)

    if !parsed.components.isEmpty {
      // Build full continuation: all list markers become equal-width spaces,
      // blockquotes are preserved.
      let continuation = buildFullContinuationPrefix(parsed.components)
      let insertion = "\n\(continuation)"
      let newMarkdown = workNS.replacingCharacters(
        in: NSRange(location: workPos, length: 0), with: insertion)
      let newPos = workPos + (insertion as NSString).length
      return EditorState(markdown: newMarkdown, selection: .cursor(newPos))
    }

    // No structural prefix — check for continuation line belonging to a parent.
    if let parent = findParentListItem(in: workNS, at: workPos) {
      let markerWidth = parent.prefix.count
      let indent = String(repeating: " ", count: markerWidth)
      let insertion = "\n\(indent)"
      let newMarkdown = workNS.replacingCharacters(
        in: NSRange(location: workPos, length: 0), with: insertion)
      let newPos = workPos + (insertion as NSString).length
      return EditorState(markdown: newMarkdown, selection: .cursor(newPos))
    }

    // Not in a list item: paragraph break (same as Return for now).
    if isInsideFencedCodeBlock(workNS, at: workPos) {
      return handleInsertText(
        EditorState(markdown: workMarkdown, selection: .cursor(workPos)),
        text: "\n")
    }
    let (_, shiftLineText) = currentLine(workNS, at: workPos)
    let shiftLineContent = shiftLineText.trimmingCharacters(in: .newlines)
    return handleInsertText(
      EditorState(markdown: workMarkdown, selection: .cursor(workPos)),
      text: shiftLineContent.isEmpty ? "\n" : "\n\n")
  }

  /// Regex matching a bare blockquote prefix (optional leading whitespace, then one
  /// or more `>` with `> ` separators, ending in `>` without a trailing space).
  /// Used to inject a trailing space when the user types content after a bare `>`.
  /// The leading whitespace allows matching blockquotes nested inside list items.
  private static let bareBlockquotePrefixPattern = try! NSRegularExpression(
    pattern: #"^[ \t]*(> )*>$"#, options: [])

  private static func handleInsertText(_ state: EditorState, text: String) -> EditorState {
    let nsMarkdown = state.markdown as NSString

    // Resolve insertion position (and delete selection if range).
    let insertPos: Int
    let baseMarkdown: NSString
    switch state.selection {
    case .cursor(let pos):
      insertPos = min(pos, nsMarkdown.length)
      baseMarkdown = nsMarkdown
    case .range(let anchor, let head):
      let start = min(anchor, head)
      let end = max(anchor, head)
      let clampedStart = min(start, nsMarkdown.length)
      let clampedEnd = min(end, nsMarkdown.length)
      let replaceRange = NSRange(location: clampedStart, length: clampedEnd - clampedStart)
      baseMarkdown = nsMarkdown.replacingCharacters(in: replaceRange, with: "") as NSString
      insertPos = clampedStart
    }

    // Check if we should inject a space after a bare blockquote `>`.
    // This normalizes `>text` to `> text`, matching canonical blockquote format.
    // Uses the prefix parser to handle arbitrarily nested structures like `> >   >`.
    let effectiveText: String
    if !text.hasPrefix(" ") && !text.hasPrefix("\n") && insertPos > 0 {
      let (lineRange, lineText) = currentLine(baseMarkdown, at: insertPos)
      let posInLine = insertPos - lineRange.location
      if posInLine > 0 && needsBlockquoteSpaceInjection(lineText: lineText, posInLine: posInLine) {
        effectiveText = " \(text)"
      } else {
        effectiveText = text
      }
    } else {
      effectiveText = text
    }

    let newMarkdown = baseMarkdown.replacingCharacters(
      in: NSRange(location: insertPos, length: 0), with: effectiveText)
    let newPos = insertPos + (effectiveText as NSString).length
    return EditorState(markdown: newMarkdown, selection: .cursor(newPos))
  }

  private static func handleDeleteBackward(_ state: EditorState) -> EditorState {
    let nsMarkdown = state.markdown as NSString

    switch state.selection {
    case .cursor(let pos):
      guard pos > 0 else { return state }
      let clampedPos = min(pos, nsMarkdown.length)

      // Check if cursor is right after a structural marker (blockquote, list, checkbox).
      // If so, remove the entire marker as a whole unit instead of just one character.
      // Uses the prefix parser to handle arbitrarily nested blockquote/list combinations.
      let (lineRange, lineText) = currentLine(nsMarkdown, at: clampedPos)
      let posInLine = clampedPos - lineRange.location

      let parsed = parsePrefix(lineText)

      // Walk through parsed components, checking if cursor is at a component boundary.
      var componentOffset = 0
      for (i, comp) in parsed.components.enumerated() {
        let width = comp.charWidth
        let componentEnd = componentOffset + width

        if posInLine == componentEnd {
          // Cursor is right after this component — delete it as a whole unit.
          // For checkbox, also delete the preceding unordered marker.
          // For list markers preceded by indent, also delete the indent.
          var deleteStart = componentOffset
          var deleteLen = width
          if case .checkbox = comp, i > 0, case .unordered = parsed.components[i - 1] {
            // Delete both unordered marker + checkbox as a unit
            let unorderedWidth = parsed.components[i - 1].charWidth
            deleteStart -= unorderedWidth
            deleteLen += unorderedWidth
            // Also check for indent before unordered
            if i >= 2, case .indent(let s) = parsed.components[i - 2] {
              deleteStart -= s.count
              deleteLen += s.count
            }
          } else if comp.isListMarker, i > 0, case .indent(let s) = parsed.components[i - 1] {
            deleteStart -= s.count
            deleteLen += s.count
          }
          let markerAbsRange = NSRange(
            location: lineRange.location + deleteStart,
            length: deleteLen)
          let newMarkdown = nsMarkdown.replacingCharacters(in: markerAbsRange, with: "")
          return EditorState(
            markdown: newMarkdown, selection: .cursor(markerAbsRange.location))
        }

        componentOffset = componentEnd
      }

      // Match bare blockquote: a `>` without trailing space (e.g. `>` or `> >`).
      // Deleting backward here removes the bare `>` (1 char) as a whole unit.
      if posInLine > 0 {
        let lineNS = lineText as NSString
        let prefixText = lineNS.substring(to: posInLine)
        if bareBlockquotePrefixPattern.firstMatch(
          in: prefixText, range: NSRange(location: 0, length: (prefixText as NSString).length)) != nil
        {
          let markerAbsRange = NSRange(
            location: lineRange.location + posInLine - 1,
            length: 1)
          let newMarkdown = nsMarkdown.replacingCharacters(in: markerAbsRange, with: "")
          return EditorState(
            markdown: newMarkdown, selection: .cursor(markerAbsRange.location))
        }
      }

      // Check for \n\n paragraph boundary: consume both newlines to merge paragraphs.
      // Without this, deleting one \n would leave a lone \n that is a CommonMark
      // soft line break rather than a paragraph separator.
      if clampedPos >= 2
        && nsMarkdown.character(at: clampedPos - 1) == UInt16(0x000A)
        && nsMarkdown.character(at: clampedPos - 2) == UInt16(0x000A)
      {
        let deleteRange = NSRange(location: clampedPos - 2, length: 2)
        let newMarkdown = nsMarkdown.replacingCharacters(in: deleteRange, with: "")
        return EditorState(markdown: newMarkdown, selection: .cursor(clampedPos - 2))
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

    // Use the prefix parser to find the innermost list marker at any nesting depth.
    let parsed = parsePrefix(lineText)
    let innermostListIdx = parsed.components.lastIndex(where: { $0.isListMarker })

    if innermostListIdx != nil {
      // On a list item line: add 4 spaces before the innermost list marker.
      // Compute the character offset of the innermost list marker (including
      // any preceding indent that forms a nesting unit with it).
      var offset = 0
      for (i, comp) in parsed.components.enumerated() {
        if i == innermostListIdx {
          // If preceded by indent, insert BEFORE the indent (increasing nesting)
          if i > 0, case .indent = parsed.components[i - 1] {
            offset -= parsed.components[i - 1].charWidth
          }
          break
        }
        offset += comp.charWidth
      }
      let indent = "    "
      let insertPos = lineRange.location + offset

      // Collect continuation line ranges that belong to this list item.
      let continuationRanges = findContinuationLineRanges(
        in: nsMarkdown, afterLineRange: lineRange, prefixWidth: parsed.prefixLength)

      // Apply insertions from back to front so ranges stay valid.
      var newMarkdown = nsMarkdown.replacingCharacters(
        in: NSRange(location: insertPos, length: 0), with: indent) as NSString
      let extraChars = 4  // characters added to the list item line
      for contRange in continuationRanges.reversed() {
        let adjustedLoc = contRange.location + extraChars
        newMarkdown = newMarkdown.replacingCharacters(
          in: NSRange(location: adjustedLoc, length: 0), with: indent) as NSString
      }

      let newPos = pos + 4
      return EditorState(markdown: newMarkdown as String, selection: .cursor(newPos))
    }

    // Check if we're on a continuation line belonging to a list item
    if findParentListItem(in: nsMarkdown, at: pos) != nil {
      let (bqPrefix, _) = extractBlockquotePrefix(lineText)
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

    // Use the prefix parser to find the innermost list marker at any nesting depth.
    let parsed = parsePrefix(lineText)
    let innermostListIdx = parsed.components.lastIndex(where: { $0.isListMarker })

    if innermostListIdx != nil {
      // Find the indent component before the innermost list marker.
      var indentOffset = 0
      var indentLength = 0
      for (i, comp) in parsed.components.enumerated() {
        if i == innermostListIdx {
          // Check preceding indent
          if i > 0, case .indent(let s) = parsed.components[i - 1] {
            indentOffset -= s.count
            indentLength = s.count
          }
          break
        }
        indentOffset += comp.charWidth
      }

      let spacesToRemove = min(indentLength, 4)
      guard spacesToRemove > 0 else { return state }

      // Collect continuation line ranges that belong to this list item.
      let continuationRanges = findContinuationLineRanges(
        in: nsMarkdown, afterLineRange: lineRange, prefixWidth: parsed.prefixLength)

      let removeStart = lineRange.location + indentOffset + (indentLength - spacesToRemove)
      let removeRange = NSRange(location: removeStart, length: spacesToRemove)

      // Remove spaces from continuation lines (back to front), then the list item line.
      var newMarkdown = nsMarkdown as String as NSString
      for contRange in continuationRanges.reversed() {
        let contLineText = newMarkdown.substring(with: contRange)
        let contLeading = contLineText.prefix(while: { $0 == " " || $0 == "\t" })
        let contRemove = min(contLeading.count, spacesToRemove)
        if contRemove > 0 {
          newMarkdown = newMarkdown.replacingCharacters(
            in: NSRange(location: contRange.location, length: contRemove), with: "") as NSString
        }
      }
      // Now remove from the list item line itself.
      newMarkdown = newMarkdown.replacingCharacters(in: removeRange, with: "") as NSString

      let newPos = max(removeStart, pos - spacesToRemove)
      return EditorState(markdown: newMarkdown as String, selection: .cursor(newPos))
    }

    // Check if we're on a continuation line belonging to a list item
    let (bqPrefix, strippedLine) = extractBlockquotePrefix(lineText)
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

  // MARK: - Continuation Line Discovery

  /// Find the ranges of continuation lines that belong to a list item.
  /// A continuation line follows the list item, is not itself a list marker,
  /// and has leading whitespace >= `prefixWidth` (the full prefix width of the
  /// parent list marker line).
  private static func findContinuationLineRanges(
    in nsMarkdown: NSString, afterLineRange lineRange: NSRange, prefixWidth: Int
  ) -> [NSRange] {
    var ranges: [NSRange] = []
    var searchPos = NSMaxRange(lineRange)
    while searchPos < nsMarkdown.length {
      let nextRange = nsMarkdown.lineRange(for: NSRange(location: searchPos, length: 0))
      let nextText = nsMarkdown.substring(with: nextRange)
      let nextNS = nextText as NSString

      // Stop if this line has its own list marker
      if anyListMarkerWidthPattern.firstMatch(
        in: nextText, range: NSRange(location: 0, length: nextNS.length)) != nil
      {
        break
      }

      // Stop if the line doesn't start with enough whitespace
      let leading = nextText.prefix(while: { $0 == " " || $0 == "\t" })
      if leading.count < prefixWidth {
        break
      }

      ranges.append(nextRange)
      searchPos = NSMaxRange(nextRange)
    }
    return ranges
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

    // Track running counters per scope key (the prefix string before the ordered marker).
    // Uses parsePrefix to handle deeply nested structures like `> >   > 1. Item`.
    var counters: [String: Int] = [:]
    var activeScopes: Set<String> = []

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

    var lastPrefixLength: [String: Int] = [:]

    for line in lines {
      let lineNS = line as NSString
      let parsed = parsePrefix(line)

      // Check if the prefix contains an ordered marker (at any nesting depth).
      let orderedIdx = parsed.components.lastIndex(where: {
        if case .ordered = $0 { return true }
        return false
      })

      if let orderedIdx = orderedIdx, case .ordered(let oldNumber) = parsed.components[orderedIdx] {
        // Build scope key from everything before the ordered marker, normalizing
        // list markers to equivalent-width spaces so that marker lines (e.g. `> - > `)
        // and their continuation lines (e.g. `>   > `) produce the same key.
        let scopeComponents = Array(parsed.components[..<orderedIdx])
        let scopeKey = buildFullContinuationPrefix(scopeComponents)
        let oldNumberStr = String(oldNumber)

        if !activeScopes.contains(scopeKey) {
          counters[scopeKey] = 1
        }

        let correctNumber = counters[scopeKey] ?? 1
        let correctNumberStr = String(correctNumber)
        counters[scopeKey] = correctNumber + 1

        activeScopes.insert(scopeKey)
        lastPrefixLength[scopeKey] = parsed.prefixLength

        if correctNumberStr != oldNumberStr {
          // Rebuild the line with the corrected number.
          var newComponents = parsed.components
          newComponents[orderedIdx] = .ordered(number: correctNumber)
          let newPrefix = rebuildPrefix(newComponents)
          let oldPrefix = rebuildPrefix(parsed.components)
          let newLine = "\(newPrefix)\(parsed.content)"
          newLines.append(newLine)

          let lengthDiff = (newPrefix as NSString).length - (oldPrefix as NSString).length
          let oldFullPrefixEnd = charOffset + (oldPrefix as NSString).length

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
        // Not an ordered list line. Check if it's a continuation of an active scope.
        let normalizedPrefix = buildFullContinuationPrefix(parsed.components)
        let isContinuation: Bool
        if !activeScopes.isEmpty, !parsed.content.isEmpty {
          // A line is a continuation if its structural prefix starts with the
          // scope key and its total indent (prefix + content whitespace) is at
          // least as wide as the scope's full prefix (including the marker).
          isContinuation = activeScopes.contains(where: { key in
            guard let prefixLen = lastPrefixLength[key] else { return false }
            guard normalizedPrefix.hasPrefix(key) else { return false }
            let contentIndent = parsed.content.prefix(while: { $0 == " " || $0 == "\t" }).count
            return parsed.prefixLength + contentIndent >= prefixLen
          })
        } else {
          isContinuation = false
        }

        if !isContinuation {
          // Clear scopes that share a common prefix with this line.
          let keysToRemove = activeScopes.filter { normalizedPrefix.hasPrefix($0) || $0.hasPrefix(normalizedPrefix) }
          activeScopes.subtract(keysToRemove)
          for key in keysToRemove { lastPrefixLength.removeValue(forKey: key) }
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

  // MARK: - Soft Line Break Normalization

  /// Normalize soft line breaks in markdown text by replacing them with spaces.
  ///
  /// In CommonMark, a soft line break (`\n` within a paragraph) renders as a space.
  /// This function parses the text, finds soft/hard line breaks within paragraphs,
  /// and replaces the `\n` plus any continuation prefix (`> `, indentation) with
  /// a single space. Also strips trailing whitespace before the break to avoid
  /// double spaces from hard-break syntax (`  \n`).
  ///
  /// Used to normalize pasted markdown so that hard-wrapped paragraphs reflow
  /// correctly in the WYSIWYG editor.
  static func normalizeSoftLineBreaks(in text: String) -> String {
    let doc = Document(parsing: text)
    let converter = SourceRangeConverter(string: text)

    // Collect gap ranges around SoftBreak/LineBreak nodes.
    // The gap is the span between the preceding sibling's end and the
    // following sibling's start, which includes the \n and any continuation
    // prefix (blockquote markers, list indentation).
    var gapRanges: [NSRange] = []

    func walkInlines(in node: Markup) {
      let children = Array(node.children)
      for (i, child) in children.enumerated() {
        if child is SoftBreak || child is LineBreak {
          // Find preceding and following sibling ranges to compute the gap.
          let prevEnd: Int?
          if i > 0, let r = children[i - 1].range {
            prevEnd = converter.utf16Offset(from: r.upperBound)
          } else {
            prevEnd = nil
          }
          let nextStart: Int?
          if i + 1 < children.count, let r = children[i + 1].range {
            nextStart = converter.utf16Offset(from: r.lowerBound)
          } else {
            nextStart = nil
          }

          if let start = prevEnd, let end = nextStart, end > start {
            // Also strip trailing whitespace before the \n to avoid double spaces.
            let nsText = text as NSString
            var trimmedStart = start
            while trimmedStart > 0
              && nsText.character(at: trimmedStart - 1) == UInt16(0x0020)
            {
              trimmedStart -= 1
            }
            // Only trim back if there's at least one space before the break,
            // preserving the character right before the whitespace.
            let effectiveStart = trimmedStart < start ? trimmedStart : start
            gapRanges.append(NSRange(location: effectiveStart, length: end - effectiveStart))
          }
        } else {
          walkInlines(in: child)
        }
      }
    }

    func walkBlocks(_ node: Markup) {
      for child in node.children {
        if child is Paragraph {
          walkInlines(in: child)
        } else {
          walkBlocks(child)
        }
      }
    }

    walkBlocks(doc)

    guard !gapRanges.isEmpty else { return text }

    // Apply replacements in reverse order so ranges stay valid.
    var result = text as NSString
    for gap in gapRanges.reversed() {
      result = result.replacingCharacters(in: gap, with: " ") as NSString
    }

    return result as String
  }
}
