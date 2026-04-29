import AppKit
import ObjectiveC
import SwiftUI

/// An inline WYSIWYG markdown editor view.
///
/// Follows the Elm architecture:
/// 1. State (`EditorState`) drives all visuals
/// 2. User interactions are converted to `EditorEvent` values
/// 3. `EditorUpdate.update(state, event)` produces the next state
/// 4. `MarkdownRenderer.render(state)` produces a `RenderSpec`
/// 5. `RenderApplicator.apply(spec, textView)` updates the view
public struct MarkdownEditor: NSViewRepresentable {
  @Binding var state: EditorState
  @Environment(\.markdownStyle) private var style
  private let useTextKit2: Bool

  public init(state: Binding<EditorState>, useTextKit2: Bool = false) {
    self._state = state
    self.useTextKit2 = useTextKit2
  }

  public func makeNSView(context: Context) -> NSScrollView {
    let scrollView: NSScrollView
    let textView: NSTextView
    if useTextKit2 {
      // NSTextView(usingTextLayoutManager:) is the only reliable way to get a
      // TK2 stack with textStorage and textContentStorage properly linked
      // (Phase 2 testing). It returns NSTextView, not our subclass, so we
      // upgrade the class via the Objective-C runtime. Safe because
      // TextKit2MarkdownTextView adds no stored properties — only an
      // overridden characterIndexForInsertion(at:) — so memory layout matches.
      let tk2TextView = NSTextView(usingTextLayoutManager: true)
      object_setClass(tk2TextView, TextKit2MarkdownTextView.self)
      tk2TextView.minSize = NSSize(width: 0, height: 0)
      tk2TextView.maxSize = NSSize(
        width: CGFloat.greatestFiniteMagnitude,
        height: CGFloat.greatestFiniteMagnitude)
      tk2TextView.isVerticallyResizable = true
      tk2TextView.isHorizontallyResizable = false
      tk2TextView.textContentStorage?.delegate =
        context.coordinator.textKit2ContentStorageDelegate
      textView = tk2TextView

      scrollView = NSScrollView()
      scrollView.borderType = .noBorder
      scrollView.hasVerticalScroller = true
      scrollView.hasHorizontalScroller = false
      scrollView.autohidesScrollers = true
      scrollView.documentView = textView
    } else {
      scrollView = NSTextView.scrollableTextView()
      textView = scrollView.documentView as! NSTextView
    }
    context.coordinator.style = style
    context.coordinator.configure(textView)
    context.coordinator.syncToTextView(state, textView: textView)
    return scrollView
  }

  public func updateNSView(_ scrollView: NSScrollView, context: Context) {
    guard let textView = scrollView.documentView as? NSTextView else { return }
    let styleChanged = context.coordinator.style != style
    context.coordinator.style = style
    guard !context.coordinator.isProcessingEvent else { return }
    // External state change or style change
    if styleChanged || textView.string != state.markdown {
      context.coordinator.syncToTextView(state, textView: textView)
    }
  }

  public func makeCoordinator() -> Coordinator {
    Coordinator(state: $state, style: style, useTextKit2: useTextKit2)
  }

  @MainActor
  public final class Coordinator: NSObject, NSTextViewDelegate {
    var state: Binding<EditorState>
    var style: MarkdownStyle
    let useTextKit2: Bool
    var isProcessingEvent = false
    var lastSpec: RenderSpec?
    private let glyphDelegate = GlyphHidingLayoutManagerDelegate()
    /// Held by the Coordinator so it persists for the text view's lifetime
    /// (NSTextContentStorage holds its delegate weakly).
    let textKit2ContentStorageDelegate = TextKit2ContentStorageDelegate()

    init(state: Binding<EditorState>, style: MarkdownStyle = .default, useTextKit2: Bool = false) {
      self.state = state
      self.style = style
      self.useTextKit2 = useTextKit2
    }

    func configure(_ textView: NSTextView) {
      textView.delegate = self
      textView.font = MarkdownStyle.default.baseFont
      textView.isHorizontallyResizable = false
      textView.isAutomaticQuoteSubstitutionEnabled = false
      textView.isAutomaticDashSubstitutionEnabled = false
      textView.isAutomaticTextReplacementEnabled = false
      textView.isRichText = true
      textView.isGrammarCheckingEnabled = false
      textView.isContinuousSpellCheckingEnabled = false
      textView.allowsUndo = true
      textView.usesFindBar = true

      textView.autoresizingMask = [.width]
      textView.textContainer?.widthTracksTextView = true
      textView.textContainer?.containerSize = NSSize(
        width: 0, height: CGFloat.greatestFiniteMagnitude)

      // TK1 only: replace the default NSLayoutManager with our custom subclass
      // that draws full-width backgrounds for code blocks. The TK2 path uses
      // NSTextLayoutManager (will get custom NSTextLayoutFragment subclasses
      // in Phase 3) and doesn't expose an NSLayoutManager.
      if !useTextKit2,
        let textContainer = textView.textContainer,
        let textStorage = textView.textStorage,
        let oldLayoutManager = textView.layoutManager
      {
        textStorage.removeLayoutManager(oldLayoutManager)
        let codeBlockLM = CodeBlockBackgroundLayoutManager()
        codeBlockLM.delegate = glyphDelegate
        codeBlockLM.allowsNonContiguousLayout = true
        codeBlockLM.addTextContainer(textContainer)
        textStorage.addLayoutManager(codeBlockLM)
      }
    }

    /// Apply editor state to the text view (full sync).
    func syncToTextView(_ editorState: EditorState, textView: NSTextView) {
      isProcessingEvent = true
      defer { isProcessingEvent = false }

      let currentText = textView.string
      if currentText != editorState.markdown {
        if currentText.isEmpty {
          // Initial load — full replacement is fine (no scroll to preserve).
          textView.string = editorState.markdown
        } else if let diff = Self.computeDiff(old: currentText, new: editorState.markdown) {
          // External state change — apply surgically to preserve scroll & undo.
          if let textStorage = textView.textStorage {
            textStorage.beginEditing()
            textStorage.replaceCharacters(in: diff.range, with: diff.replacement)
            textStorage.endEditing()
          }
        }
      }
      textView.setSelectedRange(editorState.selection.nsRange)

      let spec = MarkdownRenderer.render(state: editorState, style: style)
      if useTextKit2 {
        TextKit2RenderApplicator.apply(spec, to: textView)
      } else {
        RenderApplicator.apply(spec, to: textView)
      }
      lastSpec = spec
    }

    /// Process an event through the Elm loop, applying text changes surgically.
    private func processEvent(_ event: EditorEvent, textView: NSTextView) {
      isProcessingEvent = true
      defer { isProcessingEvent = false }

      let oldMarkdown = state.wrappedValue.markdown
      let newState = EditorUpdate.update(state.wrappedValue, event: event)
      state.wrappedValue = newState

      // Apply text changes surgically instead of replacing the entire string.
      if let diff = Self.computeDiff(old: oldMarkdown, new: newState.markdown) {
        if let textStorage = textView.textStorage {
          textStorage.beginEditing()
          textStorage.replaceCharacters(in: diff.range, with: diff.replacement)
          textStorage.endEditing()
        }
      }

      textView.setSelectedRange(newState.selection.nsRange)

      let spec = MarkdownRenderer.render(state: newState, style: style)
      if useTextKit2 {
        TextKit2RenderApplicator.apply(spec, to: textView)
      } else {
        RenderApplicator.apply(spec, to: textView)
      }
      lastSpec = spec
    }

    // MARK: - Diff Helper

    /// Compute the minimal changed region between two strings.
    /// Returns `nil` if the strings are identical.
    private static func computeDiff(
      old: String, new: String
    ) -> (range: NSRange, replacement: String)? {
      let oldNS = old as NSString
      let newNS = new as NSString

      // Find common prefix
      let minLen = min(oldNS.length, newNS.length)
      var prefixLen = 0
      while prefixLen < minLen
        && oldNS.character(at: prefixLen) == newNS.character(at: prefixLen)
      {
        prefixLen += 1
      }

      // Find common suffix (not overlapping with prefix)
      var suffixLen = 0
      while suffixLen < minLen - prefixLen
        && oldNS.character(at: oldNS.length - 1 - suffixLen)
          == newNS.character(at: newNS.length - 1 - suffixLen)
      {
        suffixLen += 1
      }

      let oldChangedLen = oldNS.length - prefixLen - suffixLen
      let newChangedLen = newNS.length - prefixLen - suffixLen

      if oldChangedLen == 0 && newChangedLen == 0 {
        return nil  // No change
      }

      let range = NSRange(location: prefixLen, length: oldChangedLen)
      let replacement = newNS.substring(
        with: NSRange(location: prefixLen, length: newChangedLen))
      return (range, replacement)
    }

    // MARK: - NSTextViewDelegate

    public func textView(
      _ textView: NSTextView, doCommandBy commandSelector: Selector
    ) -> Bool {
      if commandSelector == #selector(NSResponder.insertNewline(_:)) {
        if NSApp.currentEvent?.modifierFlags.contains(.shift) == true {
          processEvent(.insertLineBreak, textView: textView)
        } else {
          processEvent(.insertNewline, textView: textView)
        }
        return true
      }
      if commandSelector == #selector(NSResponder.deleteBackward(_:)) {
        processEvent(.deleteBackward, textView: textView)
        return true
      }
      if commandSelector == #selector(NSResponder.deleteForward(_:)) {
        processEvent(.deleteForward, textView: textView)
        return true
      }
      if commandSelector == #selector(NSResponder.insertTab(_:)) {
        processEvent(.indent, textView: textView)
        return true
      }
      if commandSelector == #selector(NSResponder.insertBacktab(_:)) {
        processEvent(.outdent, textView: textView)
        return true
      }
      if commandSelector == #selector(NSResponder.deleteToBeginningOfLine(_:))
        || commandSelector == #selector(NSResponder.deleteToBeginningOfParagraph(_:))
      {
        processEvent(.deleteToBeginningOfLine, textView: textView)
        return true
      }
      if commandSelector == #selector(NSResponder.deleteToEndOfLine(_:))
        || commandSelector == #selector(NSResponder.deleteToEndOfParagraph(_:))
      {
        processEvent(.deleteToEndOfLine, textView: textView)
        return true
      }
      if commandSelector == #selector(NSResponder.deleteWordBackward(_:)) {
        processEvent(.deleteWordBackward, textView: textView)
        return true
      }
      if commandSelector == #selector(NSResponder.deleteWordForward(_:)) {
        processEvent(.deleteWordForward, textView: textView)
        return true
      }
      return false
    }

    public func textView(
      _ textView: NSTextView, shouldChangeTextIn affectedCharRange: NSRange,
      replacementString: String?
    ) -> Bool {
      guard !isProcessingEvent else { return true }

      // Intercept text insertion when we need to inject a blockquote space.
      // Normal typing bypasses EditorUpdate.handleInsertText (NSTextView handles
      // it natively), so space injection after bare `>` would not fire. Route
      // these cases through the Elm loop instead.
      if let text = replacementString, !text.isEmpty,
        !text.hasPrefix(" "), !text.hasPrefix("\n")
      {
        let insertPos = affectedCharRange.location
        if insertPos > 0 {
          let md = state.wrappedValue.markdown as NSString
          let lineRange = md.lineRange(for: NSRange(location: insertPos, length: 0))
          let lineText = md.substring(with: lineRange)
          let posInLine = insertPos - lineRange.location
          if EditorUpdate.needsBlockquoteSpaceInjection(lineText: lineText, posInLine: posInLine) {
            // Route through Elm loop so handleInsertText injects the space.
            processEvent(.insertText(text), textView: textView)
            return false
          }
        }
      }

      // Normalize soft line breaks in pasted/imported text.
      // Return key is handled by doCommandBy (which returns true, preventing
      // this method from being called), so \n here comes from paste, drag-drop,
      // or other multi-character input — not from user pressing Return.
      if let text = replacementString, text.contains("\n"), text.count > 1 {
        let normalized = EditorUpdate.normalizeSoftLineBreaks(in: text)
        if normalized != text {
          processEvent(.paste(normalized), textView: textView)
          return false
        }
      }

      // Allow NSTextView to handle the edit natively.
      // textDidChange will sync state and re-render attributes.
      return true
    }

    public func textDidChange(_ notification: Notification) {
      guard !isProcessingEvent, let textView = notification.object as? NSTextView else { return }

      isProcessingEvent = true
      defer { isProcessingEvent = false }

      // Read current state from text view (NSTextView already applied the edit).
      let nsRange = textView.selectedRange()
      let selection: Selection =
        nsRange.length == 0
        ? .cursor(nsRange.location)
        : .range(anchor: nsRange.location, head: nsRange.location + nsRange.length)

      let currentMarkdown = textView.string
      var newState = EditorState(markdown: currentMarkdown, selection: selection)

      // Run post-processing (renumbering, setext normalization).
      newState = EditorUpdate.postProcess(newState)

      // If post-processing changed the markdown, apply surgically.
      if newState.markdown != currentMarkdown {
        if let diff = Self.computeDiff(old: currentMarkdown, new: newState.markdown) {
          if let textStorage = textView.textStorage {
            textStorage.beginEditing()
            textStorage.replaceCharacters(in: diff.range, with: diff.replacement)
            textStorage.endEditing()
          }
        }
        textView.setSelectedRange(newState.selection.nsRange)
      }

      state.wrappedValue = newState

      // Re-render attributes.
      let spec = MarkdownRenderer.render(state: newState, style: style)
      if useTextKit2 {
        TextKit2RenderApplicator.apply(spec, to: textView)
      } else {
        RenderApplicator.apply(spec, to: textView)
      }
      lastSpec = spec
    }

    public func textViewDidChangeSelection(_ notification: Notification) {
      guard !isProcessingEvent, let textView = notification.object as? NSTextView else { return }

      // During cut/paste/typing-over-selection, NSTextView modifies the text
      // storage before posting the selection-change notification. If our state
      // still holds the pre-mutation markdown, rendering a spec from it would
      // produce out-of-bounds ranges that corrupt the layout manager. Skip the
      // cursor update here — textDidChange will handle the full resync.
      guard textView.string == state.wrappedValue.markdown else { return }

      let nsRange = textView.selectedRange()
      let selection: Selection
      if nsRange.length == 0 {
        selection = .cursor(nsRange.location)
      } else {
        selection = .range(anchor: nsRange.location, head: nsRange.location + nsRange.length)
      }

      // Only update rendering (glyph visibility), don't change text
      isProcessingEvent = true
      state.wrappedValue = EditorState(
        markdown: state.wrappedValue.markdown, selection: selection)

      let spec = MarkdownRenderer.render(state: state.wrappedValue, style: style)
      let prevHidden = lastSpec?.hiddenIndexes ?? IndexSet()
      let prevBullets = lastSpec?.bulletIndexes ?? IndexSet()
      let prevUncheckedCheckboxes = lastSpec?.uncheckedCheckboxIndexes ?? IndexSet()
      let prevCheckedCheckboxes = lastSpec?.checkedCheckboxIndexes ?? IndexSet()
      let prevCollapsedNewlines = lastSpec?.collapsedNewlineIndexes ?? IndexSet()
      if useTextKit2 {
        TextKit2RenderApplicator.applyCursorUpdate(
          spec, previousHidden: prevHidden, previousBullets: prevBullets,
          previousUncheckedCheckboxes: prevUncheckedCheckboxes,
          previousCheckedCheckboxes: prevCheckedCheckboxes,
          previousCollapsedNewlines: prevCollapsedNewlines, to: textView)
      } else {
        RenderApplicator.applyCursorUpdate(
          spec, previousHidden: prevHidden, previousBullets: prevBullets,
          previousUncheckedCheckboxes: prevUncheckedCheckboxes,
          previousCheckedCheckboxes: prevCheckedCheckboxes,
          previousCollapsedNewlines: prevCollapsedNewlines, to: textView)
      }
      lastSpec = spec
      isProcessingEvent = false
    }
  }
}
