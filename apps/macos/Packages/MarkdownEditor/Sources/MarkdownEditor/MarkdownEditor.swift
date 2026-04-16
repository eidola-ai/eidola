import AppKit
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

  public init(state: Binding<EditorState>) {
    self._state = state
  }

  public func makeNSView(context: Context) -> NSScrollView {
    let scrollView = NSTextView.scrollableTextView()
    let textView = scrollView.documentView as! NSTextView
    context.coordinator.configure(textView)
    context.coordinator.syncToTextView(state, textView: textView)
    return scrollView
  }

  public func updateNSView(_ scrollView: NSScrollView, context: Context) {
    guard let textView = scrollView.documentView as? NSTextView else { return }
    guard !context.coordinator.isProcessingEvent else { return }
    // External state change (e.g., undo, programmatic update)
    context.coordinator.syncToTextView(state, textView: textView)
  }

  public func makeCoordinator() -> Coordinator {
    Coordinator(state: $state)
  }

  @MainActor
  public final class Coordinator: NSObject, NSTextViewDelegate {
    var state: Binding<EditorState>
    var isProcessingEvent = false
    var lastSpec: RenderSpec?
    private let glyphDelegate = GlyphHidingLayoutManagerDelegate()

    init(state: Binding<EditorState>) {
      self.state = state
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

      textView.layoutManager?.delegate = glyphDelegate
      textView.layoutManager?.allowsNonContiguousLayout = true
    }

    /// Apply editor state to the text view (full sync).
    func syncToTextView(_ editorState: EditorState, textView: NSTextView) {
      isProcessingEvent = true
      defer { isProcessingEvent = false }

      if textView.string != editorState.markdown {
        textView.string = editorState.markdown
      }
      textView.setSelectedRange(editorState.selection.nsRange)

      let spec = MarkdownRenderer.render(state: editorState)
      RenderApplicator.apply(spec, to: textView)
      lastSpec = spec
    }

    /// Process an event through the Elm loop.
    private func processEvent(_ event: EditorEvent, textView: NSTextView) {
      isProcessingEvent = true
      defer { isProcessingEvent = false }

      let newState = EditorUpdate.update(state.wrappedValue, event: event)
      state.wrappedValue = newState

      textView.string = newState.markdown
      textView.setSelectedRange(newState.selection.nsRange)

      let spec = MarkdownRenderer.render(state: newState)
      RenderApplicator.apply(spec, to: textView)
      lastSpec = spec
    }

    // MARK: - NSTextViewDelegate

    public func textView(
      _ textView: NSTextView, doCommandBy commandSelector: Selector
    ) -> Bool {
      if commandSelector == #selector(NSResponder.insertNewline(_:)) {
        processEvent(.insertNewline, textView: textView)
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
      return false
    }

    public func textView(
      _ textView: NSTextView, shouldChangeTextIn affectedCharRange: NSRange,
      replacementString: String?
    ) -> Bool {
      guard let replacement = replacementString else { return true }
      // Let doCommandBy handle newline and delete
      if replacement == "\n" || (replacement.isEmpty && affectedCharRange.length > 0) {
        return true
      }
      processEvent(.insertText(replacement), textView: textView)
      return false  // We handled it
    }

    public func textViewDidChangeSelection(_ notification: Notification) {
      guard !isProcessingEvent, let textView = notification.object as? NSTextView else { return }
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

      let spec = MarkdownRenderer.render(state: state.wrappedValue)
      let prevHidden = lastSpec?.hiddenIndexes ?? IndexSet()
      let prevBullets = lastSpec?.bulletIndexes ?? IndexSet()
      RenderApplicator.applyCursorUpdate(
        spec, previousHidden: prevHidden, previousBullets: prevBullets, to: textView)
      lastSpec = spec
      isProcessingEvent = false
    }
  }
}
