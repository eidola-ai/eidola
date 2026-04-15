import AppKit
import SwiftUI

/// An inline WYSIWYG markdown editor view.
///
/// The editor renders markdown formatting in-place. When the cursor enters a formatted
/// region, raw syntax markers become visible for editing. When the cursor leaves, markers
/// hide and the text appears fully formatted.
///
/// The underlying text buffer is always valid markdown.
public struct MarkdownEditor: NSViewRepresentable {
  @Binding var text: String

  public init(text: Binding<String>) {
    self._text = text
  }

  public func makeNSView(context: Context) -> NSScrollView {
    let scrollView = NSTextView.scrollableTextView()
    let textView = scrollView.documentView as! NSTextView
    context.coordinator.configure(textView)
    textView.string = text
    context.coordinator.highlighter.highlight(textView: textView)
    return scrollView
  }

  public func updateNSView(_ scrollView: NSScrollView, context: Context) {
    guard let textView = scrollView.documentView as? NSTextView else { return }
    guard !context.coordinator.isUpdating else { return }
    if textView.string != text {
      context.coordinator.isUpdating = true
      textView.string = text
      context.coordinator.highlighter.highlight(textView: textView)
      context.coordinator.isUpdating = false
    }
  }

  public func makeCoordinator() -> Coordinator {
    Coordinator(text: $text)
  }

  @MainActor
  public final class Coordinator: NSObject, NSTextViewDelegate {
    var text: Binding<String>
    let highlighter = SyntaxHighlighter()
    var isUpdating = false
    private weak var textView: NSTextView?

    init(text: Binding<String>) {
      self.text = text
    }

    func configure(_ textView: NSTextView) {
      self.textView = textView
      textView.delegate = self
      textView.font = highlighter.style.baseFont
      textView.isHorizontallyResizable = false
      textView.isAutomaticQuoteSubstitutionEnabled = false
      textView.isAutomaticDashSubstitutionEnabled = false
      textView.isAutomaticTextReplacementEnabled = false
      textView.isRichText = false
      textView.allowsUndo = true
      textView.usesFindBar = true

      // Make text view fill the scroll view width
      textView.autoresizingMask = [.width]
      textView.textContainer?.widthTracksTextView = true
      textView.textContainer?.containerSize = NSSize(
        width: 0, height: CGFloat.greatestFiniteMagnitude)
    }

    // MARK: - NSTextViewDelegate

    public func textDidChange(_ notification: Notification) {
      guard !isUpdating, let textView = notification.object as? NSTextView else { return }
      isUpdating = true
      text.wrappedValue = textView.string
      highlighter.highlight(textView: textView)
      isUpdating = false
    }

    public func textViewDidChangeSelection(_ notification: Notification) {
      guard let textView = notification.object as? NSTextView else { return }
      highlighter.updateDelimiterVisibility(textView: textView)
    }
  }
}
