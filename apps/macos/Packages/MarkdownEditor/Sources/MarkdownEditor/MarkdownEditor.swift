import AppKit
import STTextView
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
    let scrollView = STTextView.scrollableTextView()
    let textView = scrollView.documentView as! STTextView
    context.coordinator.configure(textView)
    textView.text = text
    context.coordinator.highlighter.highlight(textView: textView)
    return scrollView
  }

  public func updateNSView(_ scrollView: NSScrollView, context: Context) {
    guard let textView = scrollView.documentView as? STTextView else { return }
    guard !context.coordinator.isUpdating else { return }
    if textView.text != text {
      context.coordinator.isUpdating = true
      textView.text = text
      context.coordinator.highlighter.highlight(textView: textView)
      context.coordinator.isUpdating = false
    }
  }

  public func makeCoordinator() -> Coordinator {
    Coordinator(text: $text)
  }

  @MainActor
  public final class Coordinator: NSObject, @preconcurrency STTextViewDelegate {
    var text: Binding<String>
    let highlighter = SyntaxHighlighter()
    var isUpdating = false
    private weak var textView: STTextView?

    init(text: Binding<String>) {
      self.text = text
    }

    func configure(_ textView: STTextView) {
      self.textView = textView
      textView.textDelegate = self
      textView.font = highlighter.style.baseFont
      textView.isHorizontallyResizable = false
      textView.highlightSelectedLine = false
      textView.showsLineNumbers = false
    }

    // MARK: - STTextViewDelegate

    public func textViewDidChangeText(_ notification: Notification) {
      guard !isUpdating, let textView = notification.object as? STTextView else { return }
      isUpdating = true
      text.wrappedValue = textView.text ?? ""
      highlighter.highlight(textView: textView)
      isUpdating = false
    }

    public func textViewDidChangeSelection(_ notification: Notification) {
      guard let textView = notification.object as? STTextView else { return }
      highlighter.updateDelimiterVisibility(textView: textView)
    }
  }
}
