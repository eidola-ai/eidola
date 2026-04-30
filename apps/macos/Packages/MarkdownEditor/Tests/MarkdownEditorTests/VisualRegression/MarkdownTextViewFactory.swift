import AppKit
import ObjectiveC

@testable import MarkdownEditor

/// Creates an NSTextView configured identically to the real MarkdownEditor
/// (TextKit 2 stack), for use in visual regression tests.
@MainActor
enum MarkdownTextViewFactory {

  struct Components {
    let textView: NSTextView
    /// Held strongly so it outlives the call (NSTextContentStorage stores
    /// its delegate weakly, mirroring the production Coordinator).
    let contentStorageDelegate: TextKit2ContentStorageDelegate
    /// Held strongly so it outlives the call (NSTextLayoutManager stores
    /// its delegate weakly, same as content storage).
    let layoutManagerDelegate: TextKit2LayoutManagerDelegate
    let window: NSWindow
  }

  static func create(size: NSSize = NSSize(width: 600, height: 400)) -> Components {
    // Spike 1 found that NSTextView(usingTextLayoutManager: true) is the only
    // reliable way to set up a TK2 stack — manual NSTextContentStorage wiring
    // doesn't link textStorage to contentStorage correctly. The production
    // code path (in MarkdownEditor.makeNSView) uses the same approach.
    let textView = NSTextView(usingTextLayoutManager: true)
    object_setClass(textView, TextKit2MarkdownTextView.self)
    textView.frame = NSRect(origin: .zero, size: size)
    textView.minSize = size
    textView.maxSize = size
    // Pin the text view to the requested bitmap size — the harness has no
    // scroll view, so we don't want the text view to auto-grow with content
    // (which would shift the bounds origin and confuse `cacheDisplay`).
    textView.isVerticallyResizable = false

    // Replicate Coordinator.configure exactly:
    textView.font = MarkdownStyle.default.baseFont
    textView.isHorizontallyResizable = false
    textView.isAutomaticQuoteSubstitutionEnabled = false
    textView.isAutomaticDashSubstitutionEnabled = false
    textView.isAutomaticTextReplacementEnabled = false
    textView.isRichText = true
    textView.isGrammarCheckingEnabled = false
    textView.isContinuousSpellCheckingEnabled = false
    textView.textContainer?.containerSize = NSSize(
      width: size.width, height: CGFloat.greatestFiniteMagnitude)
    textView.textContainer?.widthTracksTextView = false

    let contentStorageDelegate = TextKit2ContentStorageDelegate()
    textView.textContentStorage?.delegate = contentStorageDelegate

    let layoutManagerDelegate = TextKit2LayoutManagerDelegate()
    textView.textLayoutManager?.delegate = layoutManagerDelegate

    // NSTextView needs to be in a window for layout/drawing to work.
    // defer: true avoids creating a real window server surface.
    let window = NSWindow(
      contentRect: NSRect(origin: .zero, size: size),
      styleMask: .borderless, backing: .buffered, defer: true)
    window.contentView = textView

    return Components(
      textView: textView,
      contentStorageDelegate: contentStorageDelegate,
      layoutManagerDelegate: layoutManagerDelegate,
      window: window
    )
  }
}
