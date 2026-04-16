import AppKit

@testable import MarkdownEditor

/// Creates an NSTextView configured identically to the real MarkdownEditor,
/// for use in visual regression tests.
@MainActor
enum MarkdownTextViewFactory {

  struct Components {
    let textView: NSTextView
    let glyphDelegate: GlyphHidingLayoutManagerDelegate
    let layoutManager: NSLayoutManager
    let textContainer: NSTextContainer
    let textStorage: NSTextStorage
  }

  static func create(size: NSSize = NSSize(width: 600, height: 400)) -> Components {
    let textStorage = NSTextStorage()
    let layoutManager = NSLayoutManager()
    let glyphDelegate = GlyphHidingLayoutManagerDelegate()
    layoutManager.delegate = glyphDelegate
    layoutManager.allowsNonContiguousLayout = true

    let textContainer = NSTextContainer(
      containerSize: NSSize(width: size.width, height: CGFloat.greatestFiniteMagnitude))
    textContainer.widthTracksTextView = false
    layoutManager.addTextContainer(textContainer)
    textStorage.addLayoutManager(layoutManager)

    let textView = NSTextView(
      frame: NSRect(origin: .zero, size: size), textContainer: textContainer)

    // Replicate Coordinator.configure exactly:
    textView.font = MarkdownStyle.default.baseFont
    textView.isHorizontallyResizable = false
    textView.isAutomaticQuoteSubstitutionEnabled = false
    textView.isAutomaticDashSubstitutionEnabled = false
    textView.isAutomaticTextReplacementEnabled = false
    textView.isRichText = true
    textView.isGrammarCheckingEnabled = false
    textView.isContinuousSpellCheckingEnabled = false

    // NSTextView needs to be in a window for layout/drawing to work.
    // defer: true avoids creating a real window server surface.
    let window = NSWindow(
      contentRect: NSRect(origin: .zero, size: size),
      styleMask: .borderless, backing: .buffered, defer: true)
    window.contentView = textView

    return Components(
      textView: textView,
      glyphDelegate: glyphDelegate,
      layoutManager: layoutManager,
      textContainer: textContainer,
      textStorage: textStorage
    )
  }
}
