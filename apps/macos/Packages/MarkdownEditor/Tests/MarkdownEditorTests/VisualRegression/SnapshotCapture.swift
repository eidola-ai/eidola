import AppKit

@testable import MarkdownEditor

/// Captures bitmap snapshots of markdown rendered into an NSTextView.
///
/// Uses `NSTextView.cacheDisplay(in:to:)` to render the view into a bitmap,
/// bypassing the need for an on-screen window. This naturally excludes the
/// cursor blink — the insertion point is drawn separately by NSTextView.
@MainActor
enum SnapshotCapture {

  /// Create a fresh NSTextView, apply the given state, and capture a bitmap.
  static func capture(
    text: String,
    cursorPosition: Int,
    size: NSSize = NSSize(width: 600, height: 400),
    style: MarkdownStyle = .default
  ) -> NSBitmapImageRep {
    let components = MarkdownTextViewFactory.create(size: size)
    apply(text: text, cursorPosition: cursorPosition, style: style, to: components)
    return renderBitmap(from: components, size: size)
  }

  /// Apply text + cursor to existing components (for "mutated" path testing).
  static func apply(
    text: String,
    cursorPosition: Int,
    style: MarkdownStyle = .default,
    to components: MarkdownTextViewFactory.Components
  ) {
    let textView = components.textView
    textView.string = text
    let cursorRange = NSRange(
      location: min(cursorPosition, (text as NSString).length), length: 0)
    textView.setSelectedRange(cursorRange)

    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange, style: style)
    TextKit2RenderApplicator.apply(spec, to: textView)
  }

  /// Render the current state of components to a bitmap.
  static func renderBitmap(
    from components: MarkdownTextViewFactory.Components,
    size: NSSize = NSSize(width: 600, height: 400)
  ) -> NSBitmapImageRep {
    let textView = components.textView

    // Force a full TK2 re-layout. We invalidate first because layout
    // fragments are cached per text element; without explicit invalidation
    // they retain Y positions from a prior `apply` whose hidden-prefix /
    // bullet / checkbox state differed (the content delegate's display
    // string updates, but the cached layout-fragment frame doesn't).
    if let tlm = textView.textLayoutManager {
      tlm.invalidateLayout(for: tlm.documentRange)
      tlm.ensureLayout(for: tlm.documentRange)
      // Re-run the viewport layout controller so any visible fragments are
      // re-positioned against the just-recomputed content, and the text
      // view's internal frame matches the laid-out content height.
      tlm.textViewportLayoutController.layoutViewport()
    }

    // Reset the textView's bounds origin to (0, 0) so cacheDisplay captures
    // from the document top. Without a scroll view, the harness has no clip
    // view to anchor scroll position; previous TK2 viewport layouts (or the
    // text view's auto-grow when content exceeds the requested bitmap
    // height) can leave bounds origin non-zero.
    let captureRect = NSRect(origin: .zero, size: size)
    textView.setBoundsOrigin(.zero)

    // Deselect to avoid cursor rendering artifacts
    let savedSelection = textView.selectedRange()
    textView.setSelectedRange(NSRange(location: 0, length: 0))

    // Force display update
    textView.needsDisplay = true
    textView.displayIfNeeded()

    let bitmapRep = NSBitmapImageRep(
      bitmapDataPlanes: nil,
      pixelsWide: Int(size.width),
      pixelsHigh: Int(size.height),
      bitsPerSample: 8,
      samplesPerPixel: 4,
      hasAlpha: true,
      isPlanar: false,
      colorSpaceName: .calibratedRGB,
      bytesPerRow: 0,
      bitsPerPixel: 0
    )!

    textView.cacheDisplay(in: captureRect, to: bitmapRep)

    // Restore selection
    textView.setSelectedRange(savedSelection)

    return bitmapRep
  }

  /// Save a bitmap to disk for debugging. Returns the file path.
  @discardableResult
  static func saveToDisk(
    _ bitmap: NSBitmapImageRep, name: String, directory: String = "/tmp/markdown-visual-tests"
  ) -> String {
    let fm = FileManager.default
    try? fm.createDirectory(atPath: directory, withIntermediateDirectories: true)
    let path = "\(directory)/\(name).png"
    if let data = bitmap.representation(using: .png, properties: [:]) {
      try? data.write(to: URL(fileURLWithPath: path))
    }
    return path
  }
}
