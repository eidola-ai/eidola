import AppKit

@testable import MarkdownEditor

/// Captures bitmap snapshots of markdown rendered into an NSTextView.
///
/// Uses `NSLayoutManager.drawGlyphs/drawBackground` directly into a bitmap context,
/// bypassing the need for a window. This also naturally excludes the cursor blink,
/// since the insertion point is drawn by NSTextView, not NSLayoutManager.
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
    RenderApplicator.apply(spec, to: textView)
  }

  /// Render the current state of components to a bitmap.
  static func renderBitmap(
    from components: MarkdownTextViewFactory.Components,
    size: NSSize = NSSize(width: 600, height: 400)
  ) -> NSBitmapImageRep {
    let textView = components.textView
    let layoutManager = components.layoutManager
    let textLength = components.textStorage.length

    // Force synchronous layout
    if textLength > 0 {
      layoutManager.ensureLayout(forCharacterRange: NSRange(location: 0, length: textLength))
    }

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

    textView.cacheDisplay(in: textView.bounds, to: bitmapRep)

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
