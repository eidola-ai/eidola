import AppKit

/// Phase 2.1's placeholder renderer. Paints a flat colored rectangle in
/// the attachment's reserved region — proving the bridging layer wires
/// up end-to-end without depending on any real renderer.
///
/// Phase 2.2 swaps this out for the real `CodeBlockRenderer` (an embedded
/// `NSScrollView { NSTextView }` for horizontal scroll over long lines).
@MainActor
final class NoopBlockRenderer: BlockRenderer {

  private final class PlaceholderView: NSView {
    init() {
      super.init(frame: .zero)
      wantsLayer = true
      layer?.backgroundColor = NSColor.systemTeal.withAlphaComponent(0.3).cgColor
    }
    required init?(coder: NSCoder) { fatalError("not used") }
  }

  func makeView(host: BlockRenderHost) -> NSView {
    PlaceholderView()
  }

  func update(spec: BlockRendererSpec, host: BlockRenderHost) {
    // Nothing to reflow — the colored rectangle is content-independent.
  }

  func cursorPresenceChanged(_ inside: Bool, host: BlockRenderHost) {
    // No chrome flip in the placeholder.
  }

  func desiredBounds(host: BlockRenderHost) -> CGRect {
    CGRect(x: 0, y: 0, width: 0, height: host.spec.reservedHeight)
  }

  func tearDown() {
    // Nothing to release; the view goes away with the host.
  }
}
