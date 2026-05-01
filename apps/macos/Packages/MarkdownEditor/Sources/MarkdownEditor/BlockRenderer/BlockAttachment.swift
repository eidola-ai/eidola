import AppKit

/// `NSTextAttachment` subclass inserted by `TextKit2ContentStorageDelegate`
/// for the first paragraph of every block-renderer spec range.
///
/// The attachment carries a back-reference to its `BlockRenderHost`. AppKit
/// asks the attachment for a view provider via
/// `viewProvider(for:location:textContainer:)`; we lazy-init the provider
/// inside that hook (per Spike S1.1's finding that constructing it in
/// `init` and KVC-rebinding `textAttachment` later crashes the next layout
/// pass) and cache it for subsequent calls so view identity stays bound to
/// the host.
///
/// The attachment's `attachmentBounds` is sourced from the host's spec
/// `reservedHeight` and the line fragment's width, so AppKit can lay out
/// the surrounding text without consulting the live renderer.
///
/// Not annotated `@MainActor` because the AppKit overrides
/// (`viewProvider(for:...)`, `attachmentBounds(for:...)`) are inherited as
/// nonisolated. In practice all access is on the main thread — same
/// constraint pattern the S1.1 spike validated.
public final class BlockAttachment: NSTextAttachment {

  /// Back-reference to the host that owns this attachment's view. Set by
  /// `TextKit2ContentStorageDelegate` at attachment-paragraph build time.
  /// `nonisolated(unsafe)` because the attachment overrides several
  /// non-isolated `NSTextAttachment` hooks; in practice all access is on
  /// the main thread.
  nonisolated(unsafe) public weak var host: BlockRenderHost?

  /// Cached view provider. Built on first `viewProvider(for:...)` call so
  /// the provider can be constructed with the real `parentView` /
  /// `location` / `textLayoutManager` AppKit hands us — and reused on
  /// every subsequent call so the same provider returns the same view
  /// instance (matching the S1.1 spike's working shape).
  nonisolated(unsafe) private var cachedProvider: BlockAttachmentViewProvider?

  @MainActor
  public init(host: BlockRenderHost) {
    self.host = host
    super.init(data: nil, ofType: nil)
    // Bounds come from the host's reserved height; width is set when
    // AppKit asks via `attachmentBounds(for:...)`.
    self.bounds = NSRect(x: 0, y: 0, width: 0, height: host.spec.reservedHeight)
    // Suppress the default `NSTextAttachmentCell` (which paints the generic
    // document/file icon for any attachment that has no image, fileWrapper,
    // or registered provider for its fileType). Setting `attachmentCell`
    // to nil does NOT suppress it — that just makes AppKit auto-build a
    // fresh default cell on next use. The reliable trick is to install a
    // transparent 1×1 image: AppKit then constructs an
    // `NSImageAttachmentCell` over it that draws effectively nothing
    // during the transient layout states where the view provider's view
    // isn't yet mounted (which the production apply pipeline triggers on
    // every keystroke and selection change).
    self.image = Self.transparentPlaceholder
  }

  @MainActor
  private static let transparentPlaceholder: NSImage = {
    let image = NSImage(size: NSSize(width: 1, height: 1))
    image.lockFocus()
    NSColor.clear.setFill()
    NSRect(x: 0, y: 0, width: 1, height: 1).fill()
    image.unlockFocus()
    return image
  }()

  required init?(coder: NSCoder) {
    super.init(coder: coder)
  }

  override public func viewProvider(
    for parentView: NSView?,
    location: any NSTextLocation,
    textContainer: NSTextContainer?
  ) -> NSTextAttachmentViewProvider? {
    if let cachedProvider { return cachedProvider }
    let provider = BlockAttachmentViewProvider(
      textAttachment: self,
      parentView: parentView,
      textLayoutManager: textContainer?.textLayoutManager,
      location: location)
    cachedProvider = provider
    return provider
  }

  override public func attachmentBounds(
    for textContainer: NSTextContainer?,
    proposedLineFragment lineFrag: CGRect,
    glyphPosition position: CGPoint,
    characterIndex charIndex: Int
  ) -> CGRect {
    let width = textContainer?.size.width ?? lineFrag.width
    // Read the reserved height live from the host on every call — the host's
    // spec is updated by `BlockRendererRegistry.reconcile` whenever the
    // markdown source grows or shrinks (e.g. typing more lines into a code
    // block extends the reserved region). Caching the value at attachment
    // construction time froze the bounds at the first-seen height and made
    // incremental renders disagree with fresh renders for the same content.
    let weakHost = host
    let height = MainActor.assumeIsolated { weakHost?.spec.reservedHeight ?? 0 }
    return CGRect(x: 0, y: 0, width: width, height: height)
  }
}

/// View provider returned by `BlockAttachment.viewProvider(for:...)`.
///
/// `loadView()` defers to the host (via `host.ensureView()`) so the same
/// `NSView` instance is returned across the inevitable AppKit
/// unmount/re-mount cycles that accompany text changes — preserving any
/// embedded state (scroll position, selection, focus) the renderer holds.
public final class BlockAttachmentViewProvider: NSTextAttachmentViewProvider {

  /// Cached view returned across `loadView()` calls. Identity is tied to
  /// the host (the attachment's back-reference) so AppKit's unmount/re-
  /// mount cycles preserve embedded state. Same shape as Spike S1.1's
  /// `CachingProvider`.
  nonisolated(unsafe) private var cachedView: NSView?

  override public func loadView() {
    // Tell AppKit the view IS the attachment — its bounds come from the
    // attachment's `attachmentBounds(...)`, and AppKit should not also
    // draw the underlying attachment cell. Without this, AppKit may fall
    // back to drawing the default file-icon placeholder over our view
    // during transient layout states.
    tracksTextAttachmentViewBounds = true
    if let cachedView {
      view = cachedView
      return
    }
    guard let attachment = textAttachment as? BlockAttachment,
      let host = attachment.host
    else {
      // Defensive: vend a bare placeholder if the host went away. Should
      // not happen in practice; the attachment is replaced before its
      // host is disposed.
      let placeholder = MainActor.assumeIsolated { NSView(frame: .zero) }
      cachedView = placeholder
      view = placeholder
      return
    }
    let v = host.ensureView()
    cachedView = v
    view = v
  }
}
