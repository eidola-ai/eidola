import AppKit

/// Per-attachment controller for a custom-view block renderer. Owned by the
/// `BlockRendererRegistry` and held alive for the lifetime of the host's
/// spec entry.
///
/// The host bridges three coordinate systems:
/// - The main `NSTextView`'s source storage (single source of truth for
///   markdown content).
/// - The renderer's view (a custom AppKit view tree, e.g. an embedded
///   `NSScrollView { NSTextView }` for code blocks).
/// - The cursor / selection state (whose primary identity is its source
///   offset, not its visual position).
///
/// The renderer only ever reads source through the host (`sourceText()`),
/// asks the host whether the cursor is inside its range
/// (`isCursorInside`), and forwards selection updates / key events back
/// to the main responder chain via the helpers below. It never touches
/// the main `NSTextStorage` directly.
@MainActor
public final class BlockRenderHost {

  /// The tag the renderer was registered under. Convenient for renderers
  /// that share a class but want to branch on tag for minor variants.
  public let blockTypeTag: BlockTypeTag

  /// The current spec. Updated by the registry when the applicator hands a
  /// new spec for the same range. The renderer is notified through
  /// `BlockRenderer.update(spec:host:)`.
  public private(set) var spec: BlockRendererSpec

  /// Weak link back to the main `NSTextView`. Goes nil when the text view
  /// is deallocated (the registry retires hosts before then in normal
  /// teardown but defensive nilness keeps unit tests honest).
  public private(set) weak var textView: NSTextView?

  /// The renderer instance bound to this host. Built lazily on first
  /// `view` access so a host can be created and reconciled before AppKit
  /// asks for its view.
  public private(set) var renderer: BlockRenderer?

  /// The view AppKit ultimately mounts. Built once via the renderer's
  /// `makeView(host:)`. Cached here so subsequent `viewProvider.loadView()`
  /// calls return the same instance; AppKit may unmount and re-mount the
  /// view when the surrounding paragraph changes, but identity is
  /// preserved.
  public private(set) var view: NSView?

  /// The `BlockAttachment` instance backing this host. Cached so that every
  /// re-vend of the attachment-bearing display paragraph reuses the SAME
  /// `NSTextAttachment` object, which in turn keeps AppKit's view-provider
  /// mounting machinery stable.
  ///
  /// Why this matters: the content-storage delegate rebuilds the attachment
  /// paragraph on every `apply()` call (which fires on every selection /
  /// edit). If a fresh `BlockAttachment` were vended each time, AppKit would
  /// see attachment-identity churn and refuse to re-parent the cached view —
  /// falling back to the system's default "document icon" placeholder. By
  /// pinning the attachment here for the host's lifetime, the attachment,
  /// its view provider, and the embedded view all keep their identities
  /// across the unmount/re-mount cycles that accompany text changes — the
  /// same shape Spike S1.1 validated.
  ///
  /// Marked `nonisolated(unsafe)` because the field is read from
  /// `nonisolated` AppKit hooks; in practice all access is on the main
  /// thread.
  nonisolated(unsafe) private var cachedAttachment: BlockAttachment?

  /// Last-seen "is cursor inside" state. The registry compares this
  /// against the new selection on every selection change and calls
  /// `cursorPresenceChanged` only on transitions.
  internal var lastInside: Bool = false

  // MARK: - Init / lifecycle

  init(
    spec: BlockRendererSpec,
    textView: NSTextView,
    rendererFactory: () -> BlockRenderer
  ) {
    self.blockTypeTag = spec.blockTypeTag
    self.spec = spec
    self.textView = textView
    self.renderer = rendererFactory()
  }

  /// Apply a new spec to this host. The range and/or payload may have
  /// changed; the renderer is notified so it can reflow.
  func updateSpec(_ newSpec: BlockRendererSpec) {
    self.spec = newSpec
    renderer?.update(spec: newSpec, host: self)
  }

  /// Build (or return cached) the renderer's view. Called by the
  /// `BlockAttachmentViewProvider`'s `loadView()` to vend the view to
  /// AppKit. Identity is stable for the host's lifetime.
  ///
  /// Marked `nonisolated` because AppKit calls `loadView()` from a context
  /// the compiler can't prove is MainActor-isolated; in practice it always
  /// runs on the main thread during layout. The body asserts the
  /// isolation and accesses the MainActor-isolated state.
  nonisolated func ensureView() -> NSView {
    MainActor.assumeIsolated {
      if let v = view { return v }
      guard let renderer else {
        // Should be unreachable — renderer is built in init. Returning a
        // bare placeholder keeps AppKit honest if someone mis-uses the host.
        let placeholder = NSView(frame: .zero)
        view = placeholder
        return placeholder
      }
      let v = renderer.makeView(host: self)
      view = v
      return v
    }
  }

  /// Build (or return cached) the `BlockAttachment` bound to this host.
  /// Called by `TextKit2ContentStorageDelegate.buildAttachmentParagraph`
  /// every time the attachment paragraph is re-vended. Returning the same
  /// instance forever (until `dispose()`) is what stops AppKit from
  /// dropping the embedded view and falling back to the default file-icon
  /// placeholder when the paragraph is rebuilt on selection / edit.
  func ensureAttachment() -> BlockAttachment {
    if let a = cachedAttachment { return a }
    let a = BlockAttachment(host: self)
    cachedAttachment = a
    return a
  }

  /// Tear down the renderer and drop the view. Called by the registry
  /// when the host is retired (range no longer in any spec, or text view
  /// going away).
  func dispose() {
    renderer?.tearDown()
    renderer = nil
    view?.removeFromSuperview()
    view = nil
    // Drop the cached attachment; the storage will release its reference
    // when the spec range disappears from the next paragraph build.
    cachedAttachment = nil
  }

  // MARK: - Source / cursor accessors (renderer-facing)

  /// Read current source for this host's range from the main text storage.
  /// Always reads live; never cached. Returns the empty string if the
  /// text view has gone away or the range is out of bounds.
  public func sourceText() -> String {
    guard let textView, let storage = textView.textStorage else { return "" }
    let total = storage.length
    let safe = NSRange(
      location: min(spec.range.location, total),
      length: min(spec.range.length, max(0, total - min(spec.range.location, total)))
    )
    guard safe.length > 0 else { return "" }
    return (storage.string as NSString).substring(with: safe)
  }

  /// True when the main view's selection intersects (or is contained by)
  /// this host's range. A zero-length cursor exactly at either endpoint
  /// counts as inside (matches the inline-construct convention used
  /// elsewhere in the renderer).
  public var isCursorInside: Bool {
    guard let textView else { return false }
    return Self.rangeOverlapsCursor(spec.range, cursor: textView.selectedRange())
  }

  /// Move the main text view's selection to a position offset from this
  /// host's range start. Renderers use this to project a click in their
  /// own coordinate space onto a source offset. `length` defaults to 0
  /// (a zero-length cursor).
  public func setMainSelection(toSourceOffset offset: Int, length: Int = 0) {
    guard let textView else { return }
    let absolute = spec.range.location + offset
    textView.setSelectedRange(NSRange(location: absolute, length: length))
  }

  /// Forward a key-event-equivalent to the main responder chain. Used by
  /// edit-in-place renderers so a keystroke targeted at the embedded view
  /// ends up in the main `NSTextView`'s storage at the current source
  /// position. Phase 2.1 ships the no-op renderer which does not call
  /// this; it's stubbed in for Phase 2.2.
  public func forwardKeyEvent(_ event: NSEvent) {
    guard let textView, let window = textView.window else { return }
    window.makeFirstResponder(textView)
    window.sendEvent(event)
  }

  // MARK: - Helpers

  static func rangeOverlapsCursor(_ range: NSRange, cursor: NSRange) -> Bool {
    let cursorEnd = cursor.location + cursor.length
    let nodeEnd = range.location + range.length
    if cursor.location < nodeEnd && cursorEnd > range.location { return true }
    if cursor.length == 0 {
      if cursor.location == range.location || cursor.location == nodeEnd {
        return true
      }
    }
    return false
  }
}
