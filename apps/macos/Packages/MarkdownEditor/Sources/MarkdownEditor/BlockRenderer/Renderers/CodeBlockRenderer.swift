import AppKit

/// Phase 2.2's real code-block renderer. Replaces the Phase 2.1
/// `NoopBlockRenderer` (a flat colored rectangle) with an embedded
/// `NSScrollView { NSTextView }` subtree that mirrors the code-block source
/// from the main `NSTextStorage` and supports horizontal scroll for long
/// lines.
///
/// ## Source-of-truth model
///
/// The embedded `NSTextView`'s storage is a *display mirror* over the main
/// storage's slice for `spec.range`. It is never canonical — every keystroke
/// targeted at the embedded view is rejected locally and re-applied to the
/// main storage at the mapped source offset. The Elm loop fires
/// `textDidChange` on the main view, the renderer re-emits the spec, the
/// applicator calls `update(spec:host:)`, and we rewrite the embedded
/// storage from the new main slice. Single source of truth (goal #2)
/// preserved.
///
/// ## First-responder swap
///
/// When the cursor enters / leaves the host's source range, the host swaps
/// first responder between the embedded text view and the main text view.
/// Per Spike S2 this is glitch-free as long as both views share the same
/// window — caret blink and keystroke routing follow first-responder
/// cleanly. We deliberately do NOT swap on the very first `update()` call
/// (avoid stealing focus on initial render); the swap engages on the next
/// `cursorPresenceChanged(true, ...)` transition.
///
/// ## Known flicker
///
/// Spike S1.1 found that AppKit unmounts the attachment view from its
/// parent on text changes and re-mounts the SAME view instance shortly
/// after. During the gap the embedded view isn't composited, leaving a
/// brief visual gap. A previous attempt to mitigate via the host scroll
/// view's `layer.contents` failed: setting `contents` on a layer-backed
/// host whose subview (`NSTextView`) draws into that same layer turns
/// the layer into a static bitmap and bypasses the text view's drawing
/// entirely. A real fix needs an out-of-band snapshot view mounted in
/// the main text view's hierarchy (so it survives the embedded view's
/// own unmount). Deferred — accept the brief flicker on edits-inside
/// for now.
@MainActor
final class CodeBlockRenderer: NSObject, BlockRenderer {

  // MARK: - State

  /// The host scroll view returned from `makeView`. Identity is preserved
  /// across AppKit unmount/re-mount cycles.
  private weak var scrollView: NSScrollView?

  /// The embedded text view inside the scroll view's clip view. Identity
  /// is preserved across re-mounts (AppKit gives us the same instance back
  /// per S1.1). Becomes the first responder when the cursor is inside the
  /// host's source range.
  private weak var embeddedTextView: EmbeddedCodeTextView?

  /// Weak back-reference to the host this renderer instance serves. The
  /// registry creates one renderer per host (1:1) so this is unambiguous.
  /// Used by `shouldChangeTextIn` to avoid a view-hierarchy walk that
  /// fails when the embedded view's window connection is nil — which
  /// AppKit toggles constantly during TK2's viewport-element-view recycle
  /// (Spike S1.1). When the walk fails, the delegate would default to
  /// allowing local edits, decoupling embedded storage from the canonical
  /// main storage and producing "competing buffers" symptoms.
  private weak var host: BlockRenderHost?

  /// Re-entry guard for the embedded `textDidChange` / `shouldChangeTextIn`
  /// hooks. Set to `true` while we're rewriting the embedded storage from
  /// the main slice (in `update(spec:host:)`); the embedded delegate sees
  /// the flag and lets the change through without forwarding to the main
  /// storage (which would loop forever).
  private var isApplyingExternalUpdate = false

  /// Re-entry guard for the *forward* direction: `shouldChangeTextIn`
  /// allows the embedded view to apply the user's edit locally AND
  /// synchronously forwards the same edit to the main storage. The main
  /// storage mutation triggers the apply pipeline, which reconciles, and
  /// eventually calls back into our `update()`. By the time `update()`
  /// runs, the embedded view will already have the new content (from its
  /// own local edit). The guard tells `update()` to skip the rewrite,
  /// avoiding both a redundant storage replacement and the AppKit
  /// re-entrancy crash that synchronous mutation triggered before this
  /// design.
  private var isForwardingLocalEdit = false

  /// Cached length of the embedded view's storage from the last apply pass.
  /// Used to decide between in-place character replacement and a full
  /// `setAttributedString` reset (resets clobber the embedded scroll
  /// position and selection; we avoid them when content can be patched).
  private var lastAppliedEmbeddedLength: Int = 0

  /// Tracks whether we have done at least one `update()` for this host.
  /// `cursorPresenceChanged` consults this so the very first inside/outside
  /// notification (which Phase 2.1's host fires from `lastInside == false`
  /// when the spec range overlaps the cursor at first render) does not
  /// steal first responder away from whatever AppKit had focused
  /// (typically nothing yet, but be defensive).
  private var hasAppliedAtLeastOnce: Bool = false

  // MARK: - BlockRenderer

  func makeView(host: BlockRenderHost) -> NSView {
    let scrollView = NSScrollView(frame: .zero)
    scrollView.borderType = .noBorder
    scrollView.hasVerticalScroller = false
    scrollView.hasHorizontalScroller = true
    // Overlay scroller style avoids stealing the embedded text view's
    // width — the scroller floats on top instead of insetting the content
    // area. Important so the no-wrap text container width isn't reduced
    // every time the scroller appears.
    scrollView.scrollerStyle = .overlay
    scrollView.autohidesScrollers = true
    scrollView.drawsBackground = true
    scrollView.backgroundColor = .quaternaryLabelColor.withAlphaComponent(0.5)

    let textView = EmbeddedCodeTextView(frame: .zero)
    textView.minSize = NSSize(width: 0, height: 0)
    textView.maxSize = NSSize(
      width: CGFloat.greatestFiniteMagnitude,
      height: CGFloat.greatestFiniteMagnitude)
    textView.isVerticallyResizable = true
    textView.isHorizontallyResizable = true
    textView.autoresizingMask = [.width, .height]
    textView.isRichText = true
    textView.isEditable = true
    textView.isSelectable = true
    textView.allowsUndo = false
    textView.isAutomaticQuoteSubstitutionEnabled = false
    textView.isAutomaticDashSubstitutionEnabled = false
    textView.isAutomaticTextReplacementEnabled = false
    textView.isAutomaticSpellingCorrectionEnabled = false
    textView.isContinuousSpellCheckingEnabled = false
    textView.isGrammarCheckingEnabled = false
    textView.drawsBackground = false
    textView.textContainerInset = NSSize(width: 6, height: 4)
    textView.font = MarkdownStyle.default.codeFont
    textView.textColor = .labelColor
    textView.delegate = self

    if let container = textView.textContainer {
      // No-wrap: container width is unbounded so the text view's
      // `usedRect` grows horizontally for long lines; the surrounding
      // scroll view handles the horizontal scroll.
      container.widthTracksTextView = false
      container.heightTracksTextView = false
      container.size = NSSize(
        width: CGFloat.greatestFiniteMagnitude,
        height: CGFloat.greatestFiniteMagnitude)
      container.lineFragmentPadding = 0
    }

    scrollView.documentView = textView

    self.scrollView = scrollView
    self.embeddedTextView = textView
    self.host = host

    // The reconcile pipeline calls `update(spec:host:)` BEFORE AppKit
    // demands the view (which is what triggers `makeView`). That first
    // update is dropped by the `guard let textView` early-return because
    // we hadn't built the view yet. Catch up now so the view is created
    // with the current source already populated, not blank.
    update(spec: host.spec, host: host)

    return scrollView
  }

  func update(spec: BlockRendererSpec, host: BlockRenderHost) {
    guard let textView = embeddedTextView else { return }

    // Forward-edit guard: the embedded view just applied the user's
    // keystroke locally and is in the middle of synchronously syncing
    // that change up to the main storage. The reconcile-driven `update`
    // we're handling is exactly that round trip. The embedded view
    // already has the new content; rewriting its storage now would (a)
    // be redundant, and (b) cause the AppKit re-entrancy crash that
    // killed earlier synchronous designs.
    if isForwardingLocalEdit { return }

    let source = host.sourceText()
    let display = makeDisplayAttributedString(
      source: source, spec: spec, host: host)

    // Re-entry guard: the storage replacement below will fire
    // `shouldChangeTextIn` / `textDidChange` on the embedded view's
    // delegate (which is `self`). The flag tells the delegate this is our
    // own programmatic apply and not a user keystroke.
    isApplyingExternalUpdate = true
    defer {
      isApplyingExternalUpdate = false
    }

    if let storage = textView.textStorage {
      let currentLen = storage.length
      let newLen = display.length
      // Heuristic: only do an in-place full-range replace when current
      // length is non-zero AND length difference is "small" relative to
      // current length. A wholesale `setAttributedString` clobbers the
      // embedded selection / scroll position which we want to preserve
      // across text edits; an in-place replace preserves them.
      let materialDelta =
        currentLen == 0
        || abs(newLen - currentLen) > max(8, currentLen / 2)
      storage.beginEditing()
      if materialDelta {
        storage.setAttributedString(display)
      } else {
        storage.replaceCharacters(
          in: NSRange(location: 0, length: currentLen),
          with: display)
      }
      storage.endEditing()
      lastAppliedEmbeddedLength = display.length
    }

    // Force glyph layout for the full storage range. NSTextView's lazy
    // layout would otherwise wait for the next drawRect call on a
    // windowed view — and updates frequently arrive while AppKit has the
    // view temporarily off-window (e.g. between viewport-element-view
    // unmount/remount cycles), so the glyphs never get laid out and the
    // text never renders even after the view comes back. Calling
    // `ensureLayout(forCharacterRange:)` on a windowed-or-not view
    // synchronously lays out the glyphs against the text container so
    // the next drawRect has something to render.
    if let lm = textView.layoutManager, let storage = textView.textStorage {
      let fullGlyphRange = lm.glyphRange(
        forCharacterRange: NSRange(location: 0, length: storage.length),
        actualCharacterRange: nil)
      lm.ensureLayout(forGlyphRange: fullGlyphRange)
    }
    textView.needsDisplay = true
    hasAppliedAtLeastOnce = true

    // Publish the embedded view's just-laid-out content height back to
    // the host so the attachment can grow / shrink along with what the
    // user actually sees inside the embedded view (see Part B of the
    // Phase 2.2 follow-up).
    measureAndPublishHeight()
  }

  func cursorPresenceChanged(_ inside: Bool, host: BlockRenderHost) {
    // Don't steal focus on the very first update — only swap on real
    // transitions after at least one apply has run (so the embedded view
    // exists and has content to focus into).
    guard hasAppliedAtLeastOnce else { return }
    guard let mainTextView = host.textView,
      let window = mainTextView.window
    else { return }

    if inside {
      guard let embedded = embeddedTextView else { return }
      // Translate the main view's selection (a source offset) into the
      // embedded view's local coordinate so the caret appears at the right
      // place inside the embedded text view.
      let mainSel = mainTextView.selectedRange()
      let local = mapMainToLocal(
        mainRange: mainSel, spec: host.spec,
        embeddedLength: embedded.textStorage?.length ?? 0)
      embedded.setSelectedRange(local)
      window.makeFirstResponder(embedded)
    } else {
      window.makeFirstResponder(mainTextView)
    }
  }

  func desiredBounds(host: BlockRenderHost) -> CGRect {
    // Width: defer to the attachment's `attachmentBounds` clamp (which
    // uses the text container's width). The renderer doesn't know the
    // container width independently of the host.
    CGRect(
      x: 0, y: 0,
      width: CGFloat.greatestFiniteMagnitude,
      height: host.spec.reservedHeight)
  }

  func tearDown() {
    if let textView = embeddedTextView {
      textView.delegate = nil
    }
    embeddedTextView?.removeFromSuperview()
    scrollView?.removeFromSuperview()
    embeddedTextView = nil
    scrollView = nil
  }

  // MARK: - Intrinsic height feedback

  /// Measure the embedded view's just-laid-out content height and push
  /// it to the host so `BlockAttachment.attachmentBounds` reads the
  /// real used height instead of the parser's `reservedHeight`
  /// estimate. The host invalidates the layout fragment for its source
  /// range so the outer flow reflows on the next layout pass.
  ///
  /// Called from both edit paths inside `CodeBlockRenderer`:
  /// - The local-edit forward path (after `mainTextView.didChangeText`
  ///   inside `shouldChangeTextIn`), so the height tracks the user's
  ///   typing in real time.
  /// - The external-update path (end of `update(spec:host:)`), so
  ///   programmatic source changes (initial render, external edits)
  ///   also flow into the bounds.
  ///
  /// We add `textContainerInset.height * 2` because the embedded
  /// `NSTextView`'s own padding (set in `makeView`) sits outside the
  /// `usedRect` reported by the layout manager — without it the
  /// reported height clips the visible padding the embedded view
  /// actually draws with.
  ///
  /// `internal` so tests can drive the measurement directly after
  /// programmatically growing the embedded storage; production callers
  /// only invoke it through the two paths listed above.
  internal func measureAndPublishHeight() {
    guard let textView = embeddedTextView, let host = self.host else { return }
    guard let lm = textView.layoutManager,
      let container = textView.textContainer
    else { return }
    // `update(spec:host:)` already calls `ensureLayout(forGlyphRange:)`
    // before this method runs in the external-update path; for the
    // local-edit path the embedded view has just finished applying the
    // user's keystroke and TK1's layout manager has already laid out
    // the affected line. Calling `ensureLayout(for: container)` again
    // is cheap when layout is current (it returns immediately) and
    // makes us robust to callers that haven't pre-flighted layout.
    lm.ensureLayout(for: container)
    let used = lm.usedRect(for: container)
    let inset = textView.textContainerInset
    let height = used.height + inset.height * 2
    host.updateIntrinsicContentHeight(height)
  }

  // MARK: - Source ↔ embedded sync helpers

  /// Build the attributed string the embedded text view should display.
  /// Plain monospace styling — no syntax highlighting (deferred to a
  /// later phase); the embedded view's own paragraph style controls
  /// indent / line height inside the embedded coordinate system.
  private func makeDisplayAttributedString(
    source: String, spec: BlockRendererSpec, host: BlockRenderHost
  ) -> NSAttributedString {
    let style = MarkdownStyle.default
    let paragraph = NSMutableParagraphStyle()
    paragraph.lineHeightMultiple = 1.0
    paragraph.paragraphSpacing = 0
    paragraph.paragraphSpacingBefore = 0
    paragraph.lineBreakMode = .byClipping
    let attrs: [NSAttributedString.Key: Any] = [
      .font: style.codeFont,
      .foregroundColor: NSColor.labelColor,
      .paragraphStyle: paragraph.copy() as! NSParagraphStyle,
    ]
    return NSAttributedString(string: source, attributes: attrs)
  }

  /// Map a main-view source range to a local range inside the embedded
  /// text view. Clamps to the embedded storage's bounds since the spec
  /// range may have changed between the time the cursor moved and the
  /// time we apply.
  private func mapMainToLocal(
    mainRange: NSRange, spec: BlockRendererSpec, embeddedLength: Int
  ) -> NSRange {
    let localStart = max(0, mainRange.location - spec.range.location)
    let clampedStart = min(localStart, embeddedLength)
    let localEnd = max(0, (mainRange.location + mainRange.length) - spec.range.location)
    let clampedEnd = min(localEnd, embeddedLength)
    let length = max(0, clampedEnd - clampedStart)
    return NSRange(location: clampedStart, length: length)
  }

}

// MARK: - Embedded text view

/// `NSTextView` subclass for the embedded code-block content. Adds nothing
/// over the base class today; the subclass exists so future editor-UX
/// affordances (line numbers, copy button, language picker) have a typed
/// home and so tests can identify the embedded view by class.
@MainActor
final class EmbeddedCodeTextView: NSTextView {
  // Intentionally minimal. Behaviour overrides should be added as needed
  // and only when they don't compromise the "main storage is canonical"
  // rule from `CodeBlockRenderer`.
}

// MARK: - NSTextViewDelegate

extension CodeBlockRenderer: NSTextViewDelegate {

  /// Forward keystrokes targeted at the embedded view to the main
  /// `NSTextStorage` at the mapped source offset, so the main storage
  /// stays canonical (goal #2). Returning `false` tells the embedded view
  /// to skip the local edit; the main view's apply pipeline will rewrite
  /// the embedded storage in `update(spec:host:)`.
  func textView(
    _ textView: NSTextView,
    shouldChangeTextIn affectedCharRange: NSRange,
    replacementString: String?
  ) -> Bool {
    // Our own programmatic apply — let it through.
    if isApplyingExternalUpdate { return true }

    guard textView === embeddedTextView,
      let host = self.host
    else {
      // Refuse the edit. Returning `true` here would let the embedded
      // view mutate its local storage, decoupling it from the canonical
      // main storage. Better to drop the keystroke than corrupt the
      // single-source-of-truth invariant.
      return false
    }

    let mainTextView = host.textView
    let mainStorage = mainTextView?.textStorage
    let replacement = replacementString ?? ""

    let spec = host.spec
    let mainRange = NSRange(
      location: spec.range.location + affectedCharRange.location,
      length: affectedCharRange.length)

    // The embedded view IS the editing surface for its range — its
    // storage is canonical for the code-block content while the user is
    // typing inside it. Apply the user's edit locally (return true at
    // the end) so the keystroke is immediate and the cursor stays put,
    // then synchronously forward the same edit up to the main storage so
    // the markdown source stays in sync.
    //
    // Synchronous forwarding is safe — and necessary for low input
    // latency — because the `isForwardingLocalEdit` guard makes our own
    // `update()` callback a no-op during this re-entrant chain. The
    // main storage mutation triggers the apply pipeline, which
    // reconciles, which calls `update()`, which sees the guard and
    // returns immediately without rewriting the embedded storage that
    // already has the new content.
    //
    // An earlier design rejected the local edit and deferred the main
    // storage mutation via `RunLoop.main.perform`. That was supposed to
    // sidestep AppKit re-entrancy, but the runloop mode mismatch
    // (`.eventTracking` while embedded view has focus, `.common` modes
    // not always serviced) caused the deferred block to sit until focus
    // left the block — making typing appear silently broken.
    //
    // Route the upstream mutation through `NSTextView.shouldChangeText` /
    // `didChangeText` rather than mutating the storage directly. The
    // direct `replaceCharacters` path bypasses the main view's
    // delegate-notification chain — `Coordinator.textDidChange` never
    // fires, so the SwiftUI `EditorState.markdown` binding stays stale
    // until the user types in the main view. Going through
    // `shouldChangeText` / `didChangeText` plugs us back into the
    // standard NSTextView edit pipeline (delegate notifications, undo
    // grouping, accessibility events) and the Coordinator's apply loop
    // re-renders synchronously, keeping the binding in sync on every
    // keystroke. The `isForwardingLocalEdit` guard short-circuits the
    // re-entrant `update()` call that ride along.
    //
    // If the main view's delegate vetoes the change via
    // `shouldChangeText`, we propagate the rejection: returning `false`
    // here makes the embedded view drop the keystroke too, keeping the
    // two storages in lock-step rather than letting the embedded view
    // diverge from a main-storage state it failed to mutate.
    if let mainStorage, let mainTextView {
      guard mainTextView.shouldChangeText(
        in: mainRange, replacementString: replacement)
      else { return false }
      isForwardingLocalEdit = true
      defer {
        isForwardingLocalEdit = false
        measureAndPublishHeight()
      }
      mainStorage.beginEditing()
      mainStorage.replaceCharacters(in: mainRange, with: replacement)
      mainStorage.endEditing()
      mainTextView.didChangeText()
      let nsReplacement = replacement as NSString
      let newCaret = mainRange.location + nsReplacement.length
      mainTextView.setSelectedRange(NSRange(location: newCaret, length: 0))
    }

    return true
  }

}
