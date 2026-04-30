import AppKit
import ObjectiveC

/// `NSTextView` subclass used on the TextKit 2 path.
///
/// With the content delegate's length-matching invariant
/// (`displayString.length == sourceRange.length`), TK2's hit-test and
/// selection-navigation logic operate on real source offsets — clicks land
/// on the right glyph and arrow keys advance one source char per press
/// without translation.
///
/// What this subclass adds is the *editor UX* of skipping over hidden
/// source chars during character-level keyboard navigation. The cursor
/// must never come to rest on a hidden delimiter (e.g. a `**` around
/// bold), otherwise the user sees a blank "stuck" caret. We override the
/// move / extend selectors to compute the destination directly from the
/// source string + `hiddenIndexes`, skipping any contiguous hidden run as
/// a single atomic step. Word-level motion delegates to super (whose
/// destinations are now in real source coordinates) and snaps the result
/// to the nearest visible source position.
@MainActor
final class TextKit2MarkdownTextView: NSTextView {

  // MARK: - Cursor navigation

  /// Compute the document length once. Used as the upper bound for forward
  /// motion since the source-end position is always a valid landing site.
  private func documentLength() -> Int {
    (string as NSString).length
  }

  /// Walk forward from `sourceOffset` to the next visible source offset.
  ///
  /// The walk is over source positions, skipping any in
  /// `delegate.hiddenIndexes`. Paragraph boundaries are not load-bearing
  /// — every source position is either visible or hidden, and we land on
  /// the next visible one regardless of which paragraph element owns it.
  /// This is critical for the inter-block-gap case where a `\n`-only
  /// source paragraph is fully absorbed (its element is skipped from
  /// layout entirely): the absorbed `\n` is in `hiddenIndexes` so we
  /// stride past it just like any other hidden char.
  ///
  /// Returns nil at document end.
  private func nextSourceOffset(after sourceOffset: Int) -> Int? {
    guard let storage = textContentStorage,
      let delegate = storage.delegate as? TextKit2ContentStorageDelegate
    else {
      let next = sourceOffset + 1
      return next <= documentLength() ? next : nil
    }
    let docLen = documentLength()
    if sourceOffset >= docLen { return nil }

    var pos = sourceOffset + 1
    while pos < docLen {
      if !delegate.hiddenIndexes.contains(pos) {
        return pos
      }
      pos += 1
    }
    // No visible char remains; doc end is a valid landing if we weren't
    // already there.
    return docLen
  }

  /// Walk backward from `sourceOffset` to the previous visible source
  /// offset. Symmetric to `nextSourceOffset(after:)` — operates over
  /// source positions, skipping `hiddenIndexes`. Returns nil at doc start.
  private func previousSourceOffset(before sourceOffset: Int) -> Int? {
    guard sourceOffset > 0 else { return nil }
    guard let storage = textContentStorage,
      let delegate = storage.delegate as? TextKit2ContentStorageDelegate
    else {
      return sourceOffset - 1
    }
    var pos = sourceOffset - 1
    while pos > 0 {
      if !delegate.hiddenIndexes.contains(pos) {
        return pos
      }
      pos -= 1
    }
    // pos == 0 — return 0 if not hidden, otherwise still 0 (document
    // start is a valid landing even if nominally "hidden").
    return 0
  }

  // MARK: - Move overrides

  override func moveRight(_ sender: Any?) {
    let current = endOfSelection(forwardMotion: true)
    DebugTrace.log("move.start", ["dir": "right", "from": current])
    trackedSelectionAnchor = nil
    if let next = nextSourceOffset(after: current) {
      setSelectedRange(NSRange(location: next, length: 0))
      DebugTrace.log("move.end", [
        "dir": "right",
        "from": current,
        "to": next,
        "post_selection": selectedRange().location,
      ])
    } else {
      super.moveRight(sender)
      DebugTrace.log("move.end", [
        "dir": "right",
        "from": current,
        "to": "super",
        "post_selection": selectedRange().location,
      ])
    }
  }

  override func moveLeft(_ sender: Any?) {
    let current = endOfSelection(forwardMotion: false)
    DebugTrace.log("move.start", ["dir": "left", "from": current])
    trackedSelectionAnchor = nil
    if let prev = previousSourceOffset(before: current) {
      setSelectedRange(NSRange(location: prev, length: 0))
      DebugTrace.log("move.end", [
        "dir": "left",
        "from": current,
        "to": prev,
        "post_selection": selectedRange().location,
      ])
    } else {
      super.moveLeft(sender)
      DebugTrace.log("move.end", [
        "dir": "left",
        "from": current,
        "to": "super",
        "post_selection": selectedRange().location,
      ])
    }
  }

  override func moveRightAndModifySelection(_ sender: Any?) {
    DebugTrace.log("move.start", ["dir": "shift+right"])
    extendSelection(forward: true)
    DebugTrace.log("move.end", [
      "dir": "shift+right",
      "post_selection_location": selectedRange().location,
      "post_selection_length": selectedRange().length,
    ])
  }

  override func moveLeftAndModifySelection(_ sender: Any?) {
    DebugTrace.log("move.start", ["dir": "shift+left"])
    extendSelection(forward: false)
    DebugTrace.log("move.end", [
      "dir": "shift+left",
      "post_selection_location": selectedRange().location,
      "post_selection_length": selectedRange().length,
    ])
  }

  /// Word-level: delegate to super for the destination then snap to the
  /// nearest visible source position. With the length-matching invariant
  /// super's result is already a real source offset, but it may have landed
  /// on a hidden char — in which case we must skip to the next visible one.
  override func moveWordRight(_ sender: Any?) {
    let before = endOfSelection(forwardMotion: true)
    trackedSelectionAnchor = nil
    super.moveWordRight(sender)
    snapSelectionToVisible(forward: true, fallbackFrom: before)
  }

  override func moveWordLeft(_ sender: Any?) {
    let before = endOfSelection(forwardMotion: false)
    trackedSelectionAnchor = nil
    super.moveWordLeft(sender)
    snapSelectionToVisible(forward: false, fallbackFrom: before)
  }

  override func moveWordRightAndModifySelection(_ sender: Any?) {
    let anchorBefore = anchorForExtension(forward: true)
    let headBefore = endOfSelection(forwardMotion: true)
    super.moveWordRightAndModifySelection(sender)
    trackedSelectionAnchor = anchorBefore
    snapExtendedSelectionToVisible(
      anchor: anchorBefore, headBefore: headBefore, forward: true)
  }

  override func moveWordLeftAndModifySelection(_ sender: Any?) {
    let anchorBefore = anchorForExtension(forward: false)
    let headBefore = endOfSelection(forwardMotion: false)
    super.moveWordLeftAndModifySelection(sender)
    trackedSelectionAnchor = anchorBefore
    snapExtendedSelectionToVisible(
      anchor: anchorBefore, headBefore: headBefore, forward: false)
  }

  // MARK: - Helpers

  /// The "moving" end of the current selection. For a zero-length cursor,
  /// this is the cursor itself. For a range selection, this is the end
  /// (forward motion) or start (backward motion).
  private func endOfSelection(forwardMotion: Bool) -> Int {
    let sel = selectedRange()
    if sel.length == 0 { return sel.location }
    return forwardMotion ? sel.location + sel.length : sel.location
  }

  // MARK: - Anchor tracking
  //
  // The "anchor" of an extending selection is the end the user is NOT moving;
  // shift+arrow grows / shrinks the selection toward / away from it. NSRange
  // alone doesn't preserve which end is the anchor, so a naive heuristic
  // (always treat `sel.location` as the anchor) corrupts the selection when
  // the user reverses direction mid-extension. Concretely: from cursor at 4,
  // shift+right shift+right → [4, 6) with anchor=4; shift+left should shrink
  // to [4, 5), but the heuristic guesses anchor=6 and head=4, then "extends
  // backward" from head 4 to 3, producing [3, 6) — the selection grows in the
  // wrong direction.
  //
  // Fix: persist the anchor on the text view via an Objective-C associated
  // object. Stored properties on a Swift subclass installed via
  // `object_setClass` would change memory layout and corrupt the underlying
  // `NSTextView` instance, so we use the runtime side-table instead.
  // The anchor is set on every shift+arrow / shift+word-arrow that grows or
  // shrinks the selection, and cleared on every non-extending motion or
  // external selection change (`textViewDidChangeSelection` clears it after
  // the Coordinator-driven re-render so the next user gesture starts fresh).

  private static var anchorKey: UInt8 = 0

  /// The persisted anchor for an in-progress shift-arrow extension, or `nil`
  /// if no extension is active. Reading this value is cheap; it's an
  /// associated-object lookup keyed by the text view instance.
  var trackedSelectionAnchor: Int? {
    get {
      objc_getAssociatedObject(self, &Self.anchorKey) as? Int
    }
    set {
      objc_setAssociatedObject(
        self, &Self.anchorKey, newValue, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
    }
  }

  /// Resolve the anchor at the start of an extension. Prefers the tracked
  /// anchor if one is recorded AND it's still consistent with the current
  /// selection (one of the selection's endpoints sits on the tracked anchor);
  /// otherwise infers from the current selection (start for length-0
  /// selections, otherwise the end opposite the moving edge). The
  /// consistency check rejects a stale anchor left over from a prior
  /// extension that's been invalidated by a click, programmatic
  /// `setSelectedRange`, or a non-extending move whose path didn't go
  /// through the anchor-clearing code (e.g. a future motion override).
  private func anchorForExtension(forward: Bool) -> Int {
    let sel = selectedRange()
    if let tracked = trackedSelectionAnchor,
      tracked == sel.location || tracked == sel.location + sel.length
    {
      return tracked
    }
    trackedSelectionAnchor = nil
    if sel.length == 0 { return sel.location }
    // No (valid) prior extension — assume the conventional anchor (start
    // for a forward selection, end for a backward one). This matches the
    // first-extension-from-a-zero-length-cursor case after a fresh click.
    return forward ? sel.location : sel.location + sel.length
  }

  /// Read the anchor end for word-extension overrides. Word-modifying
  /// selectors don't know which direction is "forward" until they're called,
  /// so they pass the explicit direction here.
  private func anchorForExtension() -> Int {
    anchorForExtension(forward: true)
  }

  private func extendSelection(forward: Bool) {
    let sel = selectedRange()
    let anchor = anchorForExtension(forward: forward)
    // The "moving" head is the end opposite the anchor. For length-0
    // selections both ends coincide, so head == anchor and the next motion
    // bootstraps the extension with the tracked anchor === sel.location.
    let head: Int
    if sel.length == 0 {
      head = sel.location
    } else if anchor == sel.location {
      head = sel.location + sel.length
    } else {
      head = sel.location
    }
    let newHead: Int?
    if forward {
      newHead = nextSourceOffset(after: head)
    } else {
      newHead = previousSourceOffset(before: head)
    }
    guard let nh = newHead else { return }
    trackedSelectionAnchor = anchor
    let lo = min(anchor, nh)
    let hi = max(anchor, nh)
    setSelectedRange(NSRange(location: lo, length: hi - lo))
  }

  /// After delegating to super for a (non-extending) move, super has set the
  /// selection at a real source offset (length-matching invariant). If the
  /// destination is a hidden source char, snap to the nearest visible
  /// position in the direction of motion. If super made no progress (or
  /// went the wrong way), fall back to a single-source-step from
  /// `fallbackFrom`.
  private func snapSelectionToVisible(forward: Bool, fallbackFrom: Int) {
    let sel = selectedRange()
    let raw = sel.length == 0 ? sel.location : sel.location + sel.length
    let snapped: Int
    if let storage = textContentStorage,
      let delegate = storage.delegate as? TextKit2ContentStorageDelegate,
      delegate.hiddenIndexes.contains(raw)
    {
      snapped =
        (forward
          ? nextSourceOffset(after: raw)
          : previousSourceOffset(before: raw)) ?? raw
    } else {
      snapped = raw
    }
    let final: Int
    if forward, snapped > fallbackFrom {
      final = snapped
    } else if !forward, snapped < fallbackFrom {
      final = snapped
    } else {
      // Super's destination was a no-op or wrong direction; fall back to
      // single-char source step.
      final =
        (forward
          ? nextSourceOffset(after: fallbackFrom)
          : previousSourceOffset(before: fallbackFrom)) ?? fallbackFrom
    }
    setSelectedRange(NSRange(location: final, length: 0))
  }

  private func snapExtendedSelectionToVisible(
    anchor: Int, headBefore: Int, forward: Bool
  ) {
    let sel = selectedRange()
    let raw =
      forward
      ? sel.location + sel.length
      : sel.location
    var snapped = raw
    if let storage = textContentStorage,
      let delegate = storage.delegate as? TextKit2ContentStorageDelegate,
      delegate.hiddenIndexes.contains(raw)
    {
      snapped =
        (forward
          ? nextSourceOffset(after: raw)
          : previousSourceOffset(before: raw)) ?? raw
    }
    if forward, snapped <= headBefore {
      snapped = nextSourceOffset(after: headBefore) ?? headBefore
    } else if !forward, snapped >= headBefore {
      snapped = previousSourceOffset(before: headBefore) ?? headBefore
    }
    let lo = min(anchor, snapped)
    let hi = max(anchor, snapped)
    setSelectedRange(NSRange(location: lo, length: hi - lo))
  }
}
