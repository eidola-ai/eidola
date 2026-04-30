import AppKit

/// `NSTextView` subclass used on the TextKit 2 path.
///
/// Bridges TextKit 2's display-coordinate world to our source-coordinate
/// model. The content delegate's display attributedString omits hidden
/// chars (e.g. `**` in bold) and substitutes glyphs for marker chars
/// (bullets, checkboxes), so display offsets are not equal to source
/// offsets. TK2's hit-test, character-level cursor navigation, and
/// selection-extension logic all return / consume offsets that they
/// believe are source offsets but are actually computed in display
/// coordinates — empirically verified in Phase 0 / Phase 2.5.
///
/// This subclass adds two layers of translation:
///
///   1. `characterIndexForInsertion(at:)` — clicks resolve to a source
///      offset that's actually a display offset. We re-walk the paragraph
///      against the current `hiddenIndexes` set to translate it back.
///
///   2. Character-level move / extend selectors (`moveLeft(_:)`,
///      `moveRight(_:)`, `moveLeftAndModifySelection(_:)`,
///      `moveRightAndModifySelection(_:)`, and the word variants) — TK2's
///      built-in selection navigation walks display chars but reports the
///      result as a source offset, which can jump past hidden runs to
///      arbitrary positions when the display vs. source lengths diverge.
///      We compute the destination ourselves by walking source chars and
///      skipping hidden ones, then set the selection imperatively.
///
/// All other behavior (vertical motion, double-click word selection, drag
/// selection extension, etc.) is inherited from `NSTextView` — it routes
/// through `characterIndexForInsertion(at:)` for hit-tested operations,
/// which is already translated.
@MainActor
final class TextKit2MarkdownTextView: NSTextView {

  // MARK: - Hit-test (clicks)

  override func characterIndexForInsertion(at point: NSPoint) -> Int {
    let raw = super.characterIndexForInsertion(at: point)
    DebugTrace.log("hitTest.start", [
      "x": Double(point.x),
      "y": Double(point.y),
      "raw": raw,
    ])
    let translated = translateHitTestIndex(raw)
    DebugTrace.log("hitTest.end", [
      "raw": raw,
      "translated": translated,
    ])
    return translated
  }

  /// Pure translation step extracted from the override so it can be tested
  /// without depending on super's layout-dependent click resolution.
  ///
  /// `baseIndex` is what TK2's hit-test reports — empirically a *display*
  /// offset (within the paragraph) reported as if it were a source offset.
  /// We map it through the paragraph's display-to-source array to recover
  /// the actual source position the visible glyph corresponds to.
  ///
  /// When TK2 returns a value at exactly the document end (no character
  /// hit) or for a paragraph the delegate has no info about, we pass it
  /// through unchanged. The paragraph lookup is on-demand (queries the
  /// content storage) so the result is correct regardless of viewport
  /// scroll state — an earlier prefix-cache implementation went stale on
  /// scroll.
  func translateHitTestIndex(_ baseIndex: Int) -> Int {
    guard let storage = textContentStorage,
      let delegate = storage.delegate as? TextKit2ContentStorageDelegate,
      storage.location(
        storage.documentRange.location, offsetBy: baseIndex) != nil
    else { return baseIndex }

    // Find the paragraph that owns the base location by walking up: TK2's
    // enumerator from a location at exactly the boundary of two paragraphs
    // visits the *next* paragraph first, so for boundary cases we'd miss
    // the paragraph we actually clicked on. Walk by enumerating from doc
    // start until we find an element whose range contains baseIndex.
    var paragraphRange: NSRange?
    let docStart = storage.documentRange.location
    storage.enumerateTextElements(from: docStart, options: []) { element in
      guard let elemRange = element.elementRange else { return true }
      let start = storage.offset(from: docStart, to: elemRange.location)
      let length = storage.offset(
        from: elemRange.location, to: elemRange.endLocation)
      let r = NSRange(location: start, length: length)
      if NSLocationInRange(baseIndex, r) || (length == 0 && start == baseIndex) {
        paragraphRange = r
        return false
      }
      // Past the click — stop walking forward
      if start > baseIndex { return false }
      return true
    }

    guard let paragraphRange else { return baseIndex }

    let displayOffset = baseIndex - paragraphRange.location
    let map = delegate.displayToSourceMap(forParagraphSourceRange: paragraphRange)

    if displayOffset < 0 {
      // baseIndex is before the paragraph (shouldn't happen, defensive)
      return baseIndex
    }
    if displayOffset < map.count {
      return map[displayOffset]
    }
    // Display offset past last visible char → land at paragraph end
    // (start of paragraph separator / next paragraph).
    return paragraphRange.location + paragraphRange.length
  }

  // MARK: - Cursor navigation

  /// Find the paragraph (TK2 element) source range containing `sourceOffset`.
  /// Returns nil if no paragraph contains it. For the boundary case where
  /// `sourceOffset` is exactly at the start of paragraph N (== end of N-1),
  /// returns paragraph N — the more useful interpretation for forward motion.
  private func paragraphRange(containing sourceOffset: Int) -> NSRange? {
    guard let storage = textContentStorage else { return nil }
    let docStart = storage.documentRange.location
    var found: NSRange?
    storage.enumerateTextElements(from: docStart, options: []) { element in
      guard let elemRange = element.elementRange else { return true }
      let start = storage.offset(from: docStart, to: elemRange.location)
      let length = storage.offset(
        from: elemRange.location, to: elemRange.endLocation)
      let r = NSRange(location: start, length: length)
      if sourceOffset >= start && sourceOffset < start + length {
        found = r
        return false
      }
      // sourceOffset == start + length is the boundary; remember and keep
      // looking — if the next paragraph starts at the same position we'll
      // prefer it (more useful for forward motion). If no next paragraph,
      // fall back to this one.
      if sourceOffset == start + length {
        found = r
      }
      return true
    }
    return found
  }

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

  /// Word-level: delegate to super for the destination then translate
  /// through the paragraph map. Word boundaries depend on system locale
  /// and word-break tables we don't want to reimplement; super's result
  /// is in display-offset-as-source coordinates (same as character motion),
  /// so the same display→source translation applies.
  override func moveWordRight(_ sender: Any?) {
    let before = endOfSelection(forwardMotion: true)
    super.moveWordRight(sender)
    translateAndCommitSelection(forward: true, fallbackFrom: before)
  }

  override func moveWordLeft(_ sender: Any?) {
    let before = endOfSelection(forwardMotion: false)
    super.moveWordLeft(sender)
    translateAndCommitSelection(forward: false, fallbackFrom: before)
  }

  override func moveWordRightAndModifySelection(_ sender: Any?) {
    let anchorBefore = anchorForExtension()
    let headBefore = endOfSelection(forwardMotion: true)
    super.moveWordRightAndModifySelection(sender)
    translateAndCommitExtendedSelection(
      anchor: anchorBefore, headBefore: headBefore, forward: true)
  }

  override func moveWordLeftAndModifySelection(_ sender: Any?) {
    let anchorBefore = anchorForExtension()
    let headBefore = endOfSelection(forwardMotion: false)
    super.moveWordLeftAndModifySelection(sender)
    translateAndCommitExtendedSelection(
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

  /// The "anchor" end of the current selection — opposite of the moving end.
  private func anchorForExtension() -> Int {
    let sel = selectedRange()
    if sel.length == 0 { return sel.location }
    // We don't track which end is the anchor on a plain NSRange; assume
    // the start is the anchor for forward selections, end for backward.
    // That heuristic matches the TK2 NSTextSelection model when the user
    // hasn't reversed direction mid-extension.
    return sel.location
  }

  private func extendSelection(forward: Bool) {
    let sel = selectedRange()
    let head: Int
    let anchor: Int
    if sel.length == 0 {
      head = sel.location
      anchor = sel.location
    } else {
      head = forward ? sel.location + sel.length : sel.location
      anchor = forward ? sel.location : sel.location + sel.length
    }
    let newHead: Int?
    if forward {
      newHead = nextSourceOffset(after: head)
    } else {
      newHead = previousSourceOffset(before: head)
    }
    guard let nh = newHead else { return }
    let lo = min(anchor, nh)
    let hi = max(anchor, nh)
    setSelectedRange(NSRange(location: lo, length: hi - lo))
  }

  /// After delegating to super for a move, super has set the selection
  /// using TK2's display-as-source result. Translate the new head through
  /// the paragraph map; if translation moves us in the wrong direction or
  /// stalls, fall back to a single-character source step from `fallbackFrom`.
  private func translateAndCommitSelection(forward: Bool, fallbackFrom: Int) {
    let sel = selectedRange()
    let rawHead = sel.length == 0 ? sel.location : sel.location + sel.length
    let translated = displayOffsetToSource(rawHead)
    let final: Int
    if forward, translated > fallbackFrom {
      final = translated
    } else if !forward, translated < fallbackFrom {
      final = translated
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

  private func translateAndCommitExtendedSelection(
    anchor: Int, headBefore: Int, forward: Bool
  ) {
    let sel = selectedRange()
    let rawHead =
      forward
      ? sel.location + sel.length
      : sel.location
    var translated = displayOffsetToSource(rawHead)
    if forward, translated <= headBefore {
      translated = nextSourceOffset(after: headBefore) ?? headBefore
    } else if !forward, translated >= headBefore {
      translated = previousSourceOffset(before: headBefore) ?? headBefore
    }
    let lo = min(anchor, translated)
    let hi = max(anchor, translated)
    setSelectedRange(NSRange(location: lo, length: hi - lo))
  }

  /// Translate a display-offset-as-source value (what TK2 returns from
  /// nav / hit-test) into a real source offset by walking the containing
  /// paragraph's display-to-source map.
  private func displayOffsetToSource(_ offset: Int) -> Int {
    guard let storage = textContentStorage,
      let delegate = storage.delegate as? TextKit2ContentStorageDelegate
    else { return offset }
    guard let para = paragraphRange(containing: offset) else { return offset }
    let displayIdx = offset - para.location
    let map = delegate.displayToSourceMap(forParagraphSourceRange: para)
    if displayIdx < 0 { return offset }
    if displayIdx < map.count { return map[displayIdx] }
    return para.location + para.length
  }
}
