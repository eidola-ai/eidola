import AppKit

/// `NSTextView` subclass used on the TextKit 2 path. Adds the small
/// hit-test intercept identified by the Phase 0 spike: when a click resolves
/// to the source start of a paragraph that has a hidden prefix (e.g.
/// `# Heading` where `# ` is hidden), translate the cursor to the first
/// non-hidden source offset so the cursor lands at the *visual* click target
/// rather than at the start of the hidden run.
///
/// All other behavior is inherited from `NSTextView`.
@MainActor
final class TextKit2MarkdownTextView: NSTextView {

  override func characterIndexForInsertion(at point: NSPoint) -> Int {
    translateHitTestIndex(super.characterIndexForInsertion(at: point))
  }

  /// Pure translation step extracted from the override so it can be tested
  /// without depending on super's layout-dependent click resolution.
  /// If `baseIndex` is the start of a paragraph that has a hidden prefix,
  /// returns the first non-hidden source offset within that paragraph.
  /// Otherwise returns `baseIndex` unchanged.
  ///
  /// The paragraph lookup is on-demand (queries the content storage) so the
  /// result is correct regardless of viewport scroll state. An earlier
  /// approach cached prefix lengths as paragraphs were built; that cache
  /// went stale when scrolling revealed paragraphs not present at the last
  /// `apply()`.
  func translateHitTestIndex(_ baseIndex: Int) -> Int {
    guard let storage = textContentStorage,
      let delegate = storage.delegate as? TextKit2ContentStorageDelegate,
      let baseLocation = storage.location(
        storage.documentRange.location, offsetBy: baseIndex)
    else { return baseIndex }

    // Find the paragraph containing baseLocation by enumerating forward
    // from that location — TK2's enumerator visits the containing element
    // first.
    var paragraphRange: NSRange?
    storage.enumerateTextElements(from: baseLocation, options: []) { element in
      if let elemRange = element.elementRange {
        let start = storage.offset(
          from: storage.documentRange.location, to: elemRange.location)
        let length = storage.offset(from: elemRange.location, to: elemRange.endLocation)
        paragraphRange = NSRange(location: start, length: length)
      }
      return false  // first hit only
    }

    // Only translate when the click resolved to the paragraph's start —
    // mid-paragraph clicks already land on visible content.
    guard let paragraphRange, paragraphRange.location == baseIndex else {
      return baseIndex
    }

    return baseIndex + delegate.computeHiddenPrefix(forParagraphSourceRange: paragraphRange)
  }
}
