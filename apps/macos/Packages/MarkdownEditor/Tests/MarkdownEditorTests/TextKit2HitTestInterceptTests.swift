import AppKit
import ObjectiveC
import Testing

@testable import MarkdownEditor

/// Hit-test plumbing tests for the TextKit 2 path.
///
/// Originally these tests covered the `translateHitTestIndex(_:)` override
/// that compensated for TK2 returning display offsets when the displayed
/// paragraph was shorter than its source range. With the length-matching
/// invariant (every vended paragraph has `display.length == source.length`
/// via ZWSP / glyph substitution), TK2's hit-test returns real source
/// offsets directly and no override is needed.
///
/// These tests now pin the structural setup (subclass upgrade) and the
/// safety property (no crash without a content delegate). The real
/// click-lands-on-the-right-glyph contract is exercised by
/// `SelectionNavigationTests.click_on_C_in_bolded_word_lands_on_C` and
/// peers, which drive the full TK2 stack against real source text.
@MainActor
struct TextKit2HitTestInterceptTests {

  @Test
  func object_setClass_upgrades_textview_to_subclass() {
    let tv = NSTextView(usingTextLayoutManager: true)
    #expect(
      !tv.isKind(of: TextKit2MarkdownTextView.self),
      "convenience init should produce plain NSTextView before upgrade")
    object_setClass(tv, TextKit2MarkdownTextView.self)
    #expect(
      tv.isKind(of: TextKit2MarkdownTextView.self),
      "after object_setClass, the runtime class should be TextKit2MarkdownTextView")
  }

  /// Defensive: the subclass must not crash if textContentStorage has no
  /// delegate (e.g. on a TK1-backed text view that was upgraded by mistake).
  /// This was the contract the old `translateHitTestIndex` had to honor;
  /// with no override at all, hit-test simply falls through to super and
  /// can't crash on a missing delegate. Test kept as a smoke check.
  @Test
  func subclass_is_safe_when_no_delegate_installed() {
    let tv = NSTextView(usingTextLayoutManager: true)
    object_setClass(tv, TextKit2MarkdownTextView.self)
    let upgraded = tv as! TextKit2MarkdownTextView
    let raw = upgraded.characterIndexForInsertion(at: NSPoint(x: 0, y: 0))
    // No content → cursor lands at 0; the property we care about is "no crash".
    #expect(raw >= 0)
  }
}
