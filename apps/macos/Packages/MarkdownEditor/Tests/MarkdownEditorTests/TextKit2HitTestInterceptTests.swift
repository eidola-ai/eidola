import AppKit
import ObjectiveC
import Testing

@testable import MarkdownEditor

/// Phase 2.5 of the TextKit 2 migration — validates that the hit-test
/// intercept (`TextKit2MarkdownTextView.characterIndexForInsertion(at:)`)
/// is correctly installed in the production setup and that its translation
/// logic produces the right source offset for clicks at the visual start of
/// a stripped paragraph.
///
/// The Phase 0 spike showed that real click hit-testing is unreliable in
/// pure XCTest (lazy viewport layout). To work around this, the override is
/// split into the AppKit method (which calls super) and a pure
/// `translateHitTestIndex(_:)` helper that we test directly.
@MainActor
struct TextKit2HitTestInterceptTests {

  /// Holds a strong reference to the delegate (NSTextContentStorage stores
  /// its delegate weakly, mirroring why the production Coordinator owns it).
  private struct ProductionStyleSetup {
    let textView: TextKit2MarkdownTextView
    let delegate: TextKit2ContentStorageDelegate
  }

  /// Mirror MarkdownEditor.makeNSView's TK2 setup, including the
  /// `object_setClass` upgrade to TextKit2MarkdownTextView.
  private static func makeProductionStyleSetup() -> ProductionStyleSetup {
    let tv = NSTextView(usingTextLayoutManager: true)
    object_setClass(tv, TextKit2MarkdownTextView.self)
    let delegate = TextKit2ContentStorageDelegate()
    tv.textContentStorage?.delegate = delegate
    return ProductionStyleSetup(
      textView: tv as! TextKit2MarkdownTextView, delegate: delegate)
  }

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

  @Test
  func translateHitTestIndex_passes_through_when_no_prefix() {
    let setup = Self.makeProductionStyleSetup()
    // Delegate has no paragraphs registered → prefix is 0 everywhere.
    #expect(setup.textView.translateHitTestIndex(0) == 0)
    #expect(setup.textView.translateHitTestIndex(5) == 5)
    #expect(setup.textView.translateHitTestIndex(42) == 42)
  }

  @Test
  func translateHitTestIndex_adds_hidden_prefix_at_paragraph_start() throws {
    let setup = Self.makeProductionStyleSetup()
    let tv = setup.textView

    // Drive a paragraph build so the delegate populates its prefix map.
    // `# Heading\n\nbody`: heading at source 0 has hidden `# ` (length 2);
    // body at source 11 has no hidden prefix.
    let markdown = "# Heading\n\nbody"
    tv.string = markdown
    let bodyOffset = ("# Heading\n\n" as NSString).length
    let cursorRange = NSRange(location: bodyOffset, length: 0)
    let spec = MarkdownRenderer.render(
      text: markdown, cursorRange: cursorRange, style: .default)
    TextKit2RenderApplicator.apply(spec, to: tv)
    if let tlm = tv.textLayoutManager {
      tlm.ensureLayout(for: tlm.documentRange)
    }

    // TK2's hit-test reports display offsets in source-coordinate clothing
    // (empirically verified — the layout fragment lays out the display
    // string and TK2 maps display offset N to paragraph.elementRange.location
    // + N, treating the offset as a source position). Translation walks the
    // display→source map for the containing paragraph.
    //
    // Heading paragraph display = "Heading\n" (length 8); the paragraph's
    // source range is [0, 11) covering "# Heading\n". hidden = {0, 1}.
    // displayToSourceMap = [2,3,4,5,6,7,8,9,10].
    //
    // Click reported as source 0 → display offset 0 → source 2 (visible 'H').
    #expect(tv.translateHitTestIndex(0) == 2)
    // Click reported as source 5 → display offset 5 → source 7 (visible 'i').
    #expect(tv.translateHitTestIndex(5) == 7)
    // Click on body paragraph (no hidden chars) — display map is identity,
    // so the offset within the body paragraph passes through unchanged.
    #expect(tv.translateHitTestIndex(bodyOffset) == bodyOffset)
  }

  @Test
  func translateHitTestIndex_works_for_paragraphs_the_delegate_has_not_built() throws {
    // Regression for the scroll bug: an earlier implementation cached
    // hidden-prefix lengths in a map populated as the delegate built each
    // paragraph. Paragraphs that hadn't been built (e.g. off-screen until
    // the user scrolled) had no map entry, so the intercept silently
    // returned baseIndex unchanged and the cursor landed on the hidden
    // prefix. The fix is to compute prefix on-demand from the delegate's
    // index sets at hit-test time, regardless of whether the delegate has
    // been called for that paragraph.
    let setup = Self.makeProductionStyleSetup()
    let tv = setup.textView

    // Long document with headings far enough apart that, in a real app,
    // some would be off-screen at any moment.
    var lines: [String] = []
    for i in 0..<20 {
      lines.append("# Heading \(i)")
      lines.append("")
      lines.append("body \(i)")
      lines.append("")
    }
    let markdown = lines.joined(separator: "\n")
    tv.string = markdown

    // Apply spec with cursor at end (outside every heading) so all heading
    // delimiters are in hiddenIndexes.
    let cursorRange = NSRange(location: (markdown as NSString).length, length: 0)
    let spec = MarkdownRenderer.render(
      text: markdown, cursorRange: cursorRange, style: .default)
    TextKit2RenderApplicator.apply(spec, to: tv)

    // For each heading paragraph, the prefix should compute correctly even
    // if TK2 hasn't yet laid it out. We don't call ensureLayout here — that
    // simulates the post-scroll state where some paragraphs are off-screen
    // and unbuilt.
    var headingStart = 0
    for i in 0..<20 {
      let offset = tv.translateHitTestIndex(headingStart)
      #expect(
        offset == headingStart + 2,
        "heading #\(i) at source \(headingStart): expected \(headingStart + 2), got \(offset)")
      headingStart += ("# Heading \(i)\n\nbody \(i)\n\n" as NSString).length
    }
  }

  @Test
  func intercept_is_safe_when_no_delegate_installed() {
    // Defensive: the override must not crash if textContentStorage has no
    // delegate (e.g. on a TK1-backed text view that was upgraded by mistake).
    let tv = NSTextView(usingTextLayoutManager: true)
    object_setClass(tv, TextKit2MarkdownTextView.self)
    let upgraded = tv as! TextKit2MarkdownTextView
    #expect(upgraded.translateHitTestIndex(7) == 7)
  }
}
