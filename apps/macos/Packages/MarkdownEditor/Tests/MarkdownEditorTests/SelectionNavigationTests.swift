import AppKit
import Foundation
import ObjectiveC
import Testing

@testable import MarkdownEditor

/// Selection / navigation across constructs whose source ranges contain
/// hidden delimiters or substituted glyphs (bullets, checkboxes). The
/// content delegate now preserves the length-matching invariant
/// (`display.length == source.length`), so TK2's hit-test and selection
/// navigation operate on real source offsets — the contract these tests
/// pin is purely about cursor-stops-on-visible behavior.
///
/// 1. Display strings reflect the substitutions (ZWSP for hidden, • for
///    bullet, ☐ / ☒ for checkboxes) and have the same length as their
///    source range.
/// 2. Arrow keys traverse hidden runs as a single atomic step (the cursor
///    never lands on a hidden char).
/// 3. Shift+arrow extends one source step at a time across the same
///    hidden runs — bidirectional, no lost characters.
/// 4. The soft-break cursor walk continues to work.
@Suite("Selection Navigation")
@MainActor
struct SelectionNavigationTests {

  // MARK: - Test rig (mirrors TextKit2HitTestInterceptTests)

  private struct Rig {
    let textView: TextKit2MarkdownTextView
    let storage: NSTextContentStorage
    let layout: NSTextLayoutManager
    let delegate: TextKit2ContentStorageDelegate
    let layoutDelegate: TextKit2LayoutManagerDelegate
    let window: NSWindow
  }

  private static let outDir: String = {
    let thisFile = #filePath
    let testsDir = (thisFile as NSString).deletingLastPathComponent
    let testRoot = (testsDir as NSString).deletingLastPathComponent
    let packageRoot = (testRoot as NSString).deletingLastPathComponent
    let dir = (packageRoot as NSString)
      .appendingPathComponent("test-artifacts/selection-navigation")
    try? FileManager.default.createDirectory(
      atPath: dir, withIntermediateDirectories: true)
    return dir
  }()

  private static func make(
    markdown: String, cursorPosition: Int,
    size: NSSize = NSSize(width: 600, height: 400)
  ) -> Rig {
    let tv = NSTextView(usingTextLayoutManager: true)
    object_setClass(tv, TextKit2MarkdownTextView.self)
    let textView = tv as! TextKit2MarkdownTextView
    textView.frame = NSRect(origin: .zero, size: size)
    textView.minSize = NSSize(width: 0, height: 0)
    textView.maxSize = NSSize(
      width: CGFloat.greatestFiniteMagnitude,
      height: CGFloat.greatestFiniteMagnitude)
    textView.font = MarkdownStyle.default.baseFont
    textView.isRichText = true
    textView.isAutomaticQuoteSubstitutionEnabled = false
    textView.textContainer?.containerSize = NSSize(
      width: size.width, height: CGFloat.greatestFiniteMagnitude)
    textView.textContainer?.widthTracksTextView = true

    let delegate = TextKit2ContentStorageDelegate()
    textView.textContentStorage?.delegate = delegate
    let layoutDelegate = TextKit2LayoutManagerDelegate()
    textView.textLayoutManager?.delegate = layoutDelegate

    let window = NSWindow(
      contentRect: NSRect(origin: .zero, size: size),
      styleMask: .borderless, backing: .buffered, defer: true)
    window.contentView = textView

    textView.string = markdown
    let cursorRange = NSRange(location: cursorPosition, length: 0)
    textView.setSelectedRange(cursorRange)
    let spec = MarkdownRenderer.render(
      text: markdown, cursorRange: cursorRange, style: .default)
    TextKit2RenderApplicator.apply(spec, to: textView)
    if let tlm = textView.textLayoutManager {
      tlm.ensureLayout(for: tlm.documentRange)
    }
    return Rig(
      textView: textView,
      storage: textView.textContentStorage!,
      layout: textView.textLayoutManager!,
      delegate: delegate,
      layoutDelegate: layoutDelegate,
      window: window)
  }

  /// Re-render the rig at the current cursor so hidden / revealed
  /// delimiters reflect the cursor position. Mirrors what the production
  /// Coordinator does on every selection change.
  private static func rerender(_ rig: Rig, markdown: String) {
    let cursorRange = rig.textView.selectedRange()
    let spec = MarkdownRenderer.render(
      text: markdown, cursorRange: cursorRange, style: .default)
    TextKit2RenderApplicator.apply(spec, to: rig.textView)
    rig.layout.ensureLayout(for: rig.layout.documentRange)
  }

  /// Press right-arrow once via the production action selector. Re-renders
  /// the spec at the new cursor before returning so the next press sees
  /// the updated `hiddenIndexes`.
  private static func pressRight(_ rig: Rig, markdown: String) -> Int {
    rig.textView.moveRight(nil)
    Self.rerender(rig, markdown: markdown)
    return rig.textView.selectedRange().location
  }

  private static func pressLeft(_ rig: Rig, markdown: String) -> Int {
    rig.textView.moveLeft(nil)
    Self.rerender(rig, markdown: markdown)
    return rig.textView.selectedRange().location
  }

  private static func pressShiftRight(_ rig: Rig, markdown: String) -> NSRange {
    rig.textView.moveRightAndModifySelection(nil)
    Self.rerender(rig, markdown: markdown)
    return rig.textView.selectedRange()
  }

  private static func pressShiftLeft(_ rig: Rig, markdown: String) -> NSRange {
    rig.textView.moveLeftAndModifySelection(nil)
    Self.rerender(rig, markdown: markdown)
    return rig.textView.selectedRange()
  }

  /// Snapshot the current text view to disk for visual review.
  private static func snapshot(_ rig: Rig, name: String, size: NSSize) {
    let textView = rig.textView
    rig.layout.ensureLayout(for: rig.layout.documentRange)
    textView.needsDisplay = true
    textView.displayIfNeeded()
    let bitmap = NSBitmapImageRep(
      bitmapDataPlanes: nil,
      pixelsWide: Int(size.width),
      pixelsHigh: Int(size.height),
      bitsPerSample: 8,
      samplesPerPixel: 4,
      hasAlpha: true,
      isPlanar: false,
      colorSpaceName: .calibratedRGB,
      bytesPerRow: 0,
      bitsPerPixel: 0)!
    textView.cacheDisplay(in: textView.bounds, to: bitmap)
    if let data = bitmap.representation(using: .png, properties: [:]) {
      let url = URL(fileURLWithPath: "\(outDir)/\(name).png")
      try? data.write(to: url)
    }
  }

  // MARK: - 1. Bug 1: visible glyph display offsets equal source offsets

  /// Pre-refactor: the content delegate vended a *shorter* display string
  /// (hidden chars removed), so TK2's hit-test returned a display offset
  /// that didn't equal the source position. The override
  /// `translateHitTestIndex` walked a per-paragraph display→source map.
  ///
  /// Post-refactor: the delegate vends a length-matching display string
  /// (hidden chars become zero-width `U+200B` so they take up a UTF-16 unit
  /// without taking visual space). The display offset of every visible
  /// glyph equals its source offset, so TK2's hit-test returns the right
  /// source position directly — no translation override needed.
  ///
  /// This test pins the invariant: in `"A **B** C D E F G"` with cursor
  /// outside the bold, the displayed paragraph has the same length as the
  /// source and its visible glyphs land at their source offsets. The actual
  /// click→glyph mapping is exercised end-to-end by the runner; here we
  /// just verify the structural property the hit-test relies on.
  @Test
  func display_paragraph_length_matches_source_length_with_hidden_bold() {
    let md = "A **B** C D E F G"
    let rig = Self.make(markdown: md, cursorPosition: (md as NSString).length)
    Self.snapshot(rig, name: "01-bold-hidden", size: NSSize(width: 600, height: 100))

    // Walk the laid-out paragraphs and verify every one has display.length
    // == source.length.
    let cs = rig.storage
    rig.layout.enumerateTextLayoutFragments(
      from: rig.layout.documentRange.location, options: []
    ) { frag in
      guard let element = frag.textElement,
        let paragraph = element as? NSTextParagraph,
        let elemRange = element.elementRange
      else { return true }
      let length = cs.offset(from: elemRange.location, to: elemRange.endLocation)
      let displayLen = (paragraph.attributedString.string as NSString).length
      #expect(
        displayLen == length,
        "display length \(displayLen) must equal source length \(length)")
      return true
    }
  }

  // MARK: - 2. Bug 2: arrow keys walk one source-visible char per press

  @Test
  func right_arrow_walks_through_C_D_E_after_leaving_bold() {
    // Start at source 5 (after "B", cursor inside bold so `**` visible).
    // Each right-arrow should advance one source position. Crossing out of
    // the bold (source 7 → 8) re-renders to hide `**`, but the next press
    // must continue with C (8), D (10), E (12), not jump past them.
    let md = "A **B** C D E F G"
    let rig = Self.make(markdown: md, cursorPosition: 5)

    var visited = [5]
    for _ in 0..<6 {
      visited.append(Self.pressRight(rig, markdown: md))
    }
    // Expected walk: 5 → 6 → 7 → 8 → 9 → 10 → 11 (C, ' ', D, ' ' along the way).
    #expect(visited == [5, 6, 7, 8, 9, 10, 11], "got: \(visited)")
  }

  @Test
  func right_arrow_skips_hidden_asterisks_when_outside_construct() {
    // Cursor at source 1 (' ' after 'A'), outside bold. Asterisks at 2,3
    // and 5,6 are hidden. Right-arrow must skip the hidden run wholesale —
    // not stop on a hidden char, not jump past 'B'.
    let md = "A **B** C D E F G"
    let rig = Self.make(markdown: md, cursorPosition: 1)

    // First right-arrow: skips hidden 2,3 → lands on visible 4 ('B').
    let firstStop = Self.pressRight(rig, markdown: md)
    #expect(firstStop == 4, "expected to skip past **, land on B at 4; got \(firstStop)")

    // Now cursor is at 4, INSIDE the bold construct → asterisks revealed
    // by the rerender. Next right-arrow: source 5 is now a visible `*`,
    // so we land on it.
    let secondStop = Self.pressRight(rig, markdown: md)
    #expect(secondStop == 5, "expected to land on revealed `*` at 5; got \(secondStop)")
  }

  // MARK: - 3. Bug 3: Shift+arrow extends one source char at a time

  @Test
  func shift_right_arrow_extends_selection_one_source_char_per_press_across_bold() {
    let md = "A **B** C D E F G"
    let rig = Self.make(markdown: md, cursorPosition: 1)

    // Each shift+right should grow the selection by exactly one source
    // step. Crossing the hidden run is one step (no individual hidden
    // chars are landed on).
    let r1 = Self.pressShiftRight(rig, markdown: md)
    // From cursor at 1 with no selection, shift+right should select [1,4)
    // (anchor at 1, head past hidden run to 4).
    #expect(
      r1.location == 1 && r1.length == 3,
      "expected [1, 4); got [\(r1.location), \(r1.location + r1.length))")

    let r2 = Self.pressShiftRight(rig, markdown: md)
    #expect(
      r2.location == 1 && r2.location + r2.length == 5,
      "expected [1, 5); got [\(r2.location), \(r2.location + r2.length))")

    let r3 = Self.pressShiftRight(rig, markdown: md)
    #expect(
      r3.location == 1 && r3.location + r3.length == 6,
      "expected [1, 6); got [\(r3.location), \(r3.location + r3.length))")
  }

  @Test
  func shift_left_arrow_shrinks_then_extends_across_bold_symmetrically() {
    let md = "A **B** C D E F G"
    let rig = Self.make(markdown: md, cursorPosition: 8)  // cursor on 'C'
    // shift+left from cursor at 8 should select backward across hidden
    // delimiters: [7, 8). Then [4, 8). Then [3, 8) (revealed * inside).
    let r1 = Self.pressShiftLeft(rig, markdown: md)
    #expect(
      r1.location == 7 && r1.length == 1,
      "expected [7, 8); got [\(r1.location), \(r1.location + r1.length))")

    let r2 = Self.pressShiftLeft(rig, markdown: md)
    // The shift to head=4 means the entire bold block is now selected;
    // the cursor anchor at 8 is past the bold, so it stays outside —
    // **may** be hidden. Selection covers [4, 8) source.
    #expect(
      r2.location == 4 && r2.location + r2.length == 8,
      "expected [4, 8); got [\(r2.location), \(r2.location + r2.length))")
  }

  // MARK: - 4. Heading paragraph: length-matching invariant + visible H

  /// `# Heading\n\nbody` — heading paragraph has hidden `# ` (positions 0, 1).
  /// Post-refactor those hidden chars become two ZWSPs in the display
  /// string, so the displayed paragraph reads `\u{200B}\u{200B}Heading\n`
  /// (length 12, equal to source length). The visible 'H' sits at display
  /// offset 2 == source offset 2, so a real click on H lands on source 2
  /// without any translation override.
  @Test
  func heading_display_starts_with_two_zwsps_then_visible_text() {
    let md = "# Heading\n\nbody"
    let bodyOffset = ("# Heading\n\n" as NSString).length
    let rig = Self.make(markdown: md, cursorPosition: bodyOffset)

    // Find the heading paragraph and inspect its display string.
    let cs = rig.storage
    var headingDisplay: String?
    rig.layout.enumerateTextLayoutFragments(
      from: rig.layout.documentRange.location, options: []
    ) { frag in
      guard let element = frag.textElement,
        let paragraph = element as? NSTextParagraph,
        let elemRange = element.elementRange
      else { return true }
      let start = cs.offset(from: cs.documentRange.location, to: elemRange.location)
      if start == 0 {
        headingDisplay = paragraph.attributedString.string
        return false
      }
      return true
    }
    #expect(
      headingDisplay == "\u{200B}\u{200B}Heading\n",
      "got: \(String(describing: headingDisplay))")
  }

  // MARK: - 5. Soft break walk (regression pin)

  @Test
  func cursor_can_walk_across_soft_break_with_arrow_keys_via_subclass() {
    // Pin the existing soft-break navigation behavior with the new
    // override path. Source: "Hello\nworld" — the `\n` at offset 5 is a
    // soft break so it stays as a real `\n` in the source but the
    // rendered paragraphs are flush.
    let md = "Hello\nworld"
    let rig = Self.make(markdown: md, cursorPosition: 5)

    var visited = [5]
    for _ in 0..<3 {
      visited.append(Self.pressRight(rig, markdown: md))
    }
    // Expected monotonic walk: 5 → 6 ('w') → 7 ('o') → 8 ('r').
    #expect(visited == [5, 6, 7, 8], "got: \(visited)")
  }

  // MARK: - 6. List-item bullet substitution navigation

  @Test
  func list_item_displays_bullet_then_visible_text() {
    // `- item\n\nbody`. Place the cursor in the body paragraph so the
    // list item's `-` is in bullet-substitution mode. Source 0 (`-`)
    // becomes `•` (1-for-1); source 1 (` `) passes through as a real
    // space (the renderer only hides marker chars *between* the marker
    // and the first content char, so a single trailing space stays
    // visible). Display string is `• item\n` (length 7 == source length 7).
    let md = "- item\n\nbody"
    let bodyOffset = ("- item\n\n" as NSString).length
    let rig = Self.make(markdown: md, cursorPosition: bodyOffset)

    let cs = rig.storage
    var firstParagraphDisplay: String?
    rig.layout.enumerateTextLayoutFragments(
      from: rig.layout.documentRange.location, options: []
    ) { frag in
      guard let element = frag.textElement,
        let paragraph = element as? NSTextParagraph,
        let elemRange = element.elementRange
      else { return true }
      let start = cs.offset(from: cs.documentRange.location, to: elemRange.location)
      if start == 0 {
        firstParagraphDisplay = paragraph.attributedString.string
        return false
      }
      return true
    }
    #expect(
      firstParagraphDisplay == "\u{2022} item\n",
      "got: \(String(describing: firstParagraphDisplay))")
    // Length-matching invariant: display length == source length.
    let displayLen = ((firstParagraphDisplay ?? "") as NSString).length
    let sourceLen = ("- item\n" as NSString).length
    #expect(displayLen == sourceLen, "display \(displayLen) vs source \(sourceLen)")
  }

  @Test
  func right_arrow_within_revealed_list_marker_walks_one_char_at_a_time() {
    // From cursor at source 0 in `- item\n` — cursor is INSIDE the list
    // item, so `- ` is revealed. Right-arrow advances one source char at
    // a time across the visible delimiters: 0 → 1 → 2 → 3 → ...
    let md = "- item\n\nbody"
    let rig = Self.make(markdown: md, cursorPosition: 0)
    var visited = [0]
    for _ in 0..<3 {
      visited.append(Self.pressRight(rig, markdown: md))
    }
    #expect(
      visited == [0, 1, 2, 3],
      "expected one-char-per-press across revealed `- `; got: \(visited)")
  }

  // MARK: - 7. Heading: arrow keys cross the construct boundary cleanly

  @Test
  func left_arrow_from_H_in_heading_skips_revealed_prefix_when_outside() {
    // From cursor at source 2 ('H'), pressing left-arrow once should land
    // on source 1 (' '), revealing the `# ` prefix because the cursor is
    // now inside the heading. A second left-arrow lands on source 0
    // (the `#`). A third lands at source 0 - 1 = paragraph start; since
    // we're at the document start, no movement happens.
    //
    // This pins that left-arrow does NOT collapse the `# ` into a single
    // step when the cursor is inside the heading (the prefix is visible).
    let md = "# Heading\n\nbody"
    let rig = Self.make(markdown: md, cursorPosition: 2)
    let p1 = Self.pressLeft(rig, markdown: md)
    #expect(p1 == 1, "expected left-arrow to land on ' ' at 1; got \(p1)")
    let p2 = Self.pressLeft(rig, markdown: md)
    #expect(p2 == 0, "expected left-arrow to land on '#' at 0; got \(p2)")
  }

  @Test
  func right_arrow_from_after_body_into_next_heading_skips_hidden_prefix() {
    // Source: "body\n\n# Heading\nmore"
    // Positions: b=0 o=1 d=2 y=3 \n=4 \n=5 #=6 ' '=7 H=8 e=9 ...
    //
    // Cursor at source 4 (just after 'y', in the body paragraph). At this
    // cursor position the heading is OUTSIDE so `# ` (positions 6,7) are
    // hidden. The absorbed `\n` at position 5 is also hidden by the
    // renderer's inter-block gap absorption (every two `\n`s = one
    // paragraph break, the second is left as a hidden orphan).
    //
    // Right-arrow strides past the contiguous hidden run {5,6,7} and
    // lands on the next visible source position — 'H' at 8. The cursor
    // never stops on a hidden char.
    let md = "body\n\n# Heading\nmore"
    let rig = Self.make(markdown: md, cursorPosition: 4)
    let p1 = Self.pressRight(rig, markdown: md)
    #expect(
      p1 == 8,
      "right-arrow from 4 should skip hidden run and land on 'H' at 8; got \(p1)")

    // After landing at 8, cursor is inside the heading → `# ` is
    // REVEALED on the next render. Subsequent right-arrows advance one
    // source char at a time.
    let p2 = Self.pressRight(rig, markdown: md)
    #expect(
      p2 == 9,
      "right-arrow from 8 should advance to 9 ('e'); got \(p2)")
  }
}
