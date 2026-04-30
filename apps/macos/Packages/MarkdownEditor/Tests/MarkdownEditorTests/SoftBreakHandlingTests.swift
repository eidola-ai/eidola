import AppKit
import Foundation
import Testing

@testable import MarkdownEditor

/// Soft-break vs paragraph-break handling for the MarkdownEditor.
///
/// Discovery (`NewlineDiscoveryTests` in this directory, now removed)
/// established that `MarkdownParser.collectInlineNodes` discards swift-markdown
/// `SoftBreak` / `LineBreak` inline nodes. Because of that, every literal `\n`
/// inside a single AST `Paragraph` (a soft break) is treated by the renderer
/// as if it were a paragraph boundary: TextKit splits the paragraph there,
/// each half receives its own `NSTextParagraph`, and the user sees an
/// ungrammatical visual gap where the source intends a wrap.
///
/// The intended fix substitutes a single `\n` between two soft-break halves
/// with `U+2028 LINE SEPARATOR` in the *display string* (source remains
/// verbatim). TextKit treats `U+2028` as an in-paragraph line break, so the
/// halves render as one displayed paragraph, the cursor walks across it
/// naturally, and clipboard / source coordinates are unaffected.
///
/// These tests assert the desired post-fix behavior. Most assert on the
/// *displayed paragraph strings* (mirroring `TextKit2GlyphHidingTests`)
/// because that is what the user actually sees and is robust to how the fix
/// is plumbed internally. A handful of edit-flow regression tests are also
/// included so the bug fix doesn't disturb already-correct behavior.
///
/// Snapshots are written to `test-artifacts/soft-break-handling/` for visual
/// review during the implementation phase.
@Suite("Soft Break Handling")
@MainActor
struct SoftBreakHandlingTests {

  /// `U+2028 LINE SEPARATOR`. The implementation does NOT substitute soft
  /// breaks with this character (an earlier coalescing experiment did, but
  /// it broke TK2 cursor navigation through absorbed elements). Kept here
  /// only so "should not contain LS" assertions stay readable.
  private static let LS = "\u{2028}"

  /// Helper: walk `spec.styledRanges` and return the effective paragraph
  /// style at a given source offset. Later styled ranges override earlier
  /// ones (the renderer applies them in order).
  private static func paragraphStyle(
    forSource offset: Int, in spec: RenderSpec
  ) -> NSParagraphStyle? {
    var found: NSParagraphStyle?
    for sr in spec.styledRanges
    where NSLocationInRange(offset, sr.range) {
      if let ps = sr.attributes[.paragraphStyle] as? NSParagraphStyle {
        found = ps
      }
    }
    return found
  }

  /// Run the renderer against the given markdown + cursor and return the
  /// `RenderSpec`. Use this when a test wants to assert on the spec
  /// directly (e.g., paragraph-style spacing) rather than on the laid-out
  /// display strings.
  private static func renderedSpec(
    markdown: String, cursorPosition: Int, style: MarkdownStyle = .default
  ) -> RenderSpec {
    let cursorRange = NSRange(location: cursorPosition, length: 0)
    return MarkdownRenderer.render(
      text: markdown, cursorRange: cursorRange, style: style)
  }

  // MARK: - Output directory

  private static let outDir: String = {
    let thisFile = #filePath
    let testsDir = (thisFile as NSString).deletingLastPathComponent  // MarkdownEditorTests/
    let testRoot = (testsDir as NSString).deletingLastPathComponent  // Tests/
    let packageRoot = (testRoot as NSString).deletingLastPathComponent  // MarkdownEditor/
    let dir = (packageRoot as NSString).appendingPathComponent("test-artifacts/soft-break-handling")
    try? FileManager.default.createDirectory(
      atPath: dir, withIntermediateDirectories: true)
    return dir
  }()

  // MARK: - TK2 setup mirrored from TextKit2GlyphHidingTests

  private struct TK2Components {
    let textView: NSTextView
    let contentStorage: NSTextContentStorage
    let contentDelegate: TextKit2ContentStorageDelegate
    let layoutDelegate: TextKit2LayoutManagerDelegate
    let window: NSWindow
  }

  private static func makeComponents(
    size: NSSize = NSSize(width: 600, height: 400)
  ) -> TK2Components {
    let textView = NSTextView(usingTextLayoutManager: true)
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

    let contentDelegate = TextKit2ContentStorageDelegate()
    textView.textContentStorage?.delegate = contentDelegate
    let layoutDelegate = TextKit2LayoutManagerDelegate()
    textView.textLayoutManager?.delegate = layoutDelegate

    let window = NSWindow(
      contentRect: NSRect(origin: .zero, size: size),
      styleMask: .borderless, backing: .buffered, defer: true)
    window.contentView = textView

    return TK2Components(
      textView: textView,
      contentStorage: textView.textContentStorage!,
      contentDelegate: contentDelegate,
      layoutDelegate: layoutDelegate,
      window: window)
  }

  /// Drive the renderer + applicator end-to-end and return the laid-out
  /// paragraphs' display strings, keyed by source-range start offset. The
  /// keys come from the source-range location of each `NSTextElement`, so
  /// they correspond to offsets in the original markdown — making it easy to
  /// reason about which paragraph is which.
  private static func renderAndCollectDisplay(
    markdown: String, cursorPosition: Int = 0,
    style: MarkdownStyle = .default,
    components: TK2Components
  ) -> [Int: String] {
    let textView = components.textView
    textView.string = markdown
    let cursorRange = NSRange(location: cursorPosition, length: 0)
    textView.setSelectedRange(cursorRange)

    let spec = MarkdownRenderer.render(
      text: markdown, cursorRange: cursorRange, style: style)
    TextKit2RenderApplicator.apply(spec, to: textView)

    if let tlm = textView.textLayoutManager {
      tlm.ensureLayout(for: tlm.documentRange)
    }

    var byStart: [Int: String] = [:]
    let cs = components.contentStorage
    if let tlm = textView.textLayoutManager {
      tlm.enumerateTextLayoutFragments(
        from: tlm.documentRange.location, options: []
      ) { frag in
        guard let element = frag.textElement,
          let paragraph = element as? NSTextParagraph
        else { return true }
        let elemRange = element.elementRange ?? frag.rangeInElement
        let start = cs.offset(from: cs.documentRange.location, to: elemRange.location)
        byStart[start] = paragraph.attributedString.string
        return true
      }
    }
    return byStart
  }

  /// Drive the renderer + applicator end-to-end and return the custom
  /// `TextKit2LayoutFragment` instances keyed by source-range start offset.
  /// Used to assert blockquote border state for the lazy-continuation case.
  private static func renderAndCollectFragments(
    markdown: String, cursorPosition: Int = 0,
    style: MarkdownStyle = .default,
    components: TK2Components
  ) -> [Int: TextKit2LayoutFragment] {
    let textView = components.textView
    textView.string = markdown
    let cursorRange = NSRange(location: cursorPosition, length: 0)
    textView.setSelectedRange(cursorRange)

    let spec = MarkdownRenderer.render(
      text: markdown, cursorRange: cursorRange, style: style)
    TextKit2RenderApplicator.apply(spec, to: textView)

    guard let tlm = textView.textLayoutManager,
      let cs = textView.textContentStorage
    else { return [:] }
    tlm.ensureLayout(for: tlm.documentRange)

    var byStart: [Int: TextKit2LayoutFragment] = [:]
    tlm.enumerateTextLayoutFragments(
      from: tlm.documentRange.location, options: []
    ) { frag in
      let elemRange = frag.textElement?.elementRange ?? frag.rangeInElement
      let start = cs.offset(from: cs.documentRange.location, to: elemRange.location)
      if let custom = frag as? TextKit2LayoutFragment {
        byStart[start] = custom
      }
      return true
    }
    return byStart
  }

  /// Capture a bitmap snapshot of the laid-out text view to
  /// `test-artifacts/soft-break-handling/<name>.png`. Mirrors
  /// `TextKit2FragmentDecorationTests.tk2_snapshots_for_visual_review`.
  /// Always succeeds — the snapshot is for human review, not assertion.
  private static func writeSnapshot(
    _ components: TK2Components, name: String,
    size: NSSize
  ) {
    let textView = components.textView
    if let tlm = textView.textLayoutManager {
      tlm.ensureLayout(for: tlm.documentRange)
    }
    // Deselect so the cursor caret doesn't appear in the snapshot.
    let savedSelection = textView.selectedRange()
    textView.setSelectedRange(NSRange(location: 0, length: 0))
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
    textView.setSelectedRange(savedSelection)
  }

  // MARK: - 1. Single newline between body lines → in-paragraph line break

  @Test
  func single_newline_between_body_lines_renders_as_in_paragraph_line_break() {
    let c = Self.makeComponents()
    let markdown = "Hello\nworld"
    let cursorPosition = (markdown as NSString).length
    _ = Self.renderAndCollectDisplay(
      markdown: markdown, cursorPosition: cursorPosition, components: c)
    Self.writeSnapshot(c, name: "01-single-nl-body", size: NSSize(width: 600, height: 200))

    // The user-visible outcome is "Hello" and "world" on adjacent lines with
    // no paragraph gap. We achieve that by emitting two TK source paragraphs
    // (TK2 splits on `\n`) but with `paragraphSpacing = 0` between them, so
    // cursor navigation works naturally and the visual result matches a
    // soft-broken single paragraph.
    let spec = Self.renderedSpec(markdown: markdown, cursorPosition: cursorPosition)
    #expect(
      spec.lineBreakIndexes.contains(5),
      "parser should have classified the `\\n` at offset 5 as a soft break")

    // First segment ([0, 6)): trailing `paragraphSpacing` must be 0 so no gap
    // appears between it and the continuation.
    let firstStyle = Self.paragraphStyle(forSource: 0, in: spec)
    #expect(
      firstStyle?.paragraphSpacing == 0,
      "soft-break first half should have paragraphSpacing=0; got \(String(describing: firstStyle?.paragraphSpacing))"
    )
    // Second segment starts at offset 6 ("world"): leading
    // `paragraphSpacingBefore` must also be 0 for symmetric tightness.
    let secondStyle = Self.paragraphStyle(forSource: 6, in: spec)
    #expect(
      secondStyle?.paragraphSpacingBefore == 0,
      "soft-break second half should have paragraphSpacingBefore=0; got \(String(describing: secondStyle?.paragraphSpacingBefore))"
    )
  }

  // MARK: - 2. `\n\n` body → two paragraphs

  @Test
  func double_newline_between_body_lines_renders_as_paragraph_break() {
    let c = Self.makeComponents()
    let markdown = "para 1\n\npara 2"
    let display = Self.renderAndCollectDisplay(
      markdown: markdown, cursorPosition: (markdown as NSString).length,
      components: c)
    Self.writeSnapshot(c, name: "02-double-nl-body", size: NSSize(width: 600, height: 200))

    // The two content paragraphs land at their AST source offsets, with no
    // U+2028 substitution anywhere — this is a true paragraph break.
    #expect(display[0] == "para 1\n", "got: \(String(describing: display[0]))")
    #expect(display[8] == "para 2", "got: \(String(describing: display[8]))")
    for (start, str) in display {
      #expect(
        !str.contains(Self.LS),
        "paragraph at \(start) should contain no LINE SEPARATOR: \(str)")
    }
  }

  // MARK: - 3. `\n\n\n` body → one paragraph break + one preserved blank

  // Decision: AGENTS.md says "every two newlines = one paragraph break".
  // Three `\n` is one complete pair plus one orphan. The complete pair
  // produces the single paragraph break itself; the orphan does NOT
  // contribute an additional visible empty paragraph (that requires another
  // full pair). The user-observable result is therefore the same as case 2:
  // two content paragraphs, no extra visible blank line between them. We
  // assert only what the user can see: both content halves appear in
  // separate paragraphs and no soft-break character is introduced. The
  // exact handling of the orphan `\n` (collapse vs. preserve) is left to
  // the implementation phase.
  @Test
  func triple_newline_renders_as_one_paragraph_break_plus_one_collapsed_blank() {
    let c = Self.makeComponents()
    let markdown = "para 1\n\n\npara 2"
    let display = Self.renderAndCollectDisplay(
      markdown: markdown, cursorPosition: (markdown as NSString).length,
      components: c)
    Self.writeSnapshot(c, name: "03-triple-nl-body", size: NSSize(width: 600, height: 200))

    let nonEmpty = display.values.filter {
      !$0.trimmingCharacters(in: .newlines).isEmpty
    }
    #expect(
      nonEmpty.count == 2,
      "expected two non-empty content paragraphs, got: \(display)")
    let onlyEmpty = display.values.filter {
      $0.trimmingCharacters(in: .newlines).isEmpty
    }
    // One orphan `\n` does NOT produce a visible blank paragraph between
    // the two content halves (that requires two orphans = one full pair).
    // We tolerate the implementation either collapsing it or hiding it,
    // but it must not produce more than ONE empty/collapsed paragraph
    // element between para 1 and para 2.
    #expect(
      onlyEmpty.count <= 1,
      "orphan \\n must not introduce extra visible empty paragraph; got \(onlyEmpty.count) empty elements: \(display)")
    for (_, str) in display {
      #expect(
        !str.contains(Self.LS),
        "no soft-break char should appear: \(str)")
    }
  }

  // MARK: - 4. `\n\n\n\n` body → para-break + one preserved empty paragraph

  @Test
  func quad_newline_renders_as_paragraph_break_plus_one_empty_paragraph() {
    let c = Self.makeComponents()
    let markdown = "para 1\n\n\n\npara 2"
    let display = Self.renderAndCollectDisplay(
      markdown: markdown, cursorPosition: (markdown as NSString).length,
      components: c)
    Self.writeSnapshot(c, name: "04-quad-nl-body", size: NSSize(width: 600, height: 200))

    // Two pairs of `\n`: the first pair makes the paragraph break itself,
    // the second pair makes ONE preserved empty paragraph. So the user sees
    // para 1, a blank line, para 2 — three paragraphs total.
    let nonEmpty = display.values.filter {
      !$0.trimmingCharacters(in: .newlines).isEmpty
    }
    let onlyEmpty = display.values.filter {
      $0.trimmingCharacters(in: .newlines).isEmpty
    }
    #expect(
      nonEmpty.count == 2,
      "expected two non-empty content paragraphs, got: \(display)")
    #expect(
      onlyEmpty.count == 1,
      "expected exactly one preserved empty paragraph between content halves, got \(onlyEmpty.count): \(display)")
    for (_, str) in display {
      #expect(
        !str.contains(Self.LS),
        "no soft-break char should appear: \(str)")
    }
  }

  // MARK: - 5. Single newline inside a blockquote → in-paragraph line break

  @Test
  func single_newline_in_blockquote_renders_as_in_paragraph_line_break() {
    let c = Self.makeComponents()
    let markdown = "> line a\n> line b"
    let cursorPosition = (markdown as NSString).length
    _ = Self.renderAndCollectDisplay(
      markdown: markdown, cursorPosition: cursorPosition, components: c)
    Self.writeSnapshot(c, name: "05-single-nl-in-bq", size: NSSize(width: 600, height: 200))

    // The single AST paragraph inside the blockquote spans both source lines
    // with a soft break between them. After the content delegate hides the
    // `> ` prefixes the two halves visually wrap as one paragraph — assert
    // via paragraph-style spacing rather than display string merging.
    let spec = Self.renderedSpec(markdown: markdown, cursorPosition: cursorPosition)
    let nlOffset = ("> line a" as NSString).length
    #expect(
      spec.lineBreakIndexes.contains(nlOffset),
      "parser should have classified the in-blockquote `\\n` as a soft break")

    let firstStyle = Self.paragraphStyle(forSource: 0, in: spec)
    #expect(
      firstStyle?.paragraphSpacing == 0,
      "in-blockquote soft-break first half should have paragraphSpacing=0; got \(String(describing: firstStyle?.paragraphSpacing))"
    )
    let secondStyle = Self.paragraphStyle(forSource: nlOffset + 1, in: spec)
    #expect(
      secondStyle?.paragraphSpacingBefore == 0,
      "in-blockquote soft-break second half should have paragraphSpacingBefore=0; got \(String(describing: secondStyle?.paragraphSpacingBefore))"
    )
  }

  // MARK: - 6. `>\n` blank quote line → two paragraphs in the blockquote

  @Test
  func double_newline_in_blockquote_via_empty_quote_renders_as_two_paragraphs() {
    let c = Self.makeComponents()
    let markdown = "> line a\n>\n> line b"
    let display = Self.renderAndCollectDisplay(
      markdown: markdown, cursorPosition: (markdown as NSString).length,
      components: c)
    Self.writeSnapshot(c, name: "06-double-nl-in-bq", size: NSSize(width: 600, height: 200))

    // Two AST paragraphs inside the blockquote → at least two displayed
    // paragraphs visible. Neither half should carry a U+2028.
    let bqParagraphs = display.values.filter { $0.contains("line a") || $0.contains("line b") }
    #expect(
      bqParagraphs.count == 2,
      "expected two displayed blockquote paragraphs, got: \(display)")
    for s in bqParagraphs {
      #expect(
        !s.contains(Self.LS),
        "no soft-break char should appear in two-paragraph blockquote: \(s)")
    }
  }

  // MARK: - 7. Single newline in a nested blockquote → in-paragraph line break

  @Test
  func single_newline_in_nested_blockquote_renders_as_in_paragraph_line_break() {
    let c = Self.makeComponents()
    let markdown = "> > deep a\n> > deep b"
    let cursorPosition = (markdown as NSString).length
    _ = Self.renderAndCollectDisplay(
      markdown: markdown, cursorPosition: cursorPosition, components: c)
    Self.writeSnapshot(c, name: "07-single-nl-in-nested-bq", size: NSSize(width: 600, height: 200))

    let spec = Self.renderedSpec(markdown: markdown, cursorPosition: cursorPosition)
    let nlOffset = ("> > deep a" as NSString).length
    #expect(
      spec.lineBreakIndexes.contains(nlOffset),
      "parser should classify nested-blockquote `\\n` as a soft break")
    let firstStyle = Self.paragraphStyle(forSource: 0, in: spec)
    #expect(
      firstStyle?.paragraphSpacing == 0,
      "nested-blockquote soft-break first half should have paragraphSpacing=0")
    let secondStyle = Self.paragraphStyle(forSource: nlOffset + 1, in: spec)
    #expect(
      secondStyle?.paragraphSpacingBefore == 0,
      "nested-blockquote soft-break second half should have paragraphSpacingBefore=0")
  }

  // MARK: - 8. Lazy blockquote continuation — second line stays in the quote

  @Test
  func lazy_blockquote_continuation_stays_inside_blockquote() throws {
    let c = Self.makeComponents()
    let markdown = "> quote line\nplain line that lazily continues"
    let cursorPosition = (markdown as NSString).length
    _ = Self.renderAndCollectDisplay(
      markdown: markdown, cursorPosition: cursorPosition, components: c)
    Self.writeSnapshot(
      c, name: "08-lazy-continuation", size: NSSize(width: 600, height: 200))

    // CommonMark places the lazy continuation inside the blockquote — both
    // source lines are part of a single AST paragraph. The renderer should:
    //   (a) emit zero paragraph spacing between the two TK source halves
    //   (b) extend the blockquote border decoration to both halves
    let spec = Self.renderedSpec(markdown: markdown, cursorPosition: cursorPosition)
    let nlOffset = ("> quote line" as NSString).length
    #expect(
      spec.lineBreakIndexes.contains(nlOffset),
      "parser should classify the lazy-continuation `\\n` as a soft break")

    let firstStyle = Self.paragraphStyle(forSource: 0, in: spec)
    #expect(firstStyle?.paragraphSpacing == 0)
    let secondStyle = Self.paragraphStyle(forSource: nlOffset + 1, in: spec)
    #expect(secondStyle?.paragraphSpacingBefore == 0)

    // The blockquote border decoration's range must cover both source lines.
    let bqDecorations = spec.blockquoteCharacterRanges
    #expect(
      bqDecorations.count == 1,
      "expected one blockquote border level for lazy continuation, got \(bqDecorations.count)")
    if let bq = bqDecorations.first {
      let bqEnd = bq.range.location + bq.range.length
      let lazyEnd = (markdown as NSString).length
      #expect(
        bqEnd >= lazyEnd,
        "blockquote border range should cover the lazy line — bqEnd \(bqEnd) vs lazy line end \(lazyEnd)")
    }
  }

  // MARK: - 9. Mixed quote depth → two separate blockquote paragraphs

  @Test
  func mixed_quote_depth_remains_two_separate_blockquote_paragraphs() {
    let c = Self.makeComponents()
    let markdown = "> a\n> > b"
    let display = Self.renderAndCollectDisplay(
      markdown: markdown, cursorPosition: (markdown as NSString).length,
      components: c)
    Self.writeSnapshot(
      c, name: "09-mixed-quote-depth", size: NSSize(width: 600, height: 200))

    // Two AST paragraphs at different depths — must remain visually
    // distinct, with no soft-break character bridging them.
    #expect(
      display.count == 2,
      "expected two displayed paragraphs (depth 1 + depth 2), got \(display.count): \(display)")
    for (_, s) in display {
      #expect(
        !s.contains(Self.LS),
        "mixed-depth paragraphs must not carry a soft-break char: \(s)")
    }
  }

  // MARK: - 10. List item continuation → single paragraph

  @Test
  func list_item_continuation_renders_in_paragraph() {
    let c = Self.makeComponents()
    let markdown = "- item\n  continued line of the same item"
    let cursorPosition = (markdown as NSString).length
    _ = Self.renderAndCollectDisplay(
      markdown: markdown, cursorPosition: cursorPosition, components: c)
    Self.writeSnapshot(
      c, name: "10-list-continuation", size: NSSize(width: 600, height: 200))

    // List item with a continuation line is one AST paragraph with a soft
    // break. Same paragraph-spacing-zero contract as body paragraphs.
    let spec = Self.renderedSpec(markdown: markdown, cursorPosition: cursorPosition)
    let nlOffset = ("- item" as NSString).length
    #expect(
      spec.lineBreakIndexes.contains(nlOffset),
      "parser should classify list-continuation `\\n` as a soft break")

    let firstStyle = Self.paragraphStyle(forSource: 0, in: spec)
    #expect(
      firstStyle?.paragraphSpacing == 0,
      "list-continuation first half should have paragraphSpacing=0; got \(String(describing: firstStyle?.paragraphSpacing))"
    )
    let secondStyle = Self.paragraphStyle(forSource: nlOffset + 1, in: spec)
    #expect(
      secondStyle?.paragraphSpacingBefore == 0,
      "list-continuation second half should have paragraphSpacingBefore=0; got \(String(describing: secondStyle?.paragraphSpacingBefore))"
    )
  }

  // MARK: - 11. Adjacent list items render as separate paragraphs

  @Test
  func adjacent_list_items_render_as_separate_paragraphs() {
    let c = Self.makeComponents()
    // Trailing body paragraph hosts the cursor so both list items are
    // "outside" the cursor and both receive bullet substitution.
    let markdown = "- item line a\n- item line b\n\nbody"
    let bodyOffset = ("- item line a\n- item line b\n\n" as NSString).length
    let display = Self.renderAndCollectDisplay(
      markdown: markdown, cursorPosition: bodyOffset,
      components: c)
    Self.writeSnapshot(
      c, name: "11-list-soft-break", size: NSSize(width: 600, height: 200))

    let listParagraphs = display.filter { $0.value.contains("item line") }
    #expect(
      listParagraphs.count == 2,
      "expected two displayed list-item paragraphs, got: \(display)")
    for (_, s) in listParagraphs {
      // Each item line should start with the bullet glyph after the `- `
      // marker substitution, and contain no soft-break character.
      #expect(s.hasPrefix("\u{2022} "), "expected bullet glyph prefix on: \(s)")
      #expect(!s.contains(Self.LS), "no soft-break char in list items: \(s)")
    }
  }

  // MARK: - 12. Body then heading (no blank line) → two paragraphs

  @Test
  func body_immediately_followed_by_heading_renders_as_two_paragraphs() {
    let c = Self.makeComponents()
    let markdown = "text\n# Heading"
    let display = Self.renderAndCollectDisplay(
      markdown: markdown, cursorPosition: (markdown as NSString).length,
      components: c)
    Self.writeSnapshot(
      c, name: "12-body-then-heading", size: NSSize(width: 600, height: 200))

    #expect(
      display.count == 2,
      "expected two displayed paragraphs, got \(display.count): \(display)")
    for (_, s) in display {
      #expect(!s.contains(Self.LS), "no soft-break char between body+heading: \(s)")
    }
  }

  // MARK: - 13. Heading then body (no blank line) → two paragraphs

  @Test
  func heading_immediately_followed_by_body_renders_as_two_paragraphs() {
    let c = Self.makeComponents()
    let markdown = "# Heading\ntext"
    let display = Self.renderAndCollectDisplay(
      markdown: markdown, cursorPosition: (markdown as NSString).length,
      components: c)
    Self.writeSnapshot(
      c, name: "13-heading-then-body", size: NSSize(width: 600, height: 200))

    #expect(
      display.count == 2,
      "expected two displayed paragraphs, got \(display.count): \(display)")
    for (_, s) in display {
      #expect(!s.contains(Self.LS), "no soft-break char between heading+body: \(s)")
    }
  }

  // MARK: - 14. Code-block internal newlines remain literal

  @Test
  func code_block_internal_newlines_remain_literal_newlines() {
    let c = Self.makeComponents()
    let markdown = "```\nline 1\nline 2\nline 3\n```"
    let display = Self.renderAndCollectDisplay(
      markdown: markdown, cursorPosition: (markdown as NSString).length,
      components: c)
    Self.writeSnapshot(
      c, name: "14-code-block-newlines", size: NSSize(width: 720, height: 280))

    // Per AGENTS.md goal #3, code blocks are leaf and avoid normalization.
    // Each interior \n must remain a real paragraph break in the displayed
    // output — NEVER a U+2028.
    for (_, s) in display {
      #expect(
        !s.contains(Self.LS),
        "code block content must not contain LINE SEPARATOR: \(s)")
    }
    // Sanity: at least one paragraph contains code content.
    let joined = display.values.joined()
    #expect(joined.contains("line 1"))
    #expect(joined.contains("line 2"))
    #expect(joined.contains("line 3"))
  }

  // MARK: - 15. Five `\n` (a..\n\n\n\n\n..b) — adjacent empty paragraphs

  // Decision: per the same "every two newlines = one paragraph break" rule
  // as case 3/4, five `\n` is two complete pairs plus one orphan. Two pairs
  // = one paragraph break + one preserved empty paragraph. The orphan adds
  // no additional empty paragraph (a third empty paragraph requires a third
  // pair). So the user sees THREE paragraphs (a, blank, b), matching case 4.
  // We pin only that user-observable shape; the precise handling of the
  // orphan `\n` is left to the implementation phase.
  @Test
  func adjacent_empty_paragraphs_preserve_user_visual_spacing() {
    let c = Self.makeComponents()
    let markdown = "a\n\n\n\n\nb"
    let display = Self.renderAndCollectDisplay(
      markdown: markdown, cursorPosition: (markdown as NSString).length,
      components: c)
    Self.writeSnapshot(
      c, name: "15-adjacent-empty-paragraphs", size: NSSize(width: 600, height: 280))

    let nonEmpty = display.values.filter {
      !$0.trimmingCharacters(in: .newlines).isEmpty
    }
    let onlyEmpty = display.values.filter {
      $0.trimmingCharacters(in: .newlines).isEmpty
    }
    #expect(
      nonEmpty.count == 2,
      "expected two non-empty content paragraphs, got: \(display)")
    #expect(
      onlyEmpty.count == 1,
      "expected exactly one preserved empty paragraph (one orphan \\n absorbed), got \(onlyEmpty.count): \(display)")
    for (_, s) in display {
      #expect(!s.contains(Self.LS), "no soft-break char anywhere: \(s)")
    }
  }

  // MARK: - 16. Hard line break (`  \n`) renders as in-paragraph line break

  // TODO: separately verify that the trailing two spaces are themselves
  // hidden by the content delegate so the displayed paragraph reads
  // "line a / line b" without the trailing whitespace. This test only pins
  // the spacing contract; the spaces-hiding behavior is a follow-on.
  @Test
  func hard_line_break_via_trailing_two_spaces_renders_as_in_paragraph_line_break() {
    let c = Self.makeComponents()
    let markdown = "line a  \nline b"
    let cursorPosition = (markdown as NSString).length
    _ = Self.renderAndCollectDisplay(
      markdown: markdown, cursorPosition: cursorPosition, components: c)
    Self.writeSnapshot(
      c, name: "16-hard-line-break", size: NSSize(width: 600, height: 200))

    // Hard line break (`  \n`) is a `LineBreak` AST node — it should be
    // recorded in `lineBreakIndexes` exactly the same way as a `SoftBreak`,
    // and produce zero paragraph spacing between the two halves.
    let spec = Self.renderedSpec(markdown: markdown, cursorPosition: cursorPosition)
    let nlOffset = ("line a  " as NSString).length
    #expect(
      spec.lineBreakIndexes.contains(nlOffset),
      "parser should classify hard-break `\\n` (after `  `) as a line break")

    let firstStyle = Self.paragraphStyle(forSource: 0, in: spec)
    #expect(firstStyle?.paragraphSpacing == 0)
    let secondStyle = Self.paragraphStyle(forSource: nlOffset + 1, in: spec)
    #expect(secondStyle?.paragraphSpacingBefore == 0)
  }

  // MARK: - 17. Cursor walks across a soft break with arrow keys

  @Test
  func cursor_can_walk_across_soft_break_with_arrow_keys() throws {
    let c = Self.makeComponents()
    let markdown = "Hello\nworld"
    let textView = c.textView
    textView.string = markdown
    let cursorRange = NSRange(location: 5, length: 0)
    textView.setSelectedRange(cursorRange)
    let spec = MarkdownRenderer.render(
      text: markdown, cursorRange: cursorRange, style: .default)
    TextKit2RenderApplicator.apply(spec, to: textView)
    if let tlm = textView.textLayoutManager {
      tlm.ensureLayout(for: tlm.documentRange)
    }

    let tlm = try #require(textView.textLayoutManager)
    let cs = try #require(textView.textContentStorage)
    let nav = tlm.textSelectionNavigation

    // Build an initial selection at offset 5 (right after `o` in `Hello`).
    let docStart = tlm.documentRange.location
    let startLoc = try #require(tlm.location(docStart, offsetBy: 5))
    var sel = NSTextSelection(startLoc, affinity: .downstream)

    var visited: [Int] = [5]
    // Walk three rights and record the document-offset of each new head.
    for _ in 0..<3 {
      guard let next = nav.destinationSelection(
        for: sel, direction: .right,
        destination: .character, extending: false,
        confined: false)
      else { break }
      sel = next
      let head = next.textRanges.first?.location ?? next.textRanges.first?.endLocation
      if let h = head {
        let off = cs.offset(from: docStart, to: h)
        visited.append(off)
      }
    }

    // Acceptable behavior: the cursor walks monotonically across the soft
    // break to at least offset 7 (the first character of "world"). Today the
    // selection navigation can stall at the soft-break boundary because the
    // newline lives between two paragraphs. Post-fix it must walk through.
    #expect(
      visited.last! >= 7,
      "cursor failed to walk across soft break — visited offsets: \(visited)")
    // And it must be strictly monotonic (no getting stuck on one offset).
    let monotonic = zip(visited, visited.dropFirst()).allSatisfy { $0 < $1 }
    #expect(monotonic, "cursor must advance on every right-arrow: \(visited)")
  }

  // MARK: - 18. Copy across a soft break yields literal `\n` in clipboard

  @Test
  func copy_across_soft_break_yields_literal_newline_in_clipboard() throws {
    let c = Self.makeComponents()
    let markdown = "Hello\nworld"
    let textView = c.textView
    textView.string = markdown
    let cursorRange = NSRange(location: 0, length: (markdown as NSString).length)
    textView.setSelectedRange(cursorRange)
    let spec = MarkdownRenderer.render(
      text: markdown, cursorRange: NSRange(location: 0, length: 0),
      style: .default)
    TextKit2RenderApplicator.apply(spec, to: textView)
    if let tlm = textView.textLayoutManager {
      tlm.ensureLayout(for: tlm.documentRange)
    }

    // The user-copyable representation of a selection comes from the
    // underlying textStorage / textContentStorage attributed string — that's
    // what `writeSelection(to:types:)` would put on the pasteboard. We read
    // that directly so the test isn't sensitive to pasteboard-flushing
    // quirks in headless test runs. The invariant we care about is that the
    // copy is *source* coordinates, never the U+2028 display substitution.
    let attr = try #require(textView.textStorage)
    let pasted = attr.attributedSubstring(from: cursorRange).string

    #expect(
      pasted == "Hello\nworld",
      "copy must carry the literal source string, got: \(pasted.unicodeScalars.map { String(format: "U+%04X", $0.value) }.joined(separator: " "))")
    #expect(
      !pasted.contains(Self.LS),
      "copy must NEVER contain LINE SEPARATOR: \(pasted)")
  }

  // MARK: - 18b. Soft break in second block doesn't bleed into first block's spacing

  /// Regression for a parser bug where swift-markdown's `SoftBreak` AST node
  /// has `range = nil`, and the fallback in `recordInlineLineBreak` scanned
  /// the whole document. It found the *first* `\n` (the inter-block
  /// separator after paragraph 1), recorded that as the soft-break offset,
  /// and the renderer then split paragraph 1's styled range at that offset
  /// — making paragraph 1's `paragraphSpacing` zero. Visual outcome: P1 sat
  /// flush against P2 while the actual soft break inside P2 was treated as
  /// a normal paragraph break with full spacing. Symptom: in
  /// `"P1.\n\nP2.\nP3."`, P1 and P2 looked merged and P3 was the only
  /// visually separated block.
  @Test
  func soft_break_in_second_block_does_not_zero_first_blocks_spacing() {
    let markdown = "Paragraph 1.\n\nParagraph 2.\nParagraph 3."
    let cursor = (markdown as NSString).length
    let spec = Self.renderedSpec(markdown: markdown, cursorPosition: cursor)

    // The soft break MUST be recorded inside paragraph 2 (the `\n` between
    // "Paragraph 2." and "Paragraph 3."), at source offset 26 — never at
    // the inter-block separator at offset 12.
    #expect(
      spec.lineBreakIndexes.contains(26),
      "expected soft-break recorded at the within-paragraph `\\n` at offset 26; got \(Array(spec.lineBreakIndexes))")
    #expect(
      !spec.lineBreakIndexes.contains(12),
      "soft-break must NOT be recorded at the inter-block separator at offset 12")

    // Paragraph 1's effective trailing paragraphSpacing must be the full
    // body spacing (8pt) so P1 visually separates from P2.
    let p1Style = Self.paragraphStyle(forSource: 0, in: spec)
    #expect(
      p1Style?.paragraphSpacing == MarkdownStyle.default.paragraphSpacing,
      "P1's paragraphSpacing should be \(MarkdownStyle.default.paragraphSpacing); got \(String(describing: p1Style?.paragraphSpacing))")

    // Paragraph 2 segment 1 (covers "Paragraph 2.\n") should have zero
    // trailing spacing so P2 is flush with P3.
    let p2FirstStyle = Self.paragraphStyle(forSource: 14, in: spec)
    #expect(p2FirstStyle?.paragraphSpacing == 0)

    // Paragraph 2 segment 2 (covers "Paragraph 3.") should have zero
    // leading spacing for symmetric tightness.
    let p3Style = Self.paragraphStyle(forSource: 27, in: spec)
    #expect(p3Style?.paragraphSpacingBefore == 0)
  }

  // MARK: - 18c. Enter snaps the trailing-`\n` count to an even number

  /// Enter inserts `\n\n` when the cursor's trailing `\n` count is even
  /// (including 0), and `\n` when it's odd. Combined with the renderer's
  /// "every two `\n` produces one visible empty paragraph" rule, this means
  /// each Enter press reliably advances the visible spacing by exactly one
  /// paragraph break. Without the snap, an odd trailing count would absorb
  /// the next Enter silently as an orphan.
  @Test
  func enter_snaps_trailing_newline_count_to_even() {
    // 0 trailing → insert "\n\n"
    var state = EditorState(markdown: "para1", selection: .cursor(5))
    state = EditorUpdate.update(state, event: .insertNewline)
    #expect(state.markdown == "para1\n\n", "1st Enter: \(state.markdown)")
    #expect(state.selection == .cursor(7))

    // 2 trailing → insert "\n\n" → 4 trailing
    state = EditorUpdate.update(state, event: .insertNewline)
    #expect(state.markdown == "para1\n\n\n\n", "2nd Enter: \(state.markdown)")
    #expect(state.selection == .cursor(9))

    // Now simulate a stray odd \n (e.g. left over from a prior Shift+Enter):
    // 1 trailing → insert "\n" → 2 trailing
    state = EditorState(markdown: "para1\n", selection: .cursor(6))
    state = EditorUpdate.update(state, event: .insertNewline)
    #expect(
      state.markdown == "para1\n\n",
      "Enter after stray \\n should snap to `\\n\\n`, got: \(state.markdown)")
    #expect(state.selection == .cursor(7))

    // 3 trailing → insert "\n" → 4 trailing
    state = EditorState(markdown: "para1\n\n\n", selection: .cursor(8))
    state = EditorUpdate.update(state, event: .insertNewline)
    #expect(state.markdown == "para1\n\n\n\n", "Enter at 3 trailing: \(state.markdown)")
    #expect(state.selection == .cursor(9))
  }

  // MARK: - 19. Backspace at a paragraph boundary deletes one `\n` at a time

  @Test
  func backspace_at_paragraph_boundary_deletes_one_newline_at_a_time() {
    // Hybrid newline policy: Backspace deletes ONE character at a time even
    // at a `\n\n` paragraph boundary. The first press collapses the gap to
    // a soft break; a second press joins the halves. The earlier
    // pair-consume behavior was tightly coupled to a `normalizeSoftLineBreaks`
    // pass that has been retired and is no longer correct.
    let state = EditorState(
      markdown: "para1\n\npara2", selection: .cursor(7))
    let firstBackspace = EditorUpdate.update(state, event: .deleteBackward)
    #expect(
      firstBackspace.markdown == "para1\npara2",
      "first Backspace should collapse `\\n\\n` to `\\n`, got: \(firstBackspace.markdown)")
    #expect(firstBackspace.selection == .cursor(6))

    let secondBackspace = EditorUpdate.update(firstBackspace, event: .deleteBackward)
    #expect(
      secondBackspace.markdown == "para1para2",
      "second Backspace should join the halves, got: \(secondBackspace.markdown)")
    #expect(secondBackspace.selection == .cursor(5))
  }

  // MARK: - 20. Enter at end of paragraph still creates a paragraph break

  @Test
  func enter_at_end_of_paragraph_still_creates_blank_line_for_paragraph_break() {
    // Cursor at the very end of "hello" (a single body paragraph). Enter
    // should produce `"hello\n\n"` so the next character begins a new
    // paragraph — this is the AGENTS.md "every two \n = paragraph break"
    // exception in action.
    let state = EditorState(markdown: "hello", selection: .cursor(5))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(
      result.markdown == "hello\n\n",
      "expected 'hello\\n\\n', got: \(result.markdown)")
    #expect(result.selection == .cursor(7))
  }

  // MARK: - 21. Enter inside a paragraph creates a paragraph break too

  // Decision: pin existing behavior — Enter splits the paragraph by
  // inserting `\n\n`. This is the same code path as case 20, just with
  // text on both sides. Worth pinning so the soft-break work doesn't
  // accidentally turn this into a single-newline insert.
  @Test
  func enter_inside_paragraph_creates_soft_break_not_paragraph_break() {
    let state = EditorState(
      markdown: "hello world", selection: .cursor(5))
    let result = EditorUpdate.update(state, event: .insertNewline)
    #expect(
      result.markdown == "hello\n\n world",
      "expected paragraph split via \\n\\n, got: \(result.markdown)")
    #expect(result.selection == .cursor(7))
  }
}
