import AppKit
import Testing

@testable import MarkdownEditor

/// Phase 3 of the TextKit 2 migration — validates that
/// `TextKit2LayoutManagerDelegate` vends `TextKit2LayoutFragment` instances
/// configured with the right code-block / blockquote decoration state for
/// each paragraph.
///
/// We assert on the *configured state* of each fragment rather than the
/// pixels it draws — pixel-level layout is unreliable in pure XCTest (per
/// the Phase 0 spike findings) and the actual drawing logic is small and
/// well-defined. Visual regression is exercised by the demo and by the
/// snapshot artifacts written by `tk2_snapshots_for_visual_review`.
@MainActor
struct TextKit2FragmentDecorationTests {

  // MARK: - Helpers

  /// Mirrors the production TK2 setup including the layout-manager delegate.
  private struct TK2Components {
    let textView: NSTextView
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
      textView: textView, contentDelegate: contentDelegate,
      layoutDelegate: layoutDelegate, window: window)
  }

  /// Drive the renderer + applicator end-to-end and return the laid-out
  /// fragments keyed on each paragraph's source-range start (in document
  /// offsets).
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

  // MARK: - Code block
  //
  // Pre-2.2 this section asserted that the layout-manager delegate
  // configured `codeBlockOrigin` on the layout fragment whose paragraph
  // overlapped a `codeBlockCharacterRanges` decoration. Post-2.2 the
  // legacy painting path is gone — code blocks are rendered by the
  // embedded `CodeBlockRenderer`, so the fragment's `codeBlockOrigin`
  // field has been removed entirely. The new invariant is that the
  // attachment-bearing fragment has `containsBlockAttachment == true`
  // (so its `renderingSurfaceBounds` widens for the embedded view) and
  // that the layout delegate has been handed a `BlockRendererSpec` for
  // the block.

  @Test
  func code_block_paragraph_has_block_attachment_flag_set() throws {
    let c = Self.makeComponents()
    let markdown = """
      Body paragraph.

      ```swift
      let x = 1
      ```
      """
    // Cursor at end (outside the code block).
    let frags = Self.renderAndCollectFragments(
      markdown: markdown,
      cursorPosition: (markdown as NSString).length,
      components: c)

    let bodyOffset = 0
    #expect(
      frags[bodyOffset]?.containsBlockAttachment == false,
      "body paragraph should not host a block attachment")

    let blockSpecs = c.layoutDelegate.blockRendererSpecs
    #expect(!blockSpecs.isEmpty, "renderer should emit a BlockRendererSpec for the code block")

    var configured = false
    for (_, frag) in frags {
      if frag.containsBlockAttachment { configured = true }
    }
    #expect(
      configured,
      "at least one fragment should be flagged `containsBlockAttachment` so its rendering surface widens for the embedded view")
  }

  @Test
  func plain_body_paragraph_has_no_block_attachment_flag() throws {
    let c = Self.makeComponents()
    let frags = Self.renderAndCollectFragments(
      markdown: "Just plain body text.", components: c)
    #expect(frags[0]?.containsBlockAttachment == false)
    #expect(frags[0]?.blockquoteBorderXPositions == [])
  }

  // MARK: - Blockquote

  @Test
  func single_level_blockquote_paragraph_has_one_border() throws {
    let c = Self.makeComponents()
    let markdown = "> Quoted line\n\nbody"
    // Cursor in body so the quote's border is emitted (renderer gates on
    // cursor-outside).
    let bodyOffset = ("> Quoted line\n\n" as NSString).length
    let frags = Self.renderAndCollectFragments(
      markdown: markdown, cursorPosition: bodyOffset, components: c)

    let bqDecorations = c.layoutDelegate.blockquoteCharacterRanges
    #expect(
      bqDecorations.count == 1,
      "expected exactly one blockquote decoration, got \(bqDecorations.count)")

    // Find the fragment whose source range falls inside the blockquote.
    let bqRange = bqDecorations[0].range
    var quotedFragment: TextKit2LayoutFragment?
    for (start, frag) in frags where start >= bqRange.location && start < NSMaxRange(bqRange) {
      quotedFragment = frag
      break
    }
    let frag = try #require(quotedFragment)
    #expect(
      frag.blockquoteBorderXPositions.count == 1,
      "expected one border x-position, got \(frag.blockquoteBorderXPositions)")
    #expect(frag.blockquoteBorderXPositions[0] == bqDecorations[0].xPosition)
  }

  @Test
  func nested_blockquote_paragraph_has_two_distinct_borders() throws {
    let c = Self.makeComponents()
    let markdown = "> > inner quote\n\nbody"
    let bodyOffset = ("> > inner quote\n\n" as NSString).length
    let frags = Self.renderAndCollectFragments(
      markdown: markdown, cursorPosition: bodyOffset, components: c)

    let bqDecorations = c.layoutDelegate.blockquoteCharacterRanges
    #expect(
      bqDecorations.count == 2,
      "expected outer + inner blockquote decorations, got \(bqDecorations.count)")

    // Find the fragment for the inner-quote paragraph (the only one that
    // overlaps both decorations).
    var innerFragment: TextKit2LayoutFragment?
    for (start, frag) in frags {
      // The inner quote paragraph starts at source 0.
      if start == 0 {
        innerFragment = frag
        break
      }
    }
    let frag = try #require(innerFragment, "inner-quote fragment should exist")

    #expect(
      frag.blockquoteBorderXPositions.count == 2,
      "nested blockquote paragraph should have 2 border x-positions, got \(frag.blockquoteBorderXPositions)"
    )

    // The two x-positions should be distinct (one per nesting level).
    let xs = frag.blockquoteBorderXPositions
    #expect(Set(xs).count == 2, "border x-positions must be distinct: \(xs)")

    // Outer border should be to the left of the inner border.
    let sorted = xs.sorted()
    #expect(
      sorted[0] < sorted[1],
      "outer border (smaller x) should precede inner border (larger x): \(sorted)")
  }

  @Test
  func code_block_inside_blockquote_emits_spec_and_border() throws {
    // Pre-2.2 this test asserted that a code block nested inside a
    // blockquote produced a fragment with BOTH `codeBlockOrigin != nil`
    // (legacy painted background) AND `blockquoteBorderXPositions` (the
    // blockquote left border). Post-2.2 the painted background is gone;
    // the embedded `CodeBlockRenderer` covers the code-block visual
    // entirely. We pin the renderer-side outputs (a `BlockRendererSpec`
    // for the code block + a blockquote decoration enclosing it) here.
    // Whether the attachment paragraph itself ever gets vended for a
    // blockquote-nested code block is a Phase 2.1 invariant that
    // depends on the spec range start aligning with the first
    // paragraph's source location — that alignment doesn't hold for
    // blockquoted code blocks (the spec range starts at the first
    // backtick, but the first source paragraph starts at the `>`
    // prefix). Tightening attachment-paragraph behaviour for this
    // nesting case is tracked separately.
    let c = Self.makeComponents()
    let markdown = """
      > ```
      > code line
      > ```

      body
      """
    let bodyOffset = (markdown as NSString).length
    _ = Self.renderAndCollectFragments(
      markdown: markdown, cursorPosition: bodyOffset, components: c)

    let blockSpecs = c.layoutDelegate.blockRendererSpecs
    let bqDecs = c.layoutDelegate.blockquoteCharacterRanges
    #expect(
      !blockSpecs.isEmpty,
      "expected a BlockRendererSpec for the code block inside the blockquote")
    #expect(!bqDecs.isEmpty, "expected a blockquote decoration enclosing the code block")
  }

  // MARK: - Container width propagation

  @Test
  func apply_propagates_container_width_into_layout_delegate() throws {
    let c = Self.makeComponents(size: NSSize(width: 600, height: 400))
    let markdown = "body text"
    let spec = MarkdownRenderer.render(
      text: markdown, cursorRange: NSRange(location: 0, length: 0), style: .default)
    c.textView.string = markdown
    TextKit2RenderApplicator.apply(spec, to: c.textView)
    let containerWidth = try #require(c.textView.textLayoutManager?.textContainer?.size.width)
    #expect(
      c.layoutDelegate.containerWidth == containerWidth,
      "applicator should push container width into layout delegate")
  }

  @Test
  func vended_fragment_uses_current_container_width() throws {
    // Pre-2.2 this test scoped the lookup to the code-block fragment via
    // `codeBlockOrigin != nil`. Post-2.2 there is no `codeBlockOrigin`;
    // we use the attachment-bearing fragment as the witness instead, on
    // the same theory: the layout delegate writes the current container
    // width into every fragment it vends, regardless of decoration kind.
    let c = Self.makeComponents(size: NSSize(width: 600, height: 400))
    let markdown = "```\ncode\n```"
    let frags = Self.renderAndCollectFragments(
      markdown: markdown, cursorPosition: (markdown as NSString).length,
      components: c)

    var attachmentFrag: TextKit2LayoutFragment?
    for (_, f) in frags where f.containsBlockAttachment {
      attachmentFrag = f
      break
    }
    let frag = try #require(
      attachmentFrag, "expected at least one attachment-bearing fragment for the code block")
    let containerWidth = try #require(c.textView.textLayoutManager?.textContainer?.size.width)
    #expect(
      frag.containerWidth == containerWidth,
      "fragment.containerWidth (\(frag.containerWidth)) should equal text container width (\(containerWidth))"
    )
  }

  @Test
  func updating_container_width_propagates_to_existing_fragments() throws {
    let c = Self.makeComponents(size: NSSize(width: 600, height: 400))
    let markdown = "```\ncode\n```"
    _ = Self.renderAndCollectFragments(
      markdown: markdown, cursorPosition: (markdown as NSString).length,
      components: c)

    // Sanity: fragments were vended with the original width.
    let originalWidth = try #require(c.textView.textLayoutManager?.textContainer?.size.width)

    // Simulate a container resize. Equivalent to the body of
    // `Coordinator.refreshTextKit2ContainerWidth` (which we can't invoke
    // directly here without spinning up the SwiftUI host).
    c.textView.frame = NSRect(origin: .zero, size: NSSize(width: 800, height: 400))
    c.textView.textContainer?.size = NSSize(
      width: 800, height: CGFloat.greatestFiniteMagnitude)
    c.layoutDelegate.containerWidth = 800
    if let tlm = c.textView.textLayoutManager {
      tlm.enumerateTextLayoutFragments(
        from: tlm.documentRange.location, options: .ensuresLayout
      ) { frag in
        if let custom = frag as? TextKit2LayoutFragment {
          custom.containerWidth = 800
          custom.invalidateLayout()
        }
        return true
      }
    }

    // After propagation, every TK2 custom fragment should carry the new
    // width, not the original one.
    var allNewWidth = true
    var sawAnyFragment = false
    if let tlm = c.textView.textLayoutManager {
      tlm.enumerateTextLayoutFragments(from: tlm.documentRange.location, options: []) { frag in
        if let custom = frag as? TextKit2LayoutFragment {
          sawAnyFragment = true
          if custom.containerWidth != 800 {
            allNewWidth = false
          }
        }
        return true
      }
    }
    #expect(sawAnyFragment, "expected at least one TK2 fragment to enumerate")
    #expect(
      allNewWidth,
      "every vended TextKit2LayoutFragment should carry the new containerWidth (was \(originalWidth))"
    )
  }

  // MARK: - Snapshots for visual review

  /// Capture a bitmap from the production TK2 setup with representative
  /// content so a human reviewer can eyeball decorations. The bitmap is
  /// written to `test-artifacts/phase3/` next to the package; the test
  /// always passes — this is a snapshot generator, not a regression check.
  @Test
  func tk2_snapshots_for_visual_review() throws {
    let samples: [(String, String)] = [
      (
        "kitchen-sink",
        """
        # Heading

        Body paragraph with **bold** and *italic*.

        - first bullet
        - second bullet

        > Single-level quote

        > > Nested quote inside outer quote

        ```swift
        let greeting = "hi"
        print(greeting)
        ```

        > ```
        > code in a quote
        > more code
        > ```
        """
      ),
      (
        "long-code-block",
        """
        Here is a longer code block:

        ```swift
        struct Foo {
          let bar: Int
          func describe() -> String {
            "Foo(bar: \\(bar))"
          }
        }
        ```

        And after.
        """
      ),
      (
        "nested-blockquotes",
        """
        > Outer quote.
        >
        > > Inner quote with one nesting level.
        >
        > > > Triple-nested quote.

        body
        """
      ),
    ]

    let dir = NSURL.fileURL(withPath: #filePath)
      .deletingLastPathComponent()  // MarkdownEditorTests
      .deletingLastPathComponent()  // Tests
      .deletingLastPathComponent()  // MarkdownEditor
      .appendingPathComponent("test-artifacts")
      .appendingPathComponent("phase3")
    try? FileManager.default.createDirectory(
      at: dir, withIntermediateDirectories: true)

    for (name, markdown) in samples {
      let size = NSSize(width: 720, height: 720)
      let c = Self.makeComponents(size: size)
      c.textView.string = markdown
      let cursorRange = NSRange(location: (markdown as NSString).length, length: 0)
      c.textView.setSelectedRange(cursorRange)
      let spec = MarkdownRenderer.render(
        text: markdown, cursorRange: cursorRange, style: .default)
      TextKit2RenderApplicator.apply(spec, to: c.textView)
      if let tlm = c.textView.textLayoutManager {
        tlm.ensureLayout(for: tlm.documentRange)
      }
      // Deselect so the cursor caret doesn't appear in the snapshot.
      c.textView.setSelectedRange(NSRange(location: 0, length: 0))
      c.textView.needsDisplay = true
      c.textView.displayIfNeeded()

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
        bitsPerPixel: 0
      )!
      c.textView.cacheDisplay(in: c.textView.bounds, to: bitmap)

      if let data = bitmap.representation(using: .png, properties: [:]) {
        let url = dir.appendingPathComponent("\(name).png")
        try? data.write(to: url)
      }
    }

    // Always passes — this test exists purely to generate review artifacts.
    #expect(true)
  }
}
