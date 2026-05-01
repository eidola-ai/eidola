import AppKit
import Testing

@testable import MarkdownEditor

/// Phase 2.1 of the Block Renderer migration — validates the bridging-layer
/// infrastructure that ferries `BlockRendererSpec`s from the renderer
/// through the applicator to live `BlockRenderHost`s, and from there to
/// the `BlockAttachment` the content-storage delegate vends.
///
/// These tests build a TK2 NSTextView inline (mirroring
/// `TextKit2GlyphHidingTests`) so they can drive
/// `MarkdownRenderer.render` + `TextKit2RenderApplicator.apply` end-to-end
/// and inspect the registry's host table directly.
@MainActor
struct BlockRendererInfraTests {

  // MARK: - Helpers

  private struct TK2Components {
    let textView: NSTextView
    let contentStorage: NSTextContentStorage
    let layoutManager: NSTextLayoutManager
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
      layoutManager: textView.textLayoutManager!,
      contentDelegate: contentDelegate,
      layoutDelegate: layoutDelegate,
      window: window)
  }

  /// Drive renderer + applicator end-to-end against `markdown` with `cursor`.
  private static func render(
    markdown: String,
    cursor: Int = 0,
    components: TK2Components
  ) {
    let textView = components.textView
    textView.string = markdown
    let cursorRange = NSRange(location: cursor, length: 0)
    textView.setSelectedRange(cursorRange)
    let spec = MarkdownRenderer.render(text: markdown, cursorRange: cursorRange)
    TextKit2RenderApplicator.apply(spec, to: textView)
    if let tlm = textView.textLayoutManager {
      tlm.ensureLayout(for: tlm.documentRange)
    }
  }

  // MARK: - Spec reconciliation

  @Test
  func applying_spec_creates_host_for_codeblock_range() {
    let c = Self.makeComponents()
    let markdown = "para\n\n```\nlet x = 1\n```\n\nafter"
    Self.render(markdown: markdown, components: c)

    let hosts = BlockRendererRegistry.shared.hosts(for: c.textView)
    #expect(hosts.count == 1, "expected one host for the single code block, got \(hosts.count)")
    #expect(hosts.first?.blockTypeTag == .codeBlock)
  }

  @Test
  func applying_same_spec_again_reuses_the_host() {
    let c = Self.makeComponents()
    let markdown = "```\nlet x = 1\n```"
    Self.render(markdown: markdown, components: c)
    let firstHosts = BlockRendererRegistry.shared.hosts(for: c.textView)
    let firstHost = firstHosts.first
    #expect(firstHost != nil)

    // Re-apply with the same source / cursor — host identity should hold.
    Self.render(markdown: markdown, components: c)
    let secondHosts = BlockRendererRegistry.shared.hosts(for: c.textView)
    #expect(secondHosts.count == 1)
    #expect(secondHosts.first === firstHost, "host identity must persist across re-renders")
  }

  @Test
  func applying_different_range_retires_old_host_and_creates_new() {
    let c = Self.makeComponents()
    // First render: code block at offset 0.
    Self.render(markdown: "```\na\n```", components: c)
    let firstHosts = BlockRendererRegistry.shared.hosts(for: c.textView)
    let firstHost = firstHosts.first
    #expect(firstHost != nil)
    #expect(firstHost?.spec.range.location == 0)

    // Second render: code block now at a different offset — old host
    // (location 0) retires, new host (location 6) appears.
    Self.render(markdown: "para\n\n```\na\n```", components: c)
    let secondHosts = BlockRendererRegistry.shared.hosts(for: c.textView)
    #expect(secondHosts.count == 1)
    #expect(secondHosts.first?.spec.range.location == 6)
    #expect(secondHosts.first !== firstHost, "host at new offset must be a fresh instance")
  }

  @Test
  func applying_empty_specs_retires_all_hosts() {
    let c = Self.makeComponents()
    Self.render(markdown: "```\na\n```", components: c)
    #expect(BlockRendererRegistry.shared.hosts(for: c.textView).count == 1)

    // Render markdown with no code block → spec list is empty → all
    // hosts retire.
    Self.render(markdown: "just body text", components: c)
    #expect(BlockRendererRegistry.shared.hosts(for: c.textView).isEmpty)
  }

  // MARK: - Length-matching invariant for attachment paragraph

  /// The first paragraph of a block-renderer spec range is vended with a
  /// length-matched display string `[U+FFFC][ZWSP × N-1][\n]`. The total
  /// UTF-16 length must equal the source paragraph's length so TK2's
  /// hit-test, navigation, and rendering attributes stay honest at this
  /// element boundary.
  @Test
  func attachment_paragraph_preserves_length_matching_invariant() {
    let c = Self.makeComponents()
    let markdown = "```\nlet x = 1\n```"
    Self.render(markdown: markdown, components: c)

    let cs = c.contentStorage
    var firstParagraphSourceLen: Int?
    var firstParagraphDisplayLen: Int?
    var firstParagraphContainsObjectChar = false

    if let tlm = c.textView.textLayoutManager {
      tlm.enumerateTextLayoutFragments(
        from: tlm.documentRange.location, options: []
      ) { frag in
        guard let element = frag.textElement,
          let paragraph = element as? NSTextParagraph,
          let elemRange = element.elementRange
        else { return true }
        let start = cs.offset(from: cs.documentRange.location, to: elemRange.location)
        guard start == 0 else { return true }
        let length = cs.offset(from: elemRange.location, to: elemRange.endLocation)
        firstParagraphSourceLen = length
        firstParagraphDisplayLen = (paragraph.attributedString.string as NSString).length
        firstParagraphContainsObjectChar = paragraph.attributedString.string.contains("\u{FFFC}")
        return false  // first paragraph only
      }
    }

    #expect(firstParagraphSourceLen != nil, "expected a first paragraph to be enumerated")
    #expect(
      firstParagraphContainsObjectChar,
      "first paragraph of block-renderer range must carry the U+FFFC attachment glyph")
    if let src = firstParagraphSourceLen, let disp = firstParagraphDisplayLen {
      #expect(
        disp == src,
        "attachment paragraph length-matching invariant: display \(disp) must equal source \(src)")
    }
  }

  // MARK: - Sibling paragraph hiding

  @Test
  func sibling_paragraphs_inside_block_range_are_excluded_from_layout() {
    let c = Self.makeComponents()
    let markdown = "```\nline 1\nline 2\nline 3\n```"
    Self.render(markdown: markdown, components: c)

    var enumeratedStarts: [Int] = []
    let cs = c.contentStorage
    if let tlm = c.textView.textLayoutManager {
      tlm.enumerateTextLayoutFragments(
        from: tlm.documentRange.location, options: []
      ) { frag in
        if let elemRange = frag.textElement?.elementRange {
          let s = cs.offset(from: cs.documentRange.location, to: elemRange.location)
          enumeratedStarts.append(s)
        }
        return true
      }
    }

    // Only the FIRST paragraph (offset 0) of the block should be
    // enumerated; the other source paragraphs (lines 1-3 + closing fence)
    // are hidden via `shouldEnumerate`.
    #expect(
      enumeratedStarts == [0],
      "expected exactly one enumerated paragraph at offset 0, got \(enumeratedStarts)")
  }

  // MARK: - Attachment identity stability

  /// `host.ensureAttachment()` must return the same `BlockAttachment`
  /// instance on every call within the host's lifetime. This is the fix
  /// for the bug where the placeholder file-icon replaced the renderer's
  /// view on any post-initial-render selection / edit: a fresh attachment
  /// per re-vend was making AppKit drop the cached embedded view.
  @Test
  func host_ensureAttachment_returns_same_instance_within_host_lifetime() {
    let c = Self.makeComponents()
    Self.render(markdown: "```\nlet x = 1\n```", components: c)
    guard let host = BlockRendererRegistry.shared.hosts(for: c.textView).first else {
      Issue.record("expected a host to be created")
      return
    }
    let first = host.ensureAttachment()
    let second = host.ensureAttachment()
    let third = host.ensureAttachment()
    #expect(first === second, "ensureAttachment must return the same instance on repeated calls")
    #expect(second === third, "ensureAttachment must keep returning the same instance")
  }

  /// After a host is retired (its range disappears from the spec list and
  /// reconciliation disposes it) and a new host is built for the same
  /// range, the new host must vend a DIFFERENT attachment instance — the
  /// old one was tied to the disposed host.
  @Test
  func host_ensureAttachment_returns_fresh_instance_after_dispose_and_rebuild() {
    let c = Self.makeComponents()
    Self.render(markdown: "```\nlet x = 1\n```", components: c)
    let firstHost = BlockRendererRegistry.shared.hosts(for: c.textView).first
    let firstAttachment = firstHost?.ensureAttachment()
    #expect(firstAttachment != nil)

    // Drop the code block, then re-add it at the same offset. The first
    // host is disposed; the second host is a fresh instance with its own
    // attachment.
    Self.render(markdown: "plain body text", components: c)
    #expect(BlockRendererRegistry.shared.hosts(for: c.textView).isEmpty)

    Self.render(markdown: "```\nlet x = 1\n```", components: c)
    let secondHost = BlockRendererRegistry.shared.hosts(for: c.textView).first
    let secondAttachment = secondHost?.ensureAttachment()
    #expect(secondHost !== firstHost, "rebuilt host must be a fresh instance")
    #expect(
      secondAttachment !== firstAttachment,
      "fresh host must vend a fresh attachment, not reuse the disposed host's")
  }

  // MARK: - Attachment paragraph line-height pinning

  /// The U+FFFC glyph in the vended attachment paragraph must carry a
  /// paragraph style whose `minimumLineHeight` and `maximumLineHeight`
  /// equal `spec.reservedHeight`. This is the fix for the bug where the
  /// teal block drew over preceding paragraphs: TK2 doesn't auto-grow the
  /// attachment's line to match `attachmentBounds.height`, so the
  /// paragraph style has to pin the line height explicitly to reserve
  /// vertical space for the embedded view.
  @Test
  func attachment_paragraph_pins_line_height_to_reservedHeight() {
    let c = Self.makeComponents()
    let markdown = "```\nlet x = 1\nlet y = 2\n```"
    Self.render(markdown: markdown, components: c)

    guard let host = BlockRendererRegistry.shared.hosts(for: c.textView).first else {
      Issue.record("expected a host to be created")
      return
    }
    let reservedHeight = host.spec.reservedHeight
    #expect(reservedHeight > 0, "renderer should compute a non-zero reservedHeight")

    var pinnedHeight: CGFloat?
    if let tlm = c.textView.textLayoutManager {
      tlm.enumerateTextLayoutFragments(
        from: tlm.documentRange.location, options: []
      ) { frag in
        guard let paragraph = frag.textElement as? NSTextParagraph else { return true }
        let s = paragraph.attributedString.string as NSString
        // Find the U+FFFC glyph and read its paragraph style.
        for i in 0..<s.length {
          if s.character(at: i) == UInt16(0xFFFC) {
            let attrs = paragraph.attributedString.attributes(at: i, effectiveRange: nil)
            if let para = attrs[.paragraphStyle] as? NSParagraphStyle {
              pinnedHeight = para.minimumLineHeight
              #expect(
                para.minimumLineHeight == para.maximumLineHeight,
                "min and max line height must be equal to fully pin the line")
            }
            return false
          }
        }
        return true
      }
    }

    #expect(pinnedHeight != nil, "expected to find a U+FFFC glyph with a paragraph style")
    if let pinned = pinnedHeight {
      #expect(
        abs(pinned - reservedHeight) < 0.01,
        "attachment glyph paragraph style line height must equal spec.reservedHeight (got \(pinned), expected \(reservedHeight))")
    }
  }

  // MARK: - Cursor presence detection

  @Test
  func host_isCursorInside_reflects_text_view_selection() {
    let c = Self.makeComponents()
    let markdown = "before\n\n```\ncode\n```\n\nafter"
    Self.render(markdown: markdown, cursor: 0, components: c)
    let host = BlockRendererRegistry.shared.hosts(for: c.textView).first
    #expect(host != nil)

    // Cursor at position 0 → outside the code block (which starts at 8).
    #expect(host?.isCursorInside == false)

    // Move cursor inside the code block range.
    c.textView.setSelectedRange(NSRange(location: 12, length: 0))
    #expect(host?.isCursorInside == true)

    // Move back outside.
    c.textView.setSelectedRange(NSRange(location: 0, length: 0))
    #expect(host?.isCursorInside == false)
  }

  // MARK: - Selection-change notification

  /// `notifySelectionChanged` should fire `cursorPresenceChanged` only on
  /// hosts whose inside/outside state flipped relative to the registry's
  /// last-seen `lastInside`. This test installs a counting renderer to
  /// observe the dispatch.
  @Test
  func selection_change_fires_cursorPresenceChanged_only_on_transitions() {
    final class CountingRenderer: BlockRenderer {
      nonisolated(unsafe) var presenceFireCount = 0
      nonisolated(unsafe) var lastInside: Bool?
      func makeView(host: BlockRenderHost) -> NSView {
        let v = NSView(frame: .zero)
        v.wantsLayer = true
        return v
      }
      func update(spec: BlockRendererSpec, host: BlockRenderHost) {}
      func cursorPresenceChanged(_ inside: Bool, host: BlockRenderHost) {
        presenceFireCount += 1
        lastInside = inside
      }
      func desiredBounds(host: BlockRenderHost) -> CGRect { .zero }
      func tearDown() {}
    }
    let renderer = CountingRenderer()
    BlockRendererRegistry.shared.register(.codeBlock) { renderer }
    defer {
      // Restore the no-op renderer for subsequent tests.
      BlockRendererRegistry.shared.register(.codeBlock) { NoopBlockRenderer() }
    }

    let c = Self.makeComponents()
    Self.render(markdown: "before\n\n```\ncode\n```", cursor: 0, components: c)
    #expect(renderer.presenceFireCount == 0, "no presence change yet")

    // Move cursor inside the block — first transition.
    c.textView.setSelectedRange(NSRange(location: 12, length: 0))
    BlockRendererRegistry.shared.notifySelectionChanged(
      textView: c.textView, newRange: c.textView.selectedRange())
    #expect(renderer.presenceFireCount == 1)
    #expect(renderer.lastInside == true)

    // Move within the block — no transition, no fire.
    c.textView.setSelectedRange(NSRange(location: 13, length: 0))
    BlockRendererRegistry.shared.notifySelectionChanged(
      textView: c.textView, newRange: c.textView.selectedRange())
    #expect(renderer.presenceFireCount == 1, "no transition, no fire")

    // Move back out — second transition.
    c.textView.setSelectedRange(NSRange(location: 0, length: 0))
    BlockRendererRegistry.shared.notifySelectionChanged(
      textView: c.textView, newRange: c.textView.selectedRange())
    #expect(renderer.presenceFireCount == 2)
    #expect(renderer.lastInside == false)
  }
}
