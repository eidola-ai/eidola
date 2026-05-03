import AppKit
import ObjectiveC
import SwiftUI
import Testing

@testable import MarkdownEditor

/// Phase 2.2 of the Block Renderer migration — validates the real
/// `CodeBlockRenderer` (an `NSScrollView { NSTextView }` subtree
/// anchored to a `BlockAttachment`).
///
/// These tests build a TK2 NSTextView inside a real NSWindow so the
/// renderer's first-responder swap and the embedded view's storage
/// mirror are end-to-end-exercisable. The same shape as
/// `BlockRendererInfraTests` plus a `Coordinator`-equivalent slice for
/// the deinit-cleanup test.
///
/// `.serialized` because every test in the suite installs / asserts on
/// the global `BlockRendererRegistry.shared` factory for `.codeBlock`,
/// and other suites (`BlockRendererInfraTests`) temporarily swap that
/// factory for instrumented variants. With the parallel-by-default
/// runner one suite's swap can race another's reads. Serializing within
/// this suite plus re-installing the production factory at every entry
/// point is a defensive belt-and-braces against the cross-suite
/// singleton sharing.
@MainActor
@Suite(.serialized)
struct CodeBlockRendererTests {

  /// Force the production `CodeBlockRenderer` factory back into the
  /// shared registry. Other suites (notably `BlockRendererInfraTests`)
  /// install instrumented factories with a `defer` to restore — but
  /// concurrent execution can interleave so a host built mid-swap holds
  /// the wrong renderer. Calling this at the top of every test in the
  /// suite makes the renderer's identity deterministic regardless of
  /// what other tests are doing.
  private static func installProductionRendererFactory() {
    BlockRendererRegistry.shared.register(.codeBlock) { CodeBlockRenderer() }
  }

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
    // Make the window key so `NSTextView.shouldChangeText(...)` doesn't
    // bail on input-context preconditions (it requires an editable view
    // attached to a key-window first responder chain to vouch for an
    // edit). Without this the renderer's upstream-sync path early-returns
    // and the embedded edit never propagates to main storage.
    window.makeKeyAndOrderFront(nil)
    window.makeFirstResponder(textView)

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
    markdown: String, cursor: Int = 0, components: TK2Components
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
    // Notify selection change so the registry's cursor-presence transitions
    // fire on hosts whose inside/outside state flipped — same pipeline the
    // production Coordinator drives.
    BlockRendererRegistry.shared.notifySelectionChanged(
      textView: textView, newRange: textView.selectedRange())
  }

  /// Walk the host's view subtree and find the embedded `NSTextView`
  /// (the renderer's `EmbeddedCodeTextView`). Returns `nil` if the host's
  /// view tree hasn't materialized yet.
  private static func embeddedTextView(in host: BlockRenderHost) -> NSTextView? {
    func walk(_ view: NSView?) -> NSTextView? {
      guard let view else { return nil }
      if let tv = view as? NSTextView, !(tv is TextKit2MarkdownTextView) {
        // The main view is `TextKit2MarkdownTextView`; the embedded view
        // is the renderer's own `EmbeddedCodeTextView` subclass. Either
        // way, anything that's an `NSTextView` but not the main subclass
        // is the embedded view.
        return tv
      }
      for child in view.subviews {
        if let found = walk(child) { return found }
      }
      return nil
    }
    return walk(host.view)
  }

  /// Force the host to materialize its view by asking the host to ensure
  /// its view exists. AppKit normally only builds the view when TK2 asks
  /// the attachment for its view provider during layout; tests that
  /// inspect the embedded view need the view eagerly.
  private static func forceHostView(_ host: BlockRenderHost, components: TK2Components) -> NSView {
    let v = host.ensureView()
    components.window.contentView?.addSubview(v)
    v.frame = NSRect(x: 0, y: 0, width: 400, height: 80)
    return v
  }

  // MARK: - Embedded view content mirrors source

  @Test
  func embedded_view_content_matches_source_text_after_update() {
    Self.installProductionRendererFactory()
    let c = Self.makeComponents()
    let markdown = "```\nlet x = 42\nlet y = 7\n```"
    Self.render(markdown: markdown, components: c)

    guard let host = BlockRendererRegistry.shared.hosts(for: c.textView).first else {
      Issue.record("expected one host for the code block")
      return
    }
    _ = Self.forceHostView(host, components: c)
    // Drive an explicit `update` in case `forceHostView` ran before the
    // applicator did.
    host.renderer?.update(spec: host.spec, host: host)

    guard let embedded = Self.embeddedTextView(in: host) else {
      Issue.record("expected an embedded NSTextView inside the host's view tree")
      return
    }
    let expected = host.sourceText()
    let actual = embedded.string
    #expect(
      actual == expected,
      "embedded view content (\(actual.debugDescription)) must mirror host.sourceText() (\(expected.debugDescription))")
  }

  // MARK: - First-responder swap on cursor enter / leave

  @Test
  func cursor_entering_block_swaps_first_responder_to_embedded_view() {
    Self.installProductionRendererFactory()
    let c = Self.makeComponents()
    let markdown = "before\n\n```\ncode\n```\n\nafter"
    Self.render(markdown: markdown, cursor: 0, components: c)

    guard let host = BlockRendererRegistry.shared.hosts(for: c.textView).first else {
      Issue.record("expected one host for the code block")
      return
    }
    _ = Self.forceHostView(host, components: c)
    // First update so the renderer's `hasAppliedAtLeastOnce` flag is set
    // (otherwise `cursorPresenceChanged` is a no-op on first call, by
    // design — we don't steal focus on initial render).
    host.renderer?.update(spec: host.spec, host: host)

    // Ensure the main text view starts as first responder.
    c.textView.window?.makeFirstResponder(c.textView)
    #expect(c.textView.window?.firstResponder === c.textView)

    // Move the cursor into the block range and notify the registry.
    c.textView.setSelectedRange(NSRange(location: 12, length: 0))
    BlockRendererRegistry.shared.notifySelectionChanged(
      textView: c.textView, newRange: c.textView.selectedRange())

    let embedded = Self.embeddedTextView(in: host)
    #expect(embedded != nil, "expected the embedded text view to exist")
    if let embedded {
      // AppKit's first-responder query may walk back to the field editor
      // for an NSTextView; check identity directly.
      let fr = c.textView.window?.firstResponder
      #expect(
        fr === embedded,
        "first responder should be the embedded view after cursor enters the code block (was \(String(describing: fr)))")
    }
  }

  @Test
  func cursor_leaving_block_swaps_first_responder_back_to_main_view() {
    Self.installProductionRendererFactory()
    let c = Self.makeComponents()
    let markdown = "before\n\n```\ncode\n```\n\nafter"
    Self.render(markdown: markdown, cursor: 0, components: c)

    guard let host = BlockRendererRegistry.shared.hosts(for: c.textView).first else {
      Issue.record("expected one host for the code block")
      return
    }
    _ = Self.forceHostView(host, components: c)
    host.renderer?.update(spec: host.spec, host: host)

    // Enter the block first.
    c.textView.window?.makeFirstResponder(c.textView)
    c.textView.setSelectedRange(NSRange(location: 12, length: 0))
    BlockRendererRegistry.shared.notifySelectionChanged(
      textView: c.textView, newRange: c.textView.selectedRange())

    // Now leave the block.
    c.textView.setSelectedRange(NSRange(location: 0, length: 0))
    BlockRendererRegistry.shared.notifySelectionChanged(
      textView: c.textView, newRange: c.textView.selectedRange())

    let fr = c.textView.window?.firstResponder
    #expect(
      fr === c.textView,
      "first responder should swap back to the main text view after cursor exits the code block (was \(String(describing: fr)))")
  }

  // MARK: - Edit-in-place forwards to main storage

  @Test
  func typing_in_embedded_view_modifies_main_storage_at_mapped_offset() {
    Self.installProductionRendererFactory()
    let c = Self.makeComponents()
    let markdown = "```\nlet x = 1\n```"
    Self.render(markdown: markdown, cursor: 0, components: c)

    guard let host = BlockRendererRegistry.shared.hosts(for: c.textView).first else {
      Issue.record("expected one host for the code block")
      return
    }
    _ = Self.forceHostView(host, components: c)
    host.renderer?.update(spec: host.spec, host: host)

    guard let embedded = Self.embeddedTextView(in: host),
      let renderer = host.renderer as? CodeBlockRenderer
    else {
      Issue.record("expected an embedded text view + a CodeBlockRenderer")
      return
    }

    // Install a delegate proxy on the MAIN text view so we can verify
    // the renderer's upstream edit fires `textDidChange` on the main
    // view's delegate (the production Coordinator's apply pipeline runs
    // off this notification — without it the SwiftUI binding stays
    // stale until the user types in the main view directly). Pre-Phase
    // 2.2-followup the renderer mutated `mainStorage.replaceCharacters`
    // directly which bypassed the delegate chain entirely.
    let proxy = TextDidChangeRecorder()
    c.textView.delegate = proxy

    // Simulate the embedded view asking permission to insert "Z" at
    // local offset 4 (just inside the content, after "let "). The
    // delegate is the renderer (it was set in `makeView`); it should
    // forward the change to the main storage at offset 4 + spec.range.location.
    let localRange = NSRange(location: 4, length: 0)
    let allowed = renderer.textView(
      embedded, shouldChangeTextIn: localRange, replacementString: "Z")

    #expect(
      allowed == true,
      "renderer should allow the local edit (embedded view is the editing surface for its range; the change is forwarded synchronously to main storage via the `isForwardingLocalEdit` re-entrancy guard)")

    let main = c.textView.string
    // Main storage's offset 4 is the `l` of "let"; after inserting "Z"
    // there it should be "```\nZlet x = 1\n```".
    #expect(
      main == "```\nZlet x = 1\n```",
      "main storage should reflect the embedded edit (got \(main.debugDescription))")
    let caret = c.textView.selectedRange().location
    #expect(caret == 5, "main caret should sit one past the inserted Z (got \(caret))")
    #expect(
      proxy.textDidChangeCount >= 1,
      "main view's delegate must receive textDidChange so the Coordinator's apply loop / EditorState binding stays in sync (got \(proxy.textDidChangeCount))")
  }

  @Test
  func mainview_didChangeText_fires_after_embedded_edit() {
    Self.installProductionRendererFactory()
    let c = Self.makeComponents()
    let markdown = "```\nlet x = 1\n```"
    Self.render(markdown: markdown, cursor: 0, components: c)

    guard let host = BlockRendererRegistry.shared.hosts(for: c.textView).first else {
      Issue.record("expected one host for the code block")
      return
    }
    _ = Self.forceHostView(host, components: c)
    host.renderer?.update(spec: host.spec, host: host)

    guard let embedded = Self.embeddedTextView(in: host),
      let renderer = host.renderer as? CodeBlockRenderer
    else {
      Issue.record("expected an embedded text view + a CodeBlockRenderer")
      return
    }

    let proxy = TextDidChangeRecorder()
    c.textView.delegate = proxy

    let localRange = NSRange(location: 4, length: 0)
    _ = renderer.textView(
      embedded, shouldChangeTextIn: localRange, replacementString: "Q")

    #expect(
      proxy.textDidChangeCount == 1,
      "exactly one textDidChange should fire on the main view's delegate per embedded keystroke (got \(proxy.textDidChangeCount))")
  }

  // MARK: - Intrinsic content height feedback

  @Test
  func embedded_view_height_growth_drives_attachment_bounds() {
    Self.installProductionRendererFactory()
    let c = Self.makeComponents()
    let markdown = "```\nx\n```"
    Self.render(markdown: markdown, cursor: 0, components: c)

    guard let host = BlockRendererRegistry.shared.hosts(for: c.textView).first else {
      Issue.record("expected one host for the code block")
      return
    }
    _ = Self.forceHostView(host, components: c)
    host.renderer?.update(spec: host.spec, host: host)

    guard let embedded = Self.embeddedTextView(in: host),
      let renderer = host.renderer as? CodeBlockRenderer
    else {
      Issue.record("expected an embedded text view + a CodeBlockRenderer")
      return
    }
    let attachment = host.ensureAttachment()

    // Snapshot the initial intrinsic height (may be nil before first
    // measurement, or already equal to the embedded view's single-line
    // `usedRect` height after `update()` ran). Either way, the
    // attachment should report a non-zero, finite bound.
    let baselineHeight = host.intrinsicContentHeight ?? host.spec.reservedHeight
    #expect(baselineHeight > 0, "baseline height must be > 0")

    // Programmatically grow the embedded view by writing several lines
    // straight into its storage and asking the renderer to re-measure.
    // We're testing the measurement / publish path here, not the
    // shouldChangeTextIn forwarding path (which is covered by the
    // mainview_didChangeText test); the cleanest isolation is to write
    // multi-line content directly into the embedded view's storage so
    // its `usedRect.height` clearly exceeds the single-line baseline,
    // then drive the renderer's measurement hook.
    if let storage = embedded.textStorage {
      let multiLine = NSAttributedString(
        string: "x\nA\nB\nC\nD",
        attributes: [.font: MarkdownStyle.default.codeFont])
      storage.beginEditing()
      storage.setAttributedString(multiLine)
      storage.endEditing()
    }
    if let lm = embedded.layoutManager, let container = embedded.textContainer {
      lm.ensureLayout(for: container)
    }
    renderer.measureAndPublishHeight()

    let grownHeight = host.intrinsicContentHeight ?? -1
    #expect(
      grownHeight > baselineHeight,
      "intrinsicContentHeight should grow after inserting lines (baseline \(baselineHeight) → grown \(grownHeight))")

    // The attachment's reported bounds should follow the host's
    // intrinsic height — that's what makes the outer flow reflow.
    let bounds = attachment.attachmentBounds(
      for: c.textView.textContainer,
      proposedLineFragment: NSRect(x: 0, y: 0, width: 600, height: 20),
      glyphPosition: .zero, characterIndex: 0)
    #expect(
      abs(bounds.height - grownHeight) < 0.5,
      "attachment.attachmentBounds.height (\(bounds.height)) must match host.intrinsicContentHeight (\(grownHeight))")
  }

  @Test
  func intrinsic_height_falls_back_to_reservedHeight_before_first_measurement() {
    // Build a fresh host without ever invoking `update` /
    // `shouldChangeTextIn` — `intrinsicContentHeight` should stay nil
    // and the attachment must fall back to `spec.reservedHeight` (NOT
    // zero or nil-yields-zero). Without the fallback the very first
    // attachment-bounds query (which TK2 may issue before AppKit ever
    // mounts the view-provider's view) would collapse the line to
    // height zero and the outer flow would lay out as if the block
    // didn't exist.
    let textView = NSTextView(usingTextLayoutManager: true)
    textView.frame = NSRect(x: 0, y: 0, width: 600, height: 400)
    let spec = BlockRendererSpec(
      range: NSRange(location: 0, length: 10),
      blockTypeTag: .codeBlock,
      mode: .editInPlace,
      reservedHeight: 73)
    let host = BlockRenderHost(
      spec: spec, textView: textView, rendererFactory: { CodeBlockRenderer() })
    #expect(
      host.intrinsicContentHeight == nil,
      "fresh host must not have a measured intrinsic height yet")
    let attachment = host.ensureAttachment()
    let bounds = attachment.attachmentBounds(
      for: nil,
      proposedLineFragment: NSRect(x: 0, y: 0, width: 600, height: 20),
      glyphPosition: .zero, characterIndex: 0)
    #expect(
      abs(bounds.height - 73) < 0.5,
      "attachment must fall back to spec.reservedHeight (73) before the renderer has measured (got \(bounds.height))")
  }

  // MARK: - Long single line provides horizontal scroll capacity

  @Test
  func long_single_line_produces_horizontally_scrollable_embedded_view() {
    Self.installProductionRendererFactory()
    let c = Self.makeComponents(size: NSSize(width: 200, height: 200))
    let longLine = String(repeating: "x", count: 400)
    let markdown = "```\n\(longLine)\n```"
    Self.render(markdown: markdown, components: c)

    guard let host = BlockRendererRegistry.shared.hosts(for: c.textView).first else {
      Issue.record("expected a host for the code block")
      return
    }
    _ = Self.forceHostView(host, components: c)
    host.renderer?.update(spec: host.spec, host: host)

    guard let embedded = Self.embeddedTextView(in: host) else {
      Issue.record("expected an embedded text view")
      return
    }
    // The embedded text container is configured with an unbounded width
    // (`widthTracksTextView = false`, container.size.width =
    // .greatestFiniteMagnitude) so the no-wrap path can lay out a long
    // line beyond the visible scroll-view width. We assert the
    // configuration directly because measuring the laid-out text width
    // depends on font glyph metrics that aren't load-bearing for the
    // invariant.
    let container = embedded.textContainer
    #expect(
      container?.widthTracksTextView == false,
      "embedded text container must not track view width (would clamp horizontal extent)")
    if let container {
      #expect(
        container.size.width >= 1_000,
        "embedded text container width must be effectively unbounded for horizontal scroll (got \(container.size.width))")
    }
  }

  // MARK: - Reconcile dedup

  @Test
  func reconciling_same_spec_twice_executes_reconcile_only_once() {
    let c = Self.makeComponents()
    let markdown = "```\nlet x = 1\n```"
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: markdown, cursorRange: cursorRange)
    c.textView.string = markdown
    c.textView.setSelectedRange(cursorRange)

    // Drop any state from earlier tests so we measure deltas precisely.
    BlockRendererRegistry.shared.dropAll(for: c.textView)
    let baseline = BlockRendererRegistry.shared.reconcileExecutionCount

    BlockRendererRegistry.shared.reconcileIfChanged(
      for: c.textView, specs: spec.blockRendererSpecs)
    let afterFirst = BlockRendererRegistry.shared.reconcileExecutionCount
    #expect(
      afterFirst == baseline + 1,
      "first reconcile against a fresh text view must execute (delta \(afterFirst - baseline))")

    // Apply the SAME spec again — dedup short-circuit should fire.
    BlockRendererRegistry.shared.reconcileIfChanged(
      for: c.textView, specs: spec.blockRendererSpecs)
    let afterSecond = BlockRendererRegistry.shared.reconcileExecutionCount
    #expect(
      afterSecond == afterFirst,
      "second reconcile with the SAME spec list must short-circuit (delta \(afterSecond - afterFirst))")

    // Apply a different spec — reconcile must execute.
    let nextMarkdown = "para\n\n```\nlet x = 1\n```"
    let nextSpec = MarkdownRenderer.render(
      text: nextMarkdown, cursorRange: NSRange(location: 0, length: 0))
    c.textView.string = nextMarkdown
    BlockRendererRegistry.shared.reconcileIfChanged(
      for: c.textView, specs: nextSpec.blockRendererSpecs)
    let afterThird = BlockRendererRegistry.shared.reconcileExecutionCount
    #expect(
      afterThird == afterSecond + 1,
      "reconcile with a CHANGED spec list must execute (delta \(afterThird - afterSecond))")
  }

  // MARK: - Coordinator deinit drops registry hosts

  @Test
  func coordinator_deinit_drops_registry_hosts_for_bound_text_view() {
    let textView = NSTextView(usingTextLayoutManager: true)
    object_setClass(textView, TextKit2MarkdownTextView.self)
    textView.frame = NSRect(x: 0, y: 0, width: 600, height: 400)

    // Prime the registry with a host for this text view, mimicking what
    // the production applicator would do on first apply.
    let spec = BlockRendererSpec(
      range: NSRange(location: 0, length: 10),
      blockTypeTag: .codeBlock,
      mode: .editInPlace,
      reservedHeight: 50)
    BlockRendererRegistry.shared.reconcile(for: textView, specs: [spec])
    #expect(
      BlockRendererRegistry.shared.hosts(for: textView).count == 1,
      "expected one host after priming the registry")

    // Build a Coordinator wrapped in a weak observer so we can detect
    // ARC deallocation precisely (Swift doesn't guarantee that a `do`
    // block's locals release at the closing brace — the optimizer may
    // extend the lifetime to the end of the function). The closure
    // pattern + an explicit weak observer is the canonical workaround.
    weak var weakCoordinator: MarkdownEditor.Coordinator?
    autoreleasepool {
      let coordinator = MarkdownEditor.Coordinator(
        state: .constant(EditorState(markdown: "", selection: .cursor(0))))
      coordinator.configure(textView)
      weakCoordinator = coordinator
      // `coordinator` is the only strong ref; on autoreleasepool exit
      // ARC drops it.
    }
    #expect(
      weakCoordinator == nil,
      "Coordinator must deallocate at autoreleasepool exit; otherwise the deinit hasn't run and the post-condition can't hold")

    // After the Coordinator deallocates, the registry's hosts for this
    // text view should be dropped — the `deinit` cleanup wired in
    // Phase 2.2 follow-up D.1.
    #expect(
      BlockRendererRegistry.shared.hosts(for: textView).isEmpty,
      "expected zero hosts after Coordinator deinit; registry leak otherwise")
  }
}

/// Stub `NSTextViewDelegate` that records `textDidChange` invocations.
/// Used by tests that need to verify the renderer's upstream edit fires
/// the main view's delegate chain (which is what drives the production
/// Coordinator's apply pipeline / `EditorState.markdown` binding).
@MainActor
private final class TextDidChangeRecorder: NSObject, NSTextViewDelegate {
  var textDidChangeCount: Int = 0
  func textDidChange(_ notification: Notification) {
    textDidChangeCount += 1
  }
}
