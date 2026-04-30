import AppKit
import Foundation
import ObjectiveC
import SwiftUI
import Testing

@testable import MarkdownEditor

/// Pins the scrollbar-jump regression: after a long markdown document is
/// loaded, user interactions that cause the cursor to land outside the
/// currently-laid-out viewport (clicks, up/down arrows, word/line jump
/// shortcuts) must NOT change the document's `usageBoundsForTextContainer`
/// height.
///
/// Background: TK2 lays out paragraphs lazily. Paragraphs that haven't
/// reached the viewport sit at `NSTextLayoutFragment.State.estimatedUsageBounds`,
/// where height is a font-metric-derived estimate. When an interaction
/// triggers `ensureLayout` for previously-estimated paragraphs, their
/// heights snap from estimated to actual. The cumulative document height
/// changes — `NSScrollView` sees the new bounds and updates the scroller
/// thumb, which the user perceives as a "jump."
///
/// The fix: after every `apply` we eagerly lay out the full document so
/// every fragment reaches `.layoutAvailable` before any subsequent
/// interaction can expose an estimation gap.
@Suite("Scrollbar Stability")
@MainActor
struct ScrollbarStabilityTests {

  // MARK: - Test rig

  private struct Rig {
    let textView: TextKit2MarkdownTextView
    let coordinator: MarkdownEditor.Coordinator
    let stateBox: StateBox
    let window: NSWindow
  }

  @MainActor
  private final class StateBox {
    var state: EditorState
    init(_ s: EditorState) { self.state = s }
  }

  /// A long-document rig: 60+ paragraphs of body text with mixed
  /// constructs (headings, lists, blockquotes) so the per-paragraph
  /// height estimate is more likely to differ from the laid-out height.
  /// `viewportSize` is small enough that only the first handful of
  /// paragraphs fall in the viewport, leaving the rest in
  /// `.estimatedUsageBounds` until they're realized.
  private static func make(
    markdown: String,
    cursorPosition: Int = 0,
    viewportSize: NSSize = NSSize(width: 600, height: 200)
  ) -> Rig {
    let tv = NSTextView(usingTextLayoutManager: true)
    object_setClass(tv, TextKit2MarkdownTextView.self)
    let textView = tv as! TextKit2MarkdownTextView
    textView.frame = NSRect(origin: .zero, size: viewportSize)
    textView.minSize = NSSize(width: 0, height: 0)
    textView.maxSize = NSSize(
      width: CGFloat.greatestFiniteMagnitude,
      height: CGFloat.greatestFiniteMagnitude)
    textView.isVerticallyResizable = true
    textView.font = MarkdownStyle.default.baseFont
    textView.isRichText = true
    textView.isAutomaticQuoteSubstitutionEnabled = false
    textView.textContainer?.containerSize = NSSize(
      width: viewportSize.width, height: CGFloat.greatestFiniteMagnitude)
    textView.textContainer?.widthTracksTextView = true

    let stateBox = StateBox(
      EditorState(markdown: markdown, selection: .cursor(cursorPosition)))
    let binding = Binding<EditorState>(
      get: { stateBox.state }, set: { stateBox.state = $0 })
    let coordinator = MarkdownEditor.Coordinator(state: binding)

    textView.textContentStorage?.delegate =
      coordinator.textKit2ContentStorageDelegate
    textView.textLayoutManager?.delegate =
      coordinator.textKit2LayoutManagerDelegate
    coordinator.configure(textView)

    let window = NSWindow(
      contentRect: NSRect(origin: .zero, size: viewportSize),
      styleMask: .borderless, backing: .buffered, defer: true)
    window.contentView = textView

    // Sync — runs the first render through the production path.
    coordinator.syncToTextView(stateBox.state, textView: textView)

    return Rig(
      textView: textView, coordinator: coordinator,
      stateBox: stateBox, window: window)
  }

  /// 60 body paragraphs separated by blank lines. Long enough that with a
  /// small viewport (200pt) the lazy-layout estimate vs actual gap will
  /// produce visible bounds drift if the apply path doesn't force layout.
  private static func longDocument() -> String {
    var lines: [String] = []
    for i in 1...60 {
      lines.append("Paragraph \(i): The quick brown fox jumps over the lazy dog. " +
        "Pack my box with five dozen liquor jugs. " +
        "How vexingly quick daft zebras jump.")
    }
    return lines.joined(separator: "\n\n")
  }

  // MARK: - Hypothesis verification (regression repro)

  /// After the initial `apply`, the layout manager's
  /// `usageBoundsForTextContainer` should already reflect the full,
  /// post-layout document height — not an estimate. If the height changes
  /// after subsequent cursor moves, the lazy-layout gap is exposed.
  @Test
  func usage_bounds_stable_after_initial_apply() {
    let rig = Self.make(markdown: Self.longDocument())
    guard let tlm = rig.textView.textLayoutManager else {
      Issue.record("no text layout manager")
      return
    }
    let h0 = tlm.usageBoundsForTextContainer.height
    // Force full layout — this is what the fix should already have done.
    tlm.ensureLayout(for: tlm.documentRange)
    let h1 = tlm.usageBoundsForTextContainer.height
    let msg = "usageBoundsForTextContainer height changed after explicit ensureLayout: before=\(h0) after=\(h1) — proves apply did not force full layout"
    #expect(abs(h1 - h0) < 0.5, Comment(rawValue: msg))
  }

  /// Set the cursor far down in the document (simulating a click far from
  /// the initial viewport). The bounds height MUST NOT change as the
  /// previously-estimated paragraphs realize.
  @Test
  func usage_bounds_stable_after_far_cursor_move() {
    let md = Self.longDocument()
    let rig = Self.make(markdown: md, cursorPosition: 0)
    guard let tlm = rig.textView.textLayoutManager else {
      Issue.record("no text layout manager")
      return
    }
    let initialHeight = tlm.usageBoundsForTextContainer.height
    // Pick a cursor position deep in the document — well past the
    // initial 200pt viewport.
    let mdLen = (md as NSString).length
    let farPos = mdLen - 200
    rig.textView.setSelectedRange(NSRange(location: farPos, length: 0))
    let postMoveHeight = tlm.usageBoundsForTextContainer.height
    let msg = "usageBoundsForTextContainer height changed after far cursor move: initial=\(initialHeight) post=\(postMoveHeight) — scrollbar would jump"
    #expect(abs(postMoveHeight - initialHeight) < 0.5, Comment(rawValue: msg))
  }

  /// Walk the cursor via `moveDown` several times. Each move can expose
  /// previously-estimated paragraphs as the cursor descends. Bounds must
  /// stay stable.
  @Test
  func usage_bounds_stable_during_arrow_walk_down() {
    let md = Self.longDocument()
    let rig = Self.make(markdown: md, cursorPosition: 0)
    guard let tlm = rig.textView.textLayoutManager else {
      Issue.record("no text layout manager")
      return
    }
    let initialHeight = tlm.usageBoundsForTextContainer.height
    var heights: [CGFloat] = [initialHeight]
    for _ in 0..<40 {
      rig.textView.moveDown(nil)
      heights.append(tlm.usageBoundsForTextContainer.height)
    }
    let maxDelta = heights.map { abs($0 - initialHeight) }.max() ?? 0
    let msg = "usageBoundsForTextContainer height drifted during moveDown walk: initial=\(initialHeight) heights=\(heights) maxDelta=\(maxDelta)"
    #expect(maxDelta < 0.5, Comment(rawValue: msg))
  }

  /// Word-jump (Option+Right) a bunch of times: same invariant.
  @Test
  func usage_bounds_stable_during_word_walk() {
    let md = Self.longDocument()
    let rig = Self.make(markdown: md, cursorPosition: 0)
    guard let tlm = rig.textView.textLayoutManager else {
      Issue.record("no text layout manager")
      return
    }
    let initialHeight = tlm.usageBoundsForTextContainer.height
    var heights: [CGFloat] = [initialHeight]
    for _ in 0..<80 {
      rig.textView.moveWordRight(nil)
      heights.append(tlm.usageBoundsForTextContainer.height)
    }
    let maxDelta = heights.map { abs($0 - initialHeight) }.max() ?? 0
    let msg = "usageBoundsForTextContainer height drifted during moveWordRight walk: initial=\(initialHeight) maxDelta=\(maxDelta)"
    #expect(maxDelta < 0.5, Comment(rawValue: msg))
  }

  /// Cost ceiling: a single `apply` over a moderately-sized document
  /// (60 paragraphs ≈ 8KB) should complete well within a couple of frames.
  /// This guards against order-of-magnitude regressions (e.g. accidentally
  /// quadratic layout); it is NOT a strict frame-budget assertion. Headless
  /// XCTest layout has enough run-to-run variance that a 16ms threshold
  /// flakes; 32ms (~2 frames) still catches anything that would visibly
  /// stutter while tolerating the variance.
  @Test
  func apply_with_full_document_layout_within_perf_envelope() {
    let md = Self.longDocument()
    let rig = Self.make(markdown: md, cursorPosition: 0)
    // Drive a few cursor moves and time the full apply path. Use the 95th
    // percentile so a single cold-cache outlier doesn't fail the test.
    var samples: [TimeInterval] = []
    for offset in stride(from: 100, to: 2000, by: 200) {
      let start = Date()
      rig.textView.setSelectedRange(NSRange(location: offset, length: 0))
      samples.append(Date().timeIntervalSince(start))
    }
    let sorted = samples.sorted()
    let p95 = sorted[Int(Double(sorted.count) * 0.95)]
    let msg = "apply pipeline P95 cursor-move cost \(p95 * 1000)ms is unexpectedly high; samples=\(samples.map { Int($0 * 1000) })ms"
    #expect(p95 < 0.032, Comment(rawValue: msg))
  }

  /// Plain left/right arrow walk should ALSO not drift (regression guard
  /// — the original report said these specific keys don't trigger the
  /// jump, which suggests they don't trigger the gap; but the fix should
  /// make it true uniformly).
  @Test
  func usage_bounds_stable_during_left_right_walk() {
    let md = Self.longDocument()
    let rig = Self.make(markdown: md, cursorPosition: 0)
    guard let tlm = rig.textView.textLayoutManager else {
      Issue.record("no text layout manager")
      return
    }
    let initialHeight = tlm.usageBoundsForTextContainer.height
    var heights: [CGFloat] = [initialHeight]
    for _ in 0..<40 {
      rig.textView.moveRight(nil)
      heights.append(tlm.usageBoundsForTextContainer.height)
    }
    let maxDelta = heights.map { abs($0 - initialHeight) }.max() ?? 0
    let msg = "usageBoundsForTextContainer height drifted during moveRight walk: initial=\(initialHeight) maxDelta=\(maxDelta)"
    #expect(maxDelta < 0.5, Comment(rawValue: msg))
  }
}
