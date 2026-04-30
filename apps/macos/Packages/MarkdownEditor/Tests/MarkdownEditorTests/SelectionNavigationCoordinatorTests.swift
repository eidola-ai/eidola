import AppKit
import Foundation
import ObjectiveC
import SwiftUI
import Testing

@testable import MarkdownEditor

/// Production-flow selection navigation tests.
///
/// `SelectionNavigationTests` exercises `TextKit2MarkdownTextView` with the
/// content-storage delegate wired up but no `MarkdownEditor.Coordinator`
/// observing selection changes. In production the Coordinator IS the text
/// view's delegate, so `setSelectedRange(...)` (which the move overrides
/// call) fires `textViewDidChangeSelection` synchronously, which runs
/// `MarkdownRenderer.render` + `TextKit2RenderApplicator.applyCursorUpdate`
/// — and `apply` does a textStorage attribute reset, a `recordEditAction`
/// to force paragraph rebuild, and an `invalidateLayout(for: documentRange)`,
/// all inside the override before `moveRight` returns.
///
/// When the spec change toggles `**` from hidden to visible (or vice versa)
/// across the cursor, TK2's layout invalidation re-snaps the cursor to
/// preserve its visual position — which jumps it to a different source
/// offset than the one the override just set. The bug only surfaces when
/// the Coordinator is in the loop; isolated tests miss it.
///
/// These tests reproduce the user-reported bug:
/// > Source `"A **B** C D E F G"`, cursor at 5; right-arrow walks 5→6→7
/// > correctly, but the next press jumps from 7 to 12 (before "E"),
/// > skipping " C D " entirely.
@Suite("Selection Navigation (with Coordinator)")
@MainActor
struct SelectionNavigationCoordinatorTests {

  /// A test rig that mirrors the production wiring: the Coordinator is set
  /// as the text view's delegate, so `setSelectedRange` triggers
  /// `textViewDidChangeSelection` which runs `applyCursorUpdate`.
  private struct Rig {
    let textView: TextKit2MarkdownTextView
    let coordinator: MarkdownEditor.Coordinator
    let stateBox: StateBox
    let window: NSWindow
  }

  /// Backing storage for the SwiftUI `Binding<EditorState>` that the
  /// Coordinator owns. `Binding` requires get/set closures; we route them
  /// through a class so all parties see the same mutation.
  @MainActor
  private final class StateBox {
    var state: EditorState
    init(_ s: EditorState) { self.state = s }
  }

  /// Mutable counter shared with notification observers; main-actor isolated
  /// so we can mutate it from `MainActor.assumeIsolated`.
  @MainActor
  private final class FireCounter {
    var fired = 0
  }

  /// One-shot interloper installed via `NotificationCenter` Selector
  /// observation (which sidesteps the `@Sendable` closure constraint of
  /// the block-based observer API). Mutates the cursor on the FIRST
  /// `textStorage.didProcessEditingNotification` after the override sets
  /// the selection — simulating whatever production-only mechanism causes
  /// the cursor drift the user reports.
  @MainActor
  private final class CursorInterloper: NSObject {
    weak var textView: NSTextView?
    let target: Int
    let counter: FireCounter
    init(textView: NSTextView, target: Int, counter: FireCounter) {
      self.textView = textView
      self.target = target
      self.counter = counter
    }
    @objc func handle(_ notification: Notification) {
      guard counter.fired == 0 else { return }
      counter.fired += 1
      textView?.setSelectedRange(NSRange(location: target, length: 0))
    }
  }

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

    let stateBox = StateBox(
      EditorState(markdown: markdown, selection: .cursor(cursorPosition)))
    let binding = Binding<EditorState>(
      get: { stateBox.state }, set: { stateBox.state = $0 })
    let coordinator = MarkdownEditor.Coordinator(state: binding)

    // Wire the Coordinator's persistent delegates to the text view BEFORE
    // setting the text view's delegate to the Coordinator — `configure`
    // does the latter.
    textView.textContentStorage?.delegate =
      coordinator.textKit2ContentStorageDelegate
    textView.textLayoutManager?.delegate =
      coordinator.textKit2LayoutManagerDelegate
    coordinator.configure(textView)

    let window = NSWindow(
      contentRect: NSRect(origin: .zero, size: size),
      styleMask: .borderless, backing: .buffered, defer: true)
    window.contentView = textView

    // Initial sync — installs the markdown, sets the cursor, runs the
    // first render through the production path.
    coordinator.syncToTextView(stateBox.state, textView: textView)
    if let tlm = textView.textLayoutManager {
      tlm.ensureLayout(for: tlm.documentRange)
    }
    return Rig(
      textView: textView, coordinator: coordinator,
      stateBox: stateBox, window: window)
  }

  /// Press right-arrow once. The Coordinator is wired as the text view's
  /// delegate, so `moveRight` → `setSelectedRange` → (synchronously)
  /// `textViewDidChangeSelection` → `applyCursorUpdate` → `apply`. We do
  /// NOT call any extra `rerender` here — that's the whole point: the
  /// production flow is already self-contained per keypress.
  ///
  /// We force layout after the press so any TK2 visual-position-preserve
  /// snapping kicks in (matching the live editor where the layout is
  /// actually displayed every frame).
  private static func pressRight(_ rig: Rig) -> Int {
    rig.textView.moveRight(nil)
    if let tlm = rig.textView.textLayoutManager {
      tlm.ensureLayout(for: tlm.documentRange)
    }
    rig.textView.displayIfNeeded()
    return rig.textView.selectedRange().location
  }

  private static func pressLeft(_ rig: Rig) -> Int {
    rig.textView.moveLeft(nil)
    if let tlm = rig.textView.textLayoutManager {
      tlm.ensureLayout(for: tlm.documentRange)
    }
    rig.textView.displayIfNeeded()
    return rig.textView.selectedRange().location
  }

  // MARK: - Bug repro

  /// User-reported bug: source `"A **B** C D E F G"`, cursor at source 5
  /// (after "B", `**` visible because cursor inside bold). Pressing
  /// right-arrow walks 5 → 6 → 7 correctly. Then the NEXT press jumps from
  /// 7 to 12 (before "E"), skipping " C D " entirely.
  ///
  /// Without the fix this test fails at the 4th press (5→6→7→jump). With
  /// the fix the walk continues 5→6→7→8→9→10→11.
  @Test
  func right_arrow_walks_through_C_D_E_after_leaving_bold_with_coordinator() {
    let md = "A **B** C D E F G"
    let rig = Self.make(markdown: md, cursorPosition: 5)

    var visited = [5]
    for _ in 0..<6 {
      visited.append(Self.pressRight(rig))
    }
    #expect(visited == [5, 6, 7, 8, 9, 10, 11], "got: \(visited)")
  }

  /// Symmetric: walk leftward across the construct boundary. From cursor
  /// at source 11 (just before 'E'), left-arrow should walk 11 → 10 → 9 →
  /// 8 → 7 → 6 → 5 → 4. Without the fix the cursor jumps when crossing
  /// 8 → 7 (entering the bold so `**` flips from hidden to visible).
  @Test
  func left_arrow_walks_through_D_C_B_after_entering_bold_with_coordinator() {
    let md = "A **B** C D E F G"
    let rig = Self.make(markdown: md, cursorPosition: 11)

    var visited = [11]
    for _ in 0..<7 {
      visited.append(Self.pressLeft(rig))
    }
    #expect(visited == [11, 10, 9, 8, 7, 6, 5, 4], "got: \(visited)")
  }

  /// Pin the simpler case from `SelectionNavigationTests` against the
  /// production wiring too. Cursor at 1 (' ' after 'A'), bold is OUTSIDE
  /// (delimiters hidden). Right-arrow should skip the hidden run and land
  /// on 'B' at source 4. Then the cursor is INSIDE bold so the next press
  /// lands on the now-revealed `*` at source 5.
  @Test
  func right_arrow_into_bold_reveals_delimiter_with_coordinator() {
    let md = "A **B** C D E F G"
    let rig = Self.make(markdown: md, cursorPosition: 1)

    let p1 = Self.pressRight(rig)
    #expect(p1 == 4, "expected to land on 'B' at 4; got \(p1)")
    let p2 = Self.pressRight(rig)
    #expect(p2 == 5, "expected to land on revealed `*` at 5; got \(p2)")
  }

  // MARK: - Drift guard

  /// If something in the apply pipeline (TK2's visual-position-preserve
  /// heuristic, layout invalidation, or anything else) shifts the cursor
  /// during the selection-change callback, the Coordinator's drift guard
  /// must restore the cursor to the position the move override set.
  ///
  /// We can't easily force TK2 to drift the cursor in a unit test (the
  /// test rig's layout is too lightweight to trigger the visual-preserve
  /// heuristic). Instead we install an `NSTextStorage` edit-notification
  /// observer that fires *during* `apply` (which calls
  /// `textStorage.setAttributes` and triggers `didProcessEditing`) and
  /// explicitly mutates the cursor — this simulates whatever
  /// production-only mechanism causes the drift the user reports.
  ///
  /// The drift happens INSIDE the Coordinator's
  /// `textViewDidChangeSelection` callback, between when the callback
  /// captures the intended cursor and when it returns. The drift guard
  /// (re-setting `selectedRange` after `apply` returns) restores the
  /// intended position before the callback returns. Because the re-set
  /// happens while `isProcessingEvent` is still true, it does not
  /// recursively trigger another `apply`.
  @Test
  func drift_guard_restores_cursor_after_intra_apply_shift() {
    let md = "A **B** C D E F G"
    let rig = Self.make(markdown: md, cursorPosition: 7)

    // Install an interloping observer on textStorage edit completion.
    // `apply` calls `textStorage.endEditing()` which posts this
    // notification synchronously while still inside the Coordinator's
    // `textViewDidChangeSelection` callback. The interloper mutates the
    // cursor to 12 (matching the user-reported bug position), simulating
    // TK2's visual-preserve snap-back.
    let counter = FireCounter()
    let interloper = CursorInterloper(
      textView: rig.textView, target: 12, counter: counter)
    NotificationCenter.default.addObserver(
      interloper,
      selector: #selector(CursorInterloper.handle(_:)),
      name: NSTextStorage.didProcessEditingNotification,
      object: rig.textView.textStorage)
    defer { NotificationCenter.default.removeObserver(interloper) }

    // Press right-arrow. The override sets the cursor to 8. The Coordinator's
    // textViewDidChangeSelection runs, captures intendedRange=8, then runs
    // `apply` which triggers our interloper that yanks the cursor to 12.
    // The drift guard re-reads the cursor, sees it's 12 (not 8), and
    // restores it to 8 before the callback returns.
    rig.textView.moveRight(nil)
    if let tlm = rig.textView.textLayoutManager {
      tlm.ensureLayout(for: tlm.documentRange)
    }
    rig.textView.displayIfNeeded()

    #expect(counter.fired == 1, "interloper must fire to exercise the drift guard")
    let final = rig.textView.selectedRange().location
    #expect(
      final == 8,
      "drift guard should restore cursor to 8 after interloping shift to 12; got \(final)")
  }
}
