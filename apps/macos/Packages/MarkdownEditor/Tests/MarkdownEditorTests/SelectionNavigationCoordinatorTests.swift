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
/// These tests pin the user-reported repro that motivated the
/// length-matching invariant in the content delegate:
/// > Source `"A **B** C D E F G"`, cursor at 5; right-arrow walks
/// > 5→6→7→8→9→10→11 with the Coordinator's apply-pipeline running on
/// > every press.
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

  // MARK: - Drift guard removed
  //
  // A drift-correction guard used to run inside the Coordinator's
  // `textViewDidChangeSelection` to restore the cursor when `apply`
  // re-snapped it to a different source offset. The drift was a symptom
  // of the content delegate vending paragraphs whose display length was
  // shorter than their source length — TK2 then computed cursor positions
  // in display coordinates and the visual position drifted relative to
  // the source position the caller set. With the length-matching invariant
  // in the content delegate (display.length == source.length via ZWSP /
  // glyph substitution), the underlying mismatch is gone and the guard
  // became unreachable. The previously-reported bug (`right_arrow` from
  // 7 jumping to 12) is now exercised by the runner end-to-end and stays
  // pinned at the test-rig level by the *_with_coordinator tests above.
}
