import AppKit
import Foundation
import ObjectiveC
import SwiftUI
import Testing

@testable import MarkdownEditor

/// Audit tests for the deferred-interaction issues opened against the
/// `MarkdownEditor` package after the ZWSP-substitution refactor restored the
/// `display.length == source.length` invariant.
///
/// Each test pins behavior identified in the audit:
///   1. `deleteBackward` at the visual start of an ATX heading (or any other
///      hidden-prefix paragraph) deletes the entire prefix as a single unit
///      — consistent with how `- `, `> `, `[ ]`, etc. are already handled.
///   2. Copy across hidden inline delimiters preserves the literal source
///      markdown (no ZWSPs leak into the clipboard).
///   3. Programmatic selection across hidden ranges round-trips through
///      `setSelectedRange` / `selectedRange()` without TK2 snapping or
///      clamping.
///   4. Shift+arrow that REVERSES direction mid-stream tracks the original
///      anchor explicitly (regression: the prior heuristic guessed the
///      anchor from the selection bounds and grew the selection in the
///      wrong direction).
///   5. Word-level double-click selection in `A **B** C` selects just the
///      visible word "B" (not the surrounding `**`).
@Suite("Deferred Interaction Audit")
@MainActor
struct DeferredInteractionAuditTests {

  // MARK: - Test rig (mirrors SelectionNavigationCoordinatorTests)

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

    textView.textContentStorage?.delegate =
      coordinator.textKit2ContentStorageDelegate
    textView.textLayoutManager?.delegate =
      coordinator.textKit2LayoutManagerDelegate
    coordinator.configure(textView)

    let window = NSWindow(
      contentRect: NSRect(origin: .zero, size: size),
      styleMask: .borderless, backing: .buffered, defer: true)
    window.contentView = textView

    coordinator.syncToTextView(stateBox.state, textView: textView)
    if let tlm = textView.textLayoutManager {
      tlm.ensureLayout(for: tlm.documentRange)
    }
    return Rig(
      textView: textView, coordinator: coordinator,
      stateBox: stateBox, window: window)
  }

  // MARK: - Item 1: deleteBackward at heading prefix deletes whole unit

  /// `# Heading\n\nbody` cursor at source 2 (visual start of "Heading").
  /// The hidden `# ` is a delimiter pair; per Goal #2 ("invisible characters
  /// MUST be handled thoughtfully") backspace at the visual line start
  /// demotes the heading to a body paragraph in one step rather than
  /// silently deleting the hidden space first.
  @Test
  func deleteBackward_at_heading_text_start_removes_whole_prefix() {
    let state = EditorState(markdown: "# Heading\n\nbody", selection: .cursor(2))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    #expect(result.markdown == "Heading\n\nbody")
    #expect(result.selection == .cursor(0))
  }

  @Test
  func deleteBackward_at_h2_heading_text_start_removes_whole_prefix() {
    let state = EditorState(markdown: "## Subhead\n\nbody", selection: .cursor(3))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    #expect(result.markdown == "Subhead\n\nbody")
    #expect(result.selection == .cursor(0))
  }

  @Test
  func deleteBackward_at_h3_heading_text_start_removes_whole_prefix() {
    let state = EditorState(markdown: "### Sub\n\nbody", selection: .cursor(4))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    #expect(result.markdown == "Sub\n\nbody")
    #expect(result.selection == .cursor(0))
  }

  /// Heading with no body — same whole-unit deletion.
  @Test
  func deleteBackward_at_heading_text_start_no_body_removes_whole_prefix() {
    let state = EditorState(markdown: "# Heading\n", selection: .cursor(2))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    #expect(result.markdown == "Heading\n")
    #expect(result.selection == .cursor(0))
  }

  /// Backspace ELSEWHERE inside a heading still deletes one char (we only
  /// shortcut at the prefix boundary).
  @Test
  func deleteBackward_inside_heading_content_still_deletes_one_char() {
    let state = EditorState(markdown: "# Heading\n", selection: .cursor(5))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    #expect(result.markdown == "# Heding\n")
    #expect(result.selection == .cursor(4))
  }

  /// `# ` followed by a digit at line start should NOT trigger ATX deletion
  /// when the cursor is past the content — only the `posInLine == prefixLen`
  /// boundary qualifies.
  @Test
  func deleteBackward_past_heading_content_uses_default_path() {
    // `# Heading`, cursor at the end of "Heading" (position 9). Single-char
    // deletion: removes the trailing 'g'.
    let state = EditorState(markdown: "# Heading", selection: .cursor(9))
    let result = EditorUpdate.update(state, event: .deleteBackward)
    #expect(result.markdown == "# Headin")
    #expect(result.selection == .cursor(8))
  }

  // MARK: - Item 2: Copy across hidden inline delimiters preserves source

  /// `A **B** C`, select source range [0, 7) — `A **B**`. The
  /// `NSTextView.writeSelection(to:types:)` path reads from `textStorage`
  /// (the source) so the clipboard MUST contain the literal `A **B**`,
  /// not the displayed `A \u{200B}\u{200B}B\u{200B}\u{200B}` with ZWSPs
  /// leaked in.
  @Test
  func copy_across_hidden_bold_delimiters_yields_literal_source() {
    let rig = Self.make(markdown: "A **B** C", cursorPosition: 0)
    let textView = rig.textView
    let range = NSRange(location: 0, length: 7)
    textView.setSelectedRange(range)
    let attr = textView.textStorage!
    let copied = attr.attributedSubstring(from: range).string
    #expect(copied == "A **B**", "got: \(copied.unicodeScalars.map { String(format: "U+%04X", $0.value) }.joined(separator: " "))")
    #expect(!copied.contains("\u{200B}"), "ZWSPs must NOT leak into clipboard")
  }

  // Same shape, longer hidden run: `# Heading`. Selecting the whole line
  // must yield the literal `# Heading` (not `\u{200B}\u{200B}Heading`).
  @Test
  func copy_across_hidden_heading_prefix_yields_literal_source() {
    let md = "# Heading"
    let rig = Self.make(markdown: md, cursorPosition: (md as NSString).length)
    let textView = rig.textView
    let range = NSRange(location: 0, length: (md as NSString).length)
    textView.setSelectedRange(range)
    let attr = textView.textStorage!
    let copied = attr.attributedSubstring(from: range).string
    #expect(copied == "# Heading", "got: \(copied)")
    #expect(!copied.contains("\u{200B}"), "ZWSPs must NOT leak into clipboard")
  }

  // MARK: - Item 3: Programmatic selection across hidden ranges round-trips

  /// Set a selection that spans a hidden delimiter run via
  /// `setSelectedRange`; `selectedRange()` returns the same NSRange (TK2
  /// doesn't snap or clamp it). The drag-select gesture goes through the
  /// same path internally.
  @Test
  func selection_across_hidden_bold_runs_round_trips() {
    let md = "A **B** C D E"  // length 13
    let rig = Self.make(markdown: md, cursorPosition: 0)
    let target = NSRange(location: 0, length: 11)
    rig.textView.setSelectedRange(target)
    let read = rig.textView.selectedRange()
    #expect(
      read.location == target.location && read.length == target.length,
      "expected [\(target.location), \(target.length)); got [\(read.location), \(read.length))")
  }

  /// And a selection that STARTS at a hidden source position (where a
  /// drag could plausibly anchor due to TK2 hit-testing) round-trips too.
  @Test
  func selection_starting_at_hidden_source_position_round_trips() {
    let md = "A **B** C"
    let rig = Self.make(markdown: md, cursorPosition: 0)
    let target = NSRange(location: 2, length: 5)  // [2, 7)
    rig.textView.setSelectedRange(target)
    let read = rig.textView.selectedRange()
    #expect(
      read.location == target.location && read.length == target.length,
      "expected [\(target.location), \(target.length)); got [\(read.location), \(read.length))")
  }

  // MARK: - Item 4: Shift-arrow direction reversal honors original anchor

  /// From cursor at source 4 (B in `A **B** C`, bold revealed because
  /// cursor inside), shift+right twice grows the selection toward the
  /// right; shift+left twice MUST shrink it back to the cursor at 4 — not
  /// flip the anchor and grow leftward to [3, 6).
  @Test
  func shift_arrow_direction_reversal_preserves_anchor_inside_revealed_bold() {
    // Cursor at 4 ('B'), inside bold: `**` revealed at 2,3 and 5,6 (no
    // hidden positions in [4, 6]).
    let md = "A **B** C"
    let rig = Self.make(markdown: md, cursorPosition: 4)
    rig.textView.moveRightAndModifySelection(nil)
    let r1 = rig.textView.selectedRange()
    #expect(
      r1.location == 4 && r1.location + r1.length == 5,
      "shift+right #1: expected [4, 5); got [\(r1.location), \(r1.location + r1.length))")
    rig.textView.moveRightAndModifySelection(nil)
    let r2 = rig.textView.selectedRange()
    #expect(
      r2.location == 4 && r2.location + r2.length == 6,
      "shift+right #2: expected [4, 6); got [\(r2.location), \(r2.location + r2.length))")
    // Now reverse direction. With anchor tracking, shift+left shrinks.
    rig.textView.moveLeftAndModifySelection(nil)
    let r3 = rig.textView.selectedRange()
    #expect(
      r3.location == 4 && r3.location + r3.length == 5,
      "shift+left #1 (reversal): expected [4, 5); got [\(r3.location), \(r3.location + r3.length))")
    rig.textView.moveLeftAndModifySelection(nil)
    let r4 = rig.textView.selectedRange()
    #expect(
      r4.location == 4 && r4.length == 0,
      "shift+left #2 (reversal): expected cursor at 4; got [\(r4.location), \(r4.location + r4.length))")
  }

  /// A non-extending move clears the tracked anchor, so a subsequent
  /// shift+arrow uses the conventional anchor (the start of the selection
  /// for forward motion, the end for backward). This guards against a
  /// stale anchor being silently reused after the user clicks elsewhere.
  @Test
  func non_extending_move_clears_tracked_anchor() {
    let md = "abcdefg"
    let rig = Self.make(markdown: md, cursorPosition: 3)
    rig.textView.moveRightAndModifySelection(nil)  // [3, 4) anchor=3
    rig.textView.moveLeft(nil)  // collapse — moveLeft from sel.location=3 → 2
    let postLeft = rig.textView.selectedRange()
    #expect(
      postLeft.location == 2 && postLeft.length == 0,
      "moveLeft should land at 2 with no selection; got [\(postLeft.location), \(postLeft.length))")
    rig.textView.moveRightAndModifySelection(nil)
    let r = rig.textView.selectedRange()
    // Anchor is freshly inferred at 2 (the new cursor), head extends to 3.
    // If the stale anchor (3) had leaked through, the result would be
    // [2, 3) but with anchor=3 — and a subsequent shift+left would
    // shrink toward 3 rather than extending leftward. Pin both:
    #expect(r.location == 2 && r.length == 1, "got [\(r.location), \(r.location + r.length))")
    rig.textView.moveLeftAndModifySelection(nil)
    let r2 = rig.textView.selectedRange()
    // With a fresh anchor at 2, shift+left from head=3 should shrink to
    // [2, 2) (cursor at 2). If the stale anchor (3) leaked through, the
    // anchor would be 3 and the head 2, then shift+left would extend to
    // [1, 3) — wrong.
    #expect(r2.location == 2 && r2.length == 0, "got [\(r2.location), \(r2.location + r2.length))")
  }

  // MARK: - Item 5: Word double-click selects visible word, not delimiters

  /// Double-click on visible 'B' in `A **B** C` should select just the
  /// "word" B at source [4, 5), NOT the surrounding `**B**` construct.
  /// The textStorage holds source; AppKit's word-finding walks source —
  /// so `**` are non-word characters and the click lands on B.
  @Test
  func word_selection_at_B_in_bold_selects_just_B() {
    let md = "A **B** C"
    let rig = Self.make(markdown: md, cursorPosition: 0)
    // `selectionRange(forProposedRange:granularity:)` is the AppKit
    // primitive used by double-click word selection. We exercise it
    // directly to keep this test deterministic — no event-routing
    // dependencies.
    let proposed = NSRange(location: 4, length: 0)
    let word = rig.textView.selectionRange(
      forProposedRange: proposed, granularity: .selectByWord)
    #expect(
      word.location == 4 && word.length == 1,
      "expected word [4, 5) ('B'); got [\(word.location), \(word.location + word.length))")
  }

  /// Double-click on "Heading" (anywhere in it) in `# Heading` should
  /// select [2, 9) — just the heading text, not the hidden `# ` prefix.
  @Test
  func word_selection_in_heading_excludes_hidden_prefix() {
    let md = "# Heading"
    let rig = Self.make(markdown: md, cursorPosition: 0)
    let proposed = NSRange(location: 5, length: 0)  // mid-Heading
    let word = rig.textView.selectionRange(
      forProposedRange: proposed, granularity: .selectByWord)
    #expect(
      word.location == 2 && word.length == 7,
      "expected word [2, 9) ('Heading'); got [\(word.location), \(word.location + word.length))")
  }

  // MARK: - Item 6: Cmd-F find sees source markdown, not display

  /// `usesFindBar` is enabled on the text view. The find machinery walks
  /// `NSTextView.string`, which is the source markdown. Searching for `**`
  /// in `A **B** C` MUST therefore find the two delimiter pairs at offsets
  /// 2 and 5 — NOT zero matches because the display has substituted ZWSPs.
  @Test
  func find_searches_source_markdown_not_display() {
    let md = "A **B** C"
    let rig = Self.make(markdown: md, cursorPosition: 0)
    let str = rig.textView.string
    #expect(str == md, "textView.string must equal source: got \(str)")

    var matches: [Int] = []
    let ns = str as NSString
    var search = NSRange(location: 0, length: ns.length)
    while search.length > 0 {
      let r = ns.range(of: "**", options: [], range: search)
      if r.location == NSNotFound { break }
      matches.append(r.location)
      search.location = r.location + r.length
      search.length = ns.length - search.location
    }
    #expect(matches == [2, 5], "expected to find `**` at 2 and 5; got \(matches)")
  }

  // MARK: - Item 8: Cursor at hidden ZWSP source position navigates sensibly

  /// Place the cursor at source 2 (a hidden `*` in `A **B** C` with cursor
  /// outside bold). Right-arrow advances to the next visible source
  /// position (the revealed `*` at 3 once the cursor enters the construct,
  /// or onto 'B' at 4 if it stays outside). Left-arrow walks back to the
  /// previous visible position (1 / ' ').
  ///
  /// The behavior we pin: arrow keys never *stay* on the hidden source
  /// position, and never skip clear over the entire bold run as a
  /// no-direction step.
  @Test
  func arrow_from_hidden_source_position_advances_to_visible_position() {
    let md = "A **B** C"
    let rig = Self.make(markdown: md, cursorPosition: 2)
    rig.textView.moveRight(nil)
    let post = rig.textView.selectedRange().location
    #expect(post > 2, "right-arrow from hidden source position must advance; got \(post)")
    #expect(
      post == 3 || post == 4,
      "right-arrow from 2 should land on the next visible source offset (3 if `**` revealed, 4 if not); got \(post)")
  }
}
