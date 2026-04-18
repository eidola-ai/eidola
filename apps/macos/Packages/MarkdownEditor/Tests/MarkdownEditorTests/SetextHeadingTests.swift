import AppKit
import Foundation
import Testing

@testable import MarkdownEditor

/// Tests for setext heading behavior, covering the subtle interaction between
/// single-dash suppression, multi-dash/equals heading styling, and normalization
/// on cursor movement.
///
/// Setext headings are formed by text followed by an underline of `=` (h1) or
/// `-` (h2). A single `-` is ambiguous (could be a list item start), so the
/// editor suppresses heading styling when the cursor is on that line. Two or
/// more dashes, and any count of `=`, are unambiguous and always styled.
///
/// When the cursor moves away from the underline, `EditorUpdate` normalizes
/// setext headings to ATX format (`# ` / `## `), removing the underline.
@Suite("Setext Heading Tests")
@MainActor
struct SetextHeadingTests {

  // MARK: - 1. Single `-` suppression when cursor is on the underline

  /// **Pass criteria:** With cursor on the `-` line, both lines render as plain
  /// body text. "Maybe a heading" is in normal (non-heading) font. The `-` is
  /// visible as literal text. No heading font size, no delimiter dimming.
  ///
  /// **Fail criteria:** "Maybe a heading" renders in large heading font, or the
  /// `-` is hidden/dimmed. That would mean the single-dash suppression is broken.
  @Test("Single dash: cursor on underline suppresses heading styling")
  func singleDashCursorOnUnderline() {
    // "Maybe a heading\n-"
    // Cursor at position 17 (on the `-`)
    let markdown = "Maybe a heading\n-"
    let initial = EditorState(markdown: markdown, selection: .cursor(17))

    let results = EditorTestHarness.run(
      name: "setext-single-dash-on-underline",
      initial: initial,
      events: [],
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 1)

    // Verify via RenderSpec: no heading styling should be applied
    let spec = MarkdownRenderer.render(
      text: markdown,
      cursorRange: NSRange(location: 17, length: 0))

    // With suppression active, hidden indexes should be empty (no delimiters hidden)
    // and there should be no heading-font styled ranges
    let style = MarkdownStyle.default
    let headingFont = style.headingFont(level: 2)
    let hasHeadingFont = spec.styledRanges.contains { styled in
      if let font = styled.attributes[.font] as? NSFont {
        return font.pointSize == headingFont.pointSize
      }
      return false
    }
    #expect(!hasHeadingFont, "Single-dash setext should NOT have heading font when cursor is on underline")
  }

  /// Same test but with cursor right after the `-` (at position 17 = end of `-`).
  @Test("Single dash: cursor after dash character suppresses heading styling")
  func singleDashCursorAfterDash() {
    let markdown = "Maybe a heading\n-"
    // Cursor at end of document (position 17 = after the `-`)
    let initial = EditorState(markdown: markdown, selection: .cursor(17))

    let results = EditorTestHarness.run(
      name: "setext-single-dash-after-dash",
      initial: initial,
      events: [],
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 1)

    let spec = MarkdownRenderer.render(
      text: markdown,
      cursorRange: NSRange(location: 17, length: 0))

    let style = MarkdownStyle.default
    let headingFont = style.headingFont(level: 2)
    let hasHeadingFont = spec.styledRanges.contains { styled in
      if let font = styled.attributes[.font] as? NSFont {
        return font.pointSize == headingFont.pointSize
      }
      return false
    }
    #expect(!hasHeadingFont, "Single-dash setext should NOT style as heading when cursor is after the dash")
  }

  /// **Pass criteria:** With cursor on body text below, the setext heading is
  /// normalized to ATX `## Maybe a heading` (the `-` line disappears). The
  /// heading renders in heading font with delimiter hidden.
  ///
  /// **Fail criteria:** The `-` underline remains visible, or the text stays
  /// in body font even though cursor is away from the underline.
  @Test("Single dash: cursor elsewhere triggers normalization to ATX")
  func singleDashCursorElsewhere() {
    // "Maybe a heading\n-\n\nSome body text"
    // Cursor on "Some body text" (position 19+)
    let markdown = "Maybe a heading\n-\n\nSome body text"
    let initial = EditorState(markdown: markdown, selection: .cursor(19))

    // EditorUpdate.update with setSelection normalizes setext headings
    let normalized = EditorUpdate.update(initial, event: .setSelection(.cursor(25)))

    // After normalization, the markdown should be ATX format
    #expect(
      normalized.markdown.hasPrefix("## Maybe a heading"),
      "Setext heading should be normalized to ATX when cursor is away from underline. Got: \(normalized.markdown)")

    // The `-` line should be gone
    #expect(
      !normalized.markdown.contains("\n-\n"),
      "The dash underline should be removed after normalization")

    let results = EditorTestHarness.run(
      name: "setext-single-dash-normalized",
      initial: normalized,
      events: [],
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 1)
  }

  // MARK: - 2. Single `- ` (dash + trailing space) same suppression

  /// **Pass criteria:** A single `-` followed by a space is still just one dash
  /// when trimmed. Cursor on that line should suppress heading styling, same as
  /// a bare `-`.
  ///
  /// **Fail criteria:** The trailing space causes the suppression to fail and
  /// heading styling appears while cursor is on the line.
  @Test("Single dash with trailing space: cursor on underline suppresses heading styling")
  func singleDashWithSpaceSuppressed() {
    let markdown = "Maybe a heading\n- "
    // Cursor at position 17 (on the `-`)
    let initial = EditorState(markdown: markdown, selection: .cursor(17))

    let results = EditorTestHarness.run(
      name: "setext-single-dash-space-suppressed",
      initial: initial,
      events: [],
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 1)

    let spec = MarkdownRenderer.render(
      text: markdown,
      cursorRange: NSRange(location: 17, length: 0))

    let style = MarkdownStyle.default
    let headingFont = style.headingFont(level: 2)
    let hasHeadingFont = spec.styledRanges.contains { styled in
      if let font = styled.attributes[.font] as? NSFont {
        return font.pointSize == headingFont.pointSize
      }
      return false
    }
    #expect(!hasHeadingFont, "Single dash+space setext should NOT have heading font when cursor is on underline")
  }

  // MARK: - 3. `--` or longer DOES get heading styling even with cursor on it

  /// **Pass criteria:** With cursor on the `--` line, "Definitely a heading"
  /// renders in heading font. The `--` is visible and dimmed (as a delimiter).
  ///
  /// **Fail criteria:** The heading styling is suppressed (text in body font),
  /// or the `--` is hidden when cursor is on it.
  @Test("Double dash: cursor on underline DOES apply heading styling")
  func doubleDashOnUnderline() {
    let markdown = "Definitely a heading\n--"
    // Cursor on `--` line (position 21)
    let initial = EditorState(markdown: markdown, selection: .cursor(21))

    let results = EditorTestHarness.run(
      name: "setext-double-dash-on-underline",
      initial: initial,
      events: [],
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 1)

    let spec = MarkdownRenderer.render(
      text: markdown,
      cursorRange: NSRange(location: 21, length: 0))

    let style = MarkdownStyle.default
    let headingFont = style.headingFont(level: 2)
    let hasHeadingFont = spec.styledRanges.contains { styled in
      if let font = styled.attributes[.font] as? NSFont {
        return font.pointSize == headingFont.pointSize
      }
      return false
    }
    #expect(hasHeadingFont, "Double-dash setext SHOULD have heading font even when cursor is on underline")

    // The `--` delimiter should NOT be hidden (cursor is on the heading)
    #expect(spec.hiddenIndexes.isEmpty, "Delimiter should be visible (not hidden) when cursor is on the heading")

    // The delimiter should be dimmed via temporary attributes
    #expect(!spec.temporaryAttributes.isEmpty, "Delimiter should be dimmed when cursor is on the heading")
  }

  /// Same test with longer underlines (3+ dashes).
  @Test("Triple dash: cursor on underline DOES apply heading styling")
  func tripleDashOnUnderline() {
    let markdown = "Heading text\n---"
    // Cursor on `---` (position 13)
    let initial = EditorState(markdown: markdown, selection: .cursor(13))

    let results = EditorTestHarness.run(
      name: "setext-triple-dash-on-underline",
      initial: initial,
      events: [],
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 1)

    let spec = MarkdownRenderer.render(
      text: markdown,
      cursorRange: NSRange(location: 13, length: 0))

    let style = MarkdownStyle.default
    let headingFont = style.headingFont(level: 2)
    let hasHeadingFont = spec.styledRanges.contains { styled in
      if let font = styled.attributes[.font] as? NSFont {
        return font.pointSize == headingFont.pointSize
      }
      return false
    }
    #expect(hasHeadingFont, "Triple-dash setext SHOULD have heading font when cursor is on underline")
  }

  // MARK: - 4. `=` (any count) always gets heading styling

  /// **Pass criteria:** With cursor on the `=` line, "Heading One" renders in
  /// h1 heading font. The `=` is visible and dimmed. No suppression occurs
  /// because `=` is unambiguous.
  ///
  /// **Fail criteria:** Heading styling is suppressed, or the `=` is hidden
  /// when cursor is on it.
  @Test("Single equals: cursor on underline DOES apply heading styling")
  func singleEqualsOnUnderline() {
    let markdown = "Heading One\n="
    // Cursor on `=` (position 12)
    let initial = EditorState(markdown: markdown, selection: .cursor(12))

    let results = EditorTestHarness.run(
      name: "setext-single-equals-on-underline",
      initial: initial,
      events: [],
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 1)

    let spec = MarkdownRenderer.render(
      text: markdown,
      cursorRange: NSRange(location: 12, length: 0))

    let style = MarkdownStyle.default
    let headingFont = style.headingFont(level: 1)
    let hasHeadingFont = spec.styledRanges.contains { styled in
      if let font = styled.attributes[.font] as? NSFont {
        return font.pointSize == headingFont.pointSize
      }
      return false
    }
    #expect(hasHeadingFont, "Single-equals setext SHOULD have h1 heading font (= is unambiguous)")

    // Delimiter should be visible (not hidden) and dimmed
    #expect(spec.hiddenIndexes.isEmpty, "= delimiter should be visible when cursor is on it")
    #expect(!spec.temporaryAttributes.isEmpty, "= delimiter should be dimmed when cursor is on it")
  }

  /// Multiple equals signs also always get heading styling.
  @Test("Multiple equals: cursor on underline DOES apply heading styling")
  func multipleEqualsOnUnderline() {
    let markdown = "Heading One\n==="
    // Cursor on `===` (position 13)
    let initial = EditorState(markdown: markdown, selection: .cursor(13))

    let results = EditorTestHarness.run(
      name: "setext-multi-equals-on-underline",
      initial: initial,
      events: [],
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 1)

    let spec = MarkdownRenderer.render(
      text: markdown,
      cursorRange: NSRange(location: 13, length: 0))

    let style = MarkdownStyle.default
    let headingFont = style.headingFont(level: 1)
    let hasHeadingFont = spec.styledRanges.contains { styled in
      if let font = styled.attributes[.font] as? NSFont {
        return font.pointSize == headingFont.pointSize
      }
      return false
    }
    #expect(hasHeadingFont, "Multi-equals setext SHOULD have h1 heading font")
  }

  // MARK: - 5. Normalization on cursor move

  /// **Pass criteria:** Starting with a setext heading and cursor on the
  /// underline, moving the cursor to body text triggers normalization. The
  /// markdown changes from setext to ATX format (underline disappears, `## `
  /// prefix added). The heading text renders in heading font with hidden
  /// delimiter.
  ///
  /// **Fail criteria:** The setext format persists after cursor moves away,
  /// or the normalization produces malformed ATX heading.
  @Test("Cursor move away from setext underline normalizes to ATX")
  func cursorMoveNormalizesToATX() {
    // Start with cursor on `--` underline
    let markdown = "My heading\n--\n\nBody text"
    let initial = EditorState(markdown: markdown, selection: .cursor(11))

    let events: [EditorEvent] = [
      // Move cursor to body text
      .setSelection(.cursor(20)),
    ]

    let results = EditorTestHarness.run(
      name: "setext-normalization-on-move",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 2)

    // After moving cursor away, the state should be normalized
    let finalState = results[1].state
    #expect(
      finalState.markdown.hasPrefix("## My heading"),
      "Should normalize to ATX h2. Got: \(finalState.markdown)")
    #expect(
      !finalState.markdown.contains("\n--"),
      "Underline should be removed after normalization")

    // Visuals should differ between step 0 (setext, cursor on underline) and
    // step 1 (ATX, cursor on body)
    #expect(
      results[0].bitmapHash != results[1].bitmapHash,
      "Visual should change after normalization")
  }

  /// Same test for `=` underline normalizing to `# ` (h1).
  @Test("Equals underline normalizes to ATX h1 on cursor move")
  func equalsNormalizesToH1() {
    let markdown = "Title\n==\n\nBody"
    let initial = EditorState(markdown: markdown, selection: .cursor(6))

    let events: [EditorEvent] = [
      .setSelection(.cursor(12)),
    ]

    let results = EditorTestHarness.run(
      name: "setext-equals-normalization",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 2)

    let finalState = results[1].state
    #expect(
      finalState.markdown.hasPrefix("# Title"),
      "Should normalize to ATX h1. Got: \(finalState.markdown)")
  }

  // MARK: - 6. Typing flow: `-` after text

  /// **Pass criteria:** After typing just `-` on a new line following text,
  /// the parser sees a setext h2 but the cursor is on the `-` so heading
  /// styling is suppressed. Both lines render in body font. After adding a
  /// space, still suppressed. After adding `I`, it becomes `- I` which is a
  /// list item, not a heading at all.
  ///
  /// **Fail criteria:** At any point during typing `-`, ` `, `I`, the text
  /// "Some text" renders in heading font (jarring style change).
  @Test("Typing `-` after text: suppressed until it becomes a list item")
  func typingSingleDashAfterText() {
    let initial = EditorState(markdown: "Some text\n", selection: .cursor(10))

    let events: [EditorEvent] = [
      .insertText("-"),   // Now "Some text\n-" — setext h2, but suppressed
      .insertText(" "),   // Now "Some text\n- " — still suppressed
      .insertText("I"),   // Now "Some text\n- I" — list item, not heading
    ]

    let results = EditorTestHarness.run(
      name: "setext-typing-single-dash",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 4)  // initial + 3 events

    // Step 0: "Some text\n" — plain text, cursor at end
    #expect(results[0].state.markdown == "Some text\n")

    // Step 1: "Some text\n-" — setext suppressed
    #expect(results[1].state.markdown == "Some text\n-")
    let specStep1 = MarkdownRenderer.render(
      text: results[1].state.markdown,
      cursorRange: results[1].state.selection.nsRange)
    let style = MarkdownStyle.default
    let h2Font = style.headingFont(level: 2)
    let hasHeadingStep1 = specStep1.styledRanges.contains { styled in
      if let font = styled.attributes[.font] as? NSFont {
        return font.pointSize == h2Font.pointSize
      }
      return false
    }
    #expect(!hasHeadingStep1, "After typing single `-`, heading styling should be suppressed")

    // Step 2: "Some text\n- " — still suppressed (dash + space)
    #expect(results[2].state.markdown == "Some text\n- ")

    // Step 3: "Some text\n- I" — now a list item
    #expect(results[3].state.markdown == "Some text\n- I")
    let specStep3 = MarkdownRenderer.render(
      text: results[3].state.markdown,
      cursorRange: results[3].state.selection.nsRange)
    let hasHeadingStep3 = specStep3.styledRanges.contains { styled in
      if let font = styled.attributes[.font] as? NSFont {
        return font.pointSize == h2Font.pointSize
      }
      return false
    }
    #expect(!hasHeadingStep3, "After typing `- I`, should be list item not heading")
  }

  // MARK: - 7. Typing flow: `--` after text

  /// **Pass criteria:** After typing the first `-`, heading styling is
  /// suppressed (single dash). After typing the second `-` to get `--`,
  /// heading styling kicks in immediately. "Some text" renders in heading
  /// font and `--` is dimmed.
  ///
  /// **Fail criteria:** Heading styling remains suppressed after `--`, or
  /// it appears prematurely after just one `-`.
  @Test("Typing `--` after text: heading styling kicks in at second dash")
  func typingDoubleDashAfterText() {
    let initial = EditorState(markdown: "Some text\n", selection: .cursor(10))

    let events: [EditorEvent] = [
      .insertText("-"),   // "Some text\n-" — suppressed
      .insertText("-"),   // "Some text\n--" — heading styling active
    ]

    let results = EditorTestHarness.run(
      name: "setext-typing-double-dash",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 3)  // initial + 2 events

    // Step 1: single dash — suppressed
    #expect(results[1].state.markdown == "Some text\n-")
    let specSingle = MarkdownRenderer.render(
      text: results[1].state.markdown,
      cursorRange: results[1].state.selection.nsRange)
    let style = MarkdownStyle.default
    let h2Font = style.headingFont(level: 2)
    let hasSingleHeading = specSingle.styledRanges.contains { styled in
      if let font = styled.attributes[.font] as? NSFont {
        return font.pointSize == h2Font.pointSize
      }
      return false
    }
    #expect(!hasSingleHeading, "Single dash should suppress heading styling")

    // Step 2: double dash — heading active
    #expect(results[2].state.markdown == "Some text\n--")
    let specDouble = MarkdownRenderer.render(
      text: results[2].state.markdown,
      cursorRange: results[2].state.selection.nsRange)
    let hasDoubleHeading = specDouble.styledRanges.contains { styled in
      if let font = styled.attributes[.font] as? NSFont {
        return font.pointSize == h2Font.pointSize
      }
      return false
    }
    #expect(hasDoubleHeading, "Double dash should activate heading styling")

    // Visual change should occur between step 1 and step 2
    #expect(
      results[1].bitmapHash != results[2].bitmapHash,
      "Visual should change when second dash activates heading styling")
  }

  // MARK: - Visual comparison: cursor inside vs outside

  /// Compare the visual output of a setext heading (double dash) with cursor
  /// on the underline vs cursor on body text. When cursor is on the underline,
  /// the `--` should be visible and dimmed. When cursor is on body text, the
  /// heading should be normalized to ATX format.
  ///
  /// **Pass criteria:** Step 0 shows two-line setext format with visible dimmed
  /// `--`. Step 1 shows ATX format with heading text only (underline gone).
  ///
  /// **Fail criteria:** Both steps look identical, or the underline persists
  /// when cursor is on body text.
  @Test("Setext heading visual: cursor on underline vs cursor on body")
  func setextCursorInsideVsOutside() {
    let markdown = "Heading\n--\n\nBody text"
    let initial = EditorState(markdown: markdown, selection: .cursor(8))

    let events: [EditorEvent] = [
      // Move to body text — triggers normalization
      .setSelection(.cursor(15)),
    ]

    let results = EditorTestHarness.run(
      name: "setext-cursor-inside-vs-outside",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 300))

    #expect(results.count == 2)

    // The markdown should change after cursor moves
    #expect(
      results[0].state.markdown != results[1].state.markdown,
      "Markdown should change after normalization")

    // Visuals should differ
    #expect(
      results[0].bitmapHash != results[1].bitmapHash,
      "Visual should change after normalization")
  }

  // MARK: - Edge cases

  /// Verify that a setext heading with cursor at the very start of the
  /// underline line triggers suppression for single dash.
  @Test("Single dash: cursor at start of underline line suppresses")
  func singleDashCursorAtLineStart() {
    let markdown = "Content\n-"
    // Position 8 = start of second line (the `-`)
    let spec = MarkdownRenderer.render(
      text: markdown,
      cursorRange: NSRange(location: 8, length: 0))

    let style = MarkdownStyle.default
    let h2Font = style.headingFont(level: 2)
    let hasHeadingFont = spec.styledRanges.contains { styled in
      if let font = styled.attributes[.font] as? NSFont {
        return font.pointSize == h2Font.pointSize
      }
      return false
    }
    #expect(!hasHeadingFont, "Single dash: cursor at start of underline line should suppress heading")
  }

  /// Verify that cursor on the content line (not underline) of a single-dash
  /// setext still suppresses, because the parser sees the whole construct and
  /// cursor overlap extends to both lines.
  @Test("Single dash: cursor on content line renders as heading when normalizer runs")
  func singleDashCursorOnContentLine() {
    // "Content\n-" with cursor on "Content" (position 3)
    // When EditorUpdate processes a setSelection here, it will normalize
    // because the cursor is NOT on the underline line.
    let markdown = "Content\n-"
    let state = EditorState(markdown: markdown, selection: .cursor(3))
    let normalized = EditorUpdate.update(state, event: .setSelection(.cursor(3)))

    // Should normalize to ATX
    #expect(
      normalized.markdown.hasPrefix("## Content"),
      "Cursor on content line should trigger normalization. Got: \(normalized.markdown)")
  }

  /// Verify that a setext heading with equals underline followed by body text
  /// normalizes correctly and preserves body text.
  @Test("Equals underline normalization preserves body text")
  func equalsNormalizationPreservesBody() {
    let markdown = "Title\n=\n\nParagraph one\n\nParagraph two"
    let state = EditorState(markdown: markdown, selection: .cursor(20))
    let normalized = EditorUpdate.update(state, event: .setSelection(.cursor(20)))

    #expect(
      normalized.markdown.contains("# Title"),
      "Should normalize to ATX h1")
    #expect(
      normalized.markdown.contains("Paragraph one"),
      "Should preserve first body paragraph")
    #expect(
      normalized.markdown.contains("Paragraph two"),
      "Should preserve second body paragraph")
  }

  // MARK: - Determinism

  /// Verify that incremental rendering of a setext heading (double dash, cursor
  /// on underline) matches a fresh render of the same state.
  @Test("Determinism: setext heading incremental vs fresh render match")
  func determinismSetextHeading() {
    let markdown = "Heading\n--"
    let initial = EditorState(markdown: markdown, selection: .cursor(9))

    let results = EditorTestHarness.run(
      name: "setext-determinism",
      initial: initial,
      events: [],
      size: NSSize(width: 600, height: 300))

    let finalState = results.last!.state
    let freshBitmap = SnapshotCapture.capture(
      text: finalState.markdown,
      cursorPosition: finalState.selection.head,
      size: NSSize(width: 600, height: 300))

    let incrementalBitmap = NSBitmapImageRep(
      data: try! Data(contentsOf: URL(fileURLWithPath: results.last!.imagePath)))!

    let comparison = BitmapComparator.compare(freshBitmap, incrementalBitmap)
    #expect(comparison.isMatch, "Fresh and incremental renders must match for setext heading")
  }
}
