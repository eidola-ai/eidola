import AppKit
import Foundation
import Testing

@testable import MarkdownEditor

/// Visual integration tests for checkbox list rendering.
@Suite("Checkbox List Visual Tests")
@MainActor
struct CheckboxListVisualTests {

  // MARK: - Typing flow

  @Test("Type '- [ ] Task item' character by character")
  func typeCheckboxItem() {
    let results = EditorTestHarness.runTyping(
      name: "checkbox-typing",
      characters: "- [ ] Task item")

    // Initial + 15 chars = 16 steps
    #expect(results.count == 16)

    let finalState = results.last!.state
    #expect(finalState.markdown == "- [ ] Task item")
    #expect(finalState.selection == .cursor(15))

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }
  }

  @Test("Type '- [x] Done' character by character")
  func typeCheckedItem() {
    let results = EditorTestHarness.runTyping(
      name: "checkbox-checked-typing",
      characters: "- [x] Done")

    let finalState = results.last!.state
    #expect(finalState.markdown == "- [x] Done")
    #expect(finalState.selection == .cursor(10))
  }

  // MARK: - Cursor movement: inside vs outside

  @Test("Cursor inside checkbox reveals delimiter, cursor outside hides it")
  func checkboxDelimiterVisibility() {
    let markdown = "- [ ] Task\n\nBody text"
    let initial = EditorState(markdown: markdown, selection: .cursor(8))  // inside task

    let events: [EditorEvent] = [
      .setSelection(.cursor(14)),  // move to body
      .setSelection(.cursor(8)),   // move back to task
    ]

    let results = EditorTestHarness.run(
      name: "checkbox-delimiter-visibility",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 200))

    #expect(results.count == 3)

    // Step 0 (cursor in checkbox) vs step 1 (cursor in body) should differ
    #expect(
      results[0].bitmapHash != results[1].bitmapHash,
      "Cursor inside vs outside checkbox should look different")

    // Step 2 (cursor back in checkbox) should match step 0
    #expect(
      results[0].bitmapHash == results[2].bitmapHash,
      "Same cursor position should produce same visual")
  }

  @Test("Cursor at various positions around checkbox item")
  func cursorAtVariousPositions() {
    let markdown = "- [ ] Task item\n\nBody text here"
    let positions: [(String, Int)] = [
      ("start-of-marker", 0),
      ("after-dash", 1),
      ("after-dash-space", 2),
      ("inside-bracket", 3),
      ("after-checkbox", 6),
      ("middle-of-content", 10),
      ("end-of-content", 15),
      ("blank-line", 16),
      ("body-start", 17),
      ("body-end", 31),
    ]

    var events: [EditorEvent] = []
    for (_, offset) in positions.dropFirst() {
      events.append(.setSelection(.cursor(offset)))
    }

    let initial = EditorState(
      markdown: markdown, selection: .cursor(positions[0].1))

    let results = EditorTestHarness.run(
      name: "checkbox-cursor-positions",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 200))

    #expect(results.count == positions.count)

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }

    // At least some positions should look different
    var uniqueHashes = Set<Int>()
    for r in results {
      uniqueHashes.insert(r.bitmapHash)
    }
    #expect(
      uniqueHashes.count >= 2,
      "Expected at least 2 visually distinct states, got \(uniqueHashes.count)")
  }

  // MARK: - Mixed list: bullets and checkboxes

  @Test("Mixed list with bullets and checkboxes")
  func mixedListVisual() {
    let markdown = "- Regular item\n- [ ] Unchecked task\n- [x] Completed task\n- Another regular\n\nBody"

    let positions: [(String, Int)] = [
      ("in-regular", 5),
      ("in-unchecked", 22),
      ("in-checked", 43),
      ("in-second-regular", 63),
      ("in-body", 78),
    ]

    var events: [EditorEvent] = []
    for (_, offset) in positions.dropFirst() {
      events.append(.setSelection(.cursor(offset)))
    }

    let initial = EditorState(
      markdown: markdown, selection: .cursor(positions[0].1))

    let results = EditorTestHarness.run(
      name: "checkbox-mixed-list",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 300))

    #expect(results.count == positions.count)

    var uniqueHashes = Set<Int>()
    for r in results {
      uniqueHashes.insert(r.bitmapHash)
    }
    #expect(
      uniqueHashes.count >= 3,
      "Expected at least 3 distinct visuals for different cursor positions, got \(uniqueHashes.count)")
  }

  // MARK: - Enter / Backspace visual

  @Test("Enter continues checkbox, Enter on empty ends it")
  func enterContinuesAndEndsCheckbox() {
    let events: [EditorEvent] = [
      .insertText("- [ ] Task 1"),
      .insertNewline,         // continues list: "- [ ] Task 1\n- [ ] "
      .insertNewline,         // empty item: removes marker
    ]

    let results = EditorTestHarness.run(
      name: "checkbox-enter-empty",
      initial: EditorState(),
      events: events,
      size: NSSize(width: 600, height: 200))

    // After first Enter, should have continued list with unchecked checkbox
    #expect(results[2].state.markdown == "- [ ] Task 1\n- [ ] ")
    // After second Enter on empty item, marker removed
    #expect(results[3].state.markdown == "- [ ] Task 1\n\n")

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath))
    }
  }

  @Test("Backspace removes checkbox marker visual")
  func backspaceRemovesCheckboxMarkerVisual() {
    let initial = EditorState(markdown: "- [ ] Hello", selection: .cursor(6))
    let events: [EditorEvent] = [
      .deleteBackward,  // removes "- [ ] ", leaves "Hello"
    ]

    let results = EditorTestHarness.run(
      name: "checkbox-backspace-marker",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 200))

    #expect(results[1].state.markdown == "Hello")
    #expect(results[1].state.selection == .cursor(0))

    // Visual should change (list styling removed)
    #expect(
      results[0].bitmapHash != results[1].bitmapHash,
      "Removing marker should produce visual change")
  }

  // MARK: - Determinism

  @Test("Fresh render matches incremental render for checkbox list")
  func determinismCheckbox() {
    let results = EditorTestHarness.runTyping(
      name: "determinism-checkbox",
      characters: "- [ ] Task item")

    let finalState = results.last!.state
    let freshBitmap = SnapshotCapture.capture(
      text: finalState.markdown,
      cursorPosition: finalState.selection.head,
      size: NSSize(width: 600, height: 200))

    let incrementalBitmap = NSBitmapImageRep(
      data: try! Data(contentsOf: URL(fileURLWithPath: results.last!.imagePath)))!

    let comparison = BitmapComparator.compare(freshBitmap, incrementalBitmap)
    #expect(comparison.isMatch, "Fresh and incremental renders must match for checkbox list")
  }
}
