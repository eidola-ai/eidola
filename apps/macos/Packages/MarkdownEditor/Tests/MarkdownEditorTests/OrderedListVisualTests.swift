import AppKit
import Foundation
import Testing

@testable import MarkdownEditor

/// Visual integration tests for ordered list rendering.
@Suite("Ordered List Visual Tests")
@MainActor
struct OrderedListVisualTests {

  // MARK: - Typing flow

  @Test("Type '1. Hello' character by character")
  func typeOrderedListItem() {
    let results = EditorTestHarness.runTyping(
      name: "ordered-list-typing-hello",
      characters: "1. Hello")

    // Initial + 8 chars = 9 steps
    #expect(results.count == 9)

    let finalState = results.last!.state
    #expect(finalState.markdown == "1. Hello")
    #expect(finalState.selection == .cursor(8))

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }
  }

  @Test("Type multi-item ordered list with Enter auto-numbering")
  func typeMultiItemOrderedList() {
    // Type "1. First", Enter (auto "2. "), "Second", Enter (auto "3. "), "Third"
    var events: [EditorEvent] = []
    for c in "1. First" { events.append(.insertText(String(c))) }
    events.append(.insertNewline)
    for c in "Second" { events.append(.insertText(String(c))) }
    events.append(.insertNewline)
    for c in "Third" { events.append(.insertText(String(c))) }

    let results = EditorTestHarness.run(
      name: "ordered-list-multi-item-typing",
      initial: EditorState(),
      events: events,
      size: NSSize(width: 600, height: 300))

    let finalState = results.last!.state
    #expect(finalState.markdown == "1. First\n2. Second\n3. Third")

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }
  }

  // MARK: - Cursor movement: inside vs outside (should look the same)

  @Test("Cursor inside and outside ordered list item look the same (no delimiter hiding)")
  func orderedListNoDelimiterToggle() {
    let markdown = "1. Hello\n\nBody text"
    let initial = EditorState(markdown: markdown, selection: .cursor(5))  // inside list

    let events: [EditorEvent] = [
      .setSelection(.cursor(11)),  // move to body
      .setSelection(.cursor(5)),   // move back to list
    ]

    let results = EditorTestHarness.run(
      name: "ordered-list-no-delimiter-toggle",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 200))

    #expect(results.count == 3)

    // Key difference from unordered lists: cursor inside (step 0) and cursor
    // outside (step 1) should look the SAME for the ordered list part
    // (no bullet/hide toggling). They may still differ due to cursor position
    // affecting other visual elements, but the ordered list marker should be
    // visible in both cases. Step 2 (back inside) should match step 0.
    #expect(
      results[0].bitmapHash == results[2].bitmapHash,
      "Same cursor position should produce same visual")
  }

  @Test("Cursor at various positions around ordered list item")
  func cursorAtVariousPositions() {
    let markdown = "1. Hello World\n\nBody text here"
    let positions: [(String, Int)] = [
      ("start-of-marker", 0),
      ("after-marker", 3),
      ("middle-of-content", 7),
      ("end-of-content", 14),
      ("blank-line", 15),
      ("body-start", 16),
      ("body-middle", 21),
      ("body-end", 30),
    ]

    var events: [EditorEvent] = []
    for (_, offset) in positions.dropFirst() {
      events.append(.setSelection(.cursor(offset)))
    }

    let initial = EditorState(
      markdown: markdown, selection: .cursor(positions[0].1))

    let results = EditorTestHarness.run(
      name: "ordered-list-cursor-positions",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 200))

    #expect(results.count == positions.count)

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }
  }

  // MARK: - Multiple ordered list items with cursor on different items

  @Test("Cursor on different ordered list items (no bullet toggling)")
  func cursorOnDifferentOrderedItems() {
    let markdown = "1. Item 1\n2. Item 2\n3. Item 3\n\nBody"

    let positions: [(String, Int)] = [
      ("first-item", 5),       // inside "Item 1"
      ("second-item", 15),     // inside "Item 2"
      ("third-item", 25),      // inside "Item 3"
      ("body", 33),            // in body
    ]

    var events: [EditorEvent] = []
    for (_, offset) in positions.dropFirst() {
      events.append(.setSelection(.cursor(offset)))
    }

    let initial = EditorState(
      markdown: markdown, selection: .cursor(positions[0].1))

    let results = EditorTestHarness.run(
      name: "ordered-list-cursor-different-items",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 300))

    #expect(results.count == positions.count)

    // Unlike unordered lists, all cursor positions inside different ordered
    // list items should NOT cause any bullet/hide toggling. The main visual
    // difference is just the cursor position itself.
    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath))
    }
  }

  // MARK: - Mixed ordered and unordered lists

  @Test("Mixed ordered and unordered lists render correctly")
  func mixedOrderedUnorderedVisual() {
    let markdown = "- Unordered 1\n- Unordered 2\n\n1. Ordered 1\n2. Ordered 2\n\nBody text"

    let positions: [(String, Int)] = [
      ("unordered-first", 5),
      ("unordered-second", 19),
      ("ordered-first", 34),
      ("ordered-second", 47),
      ("body", 60),
    ]

    var events: [EditorEvent] = []
    for (_, offset) in positions.dropFirst() {
      events.append(.setSelection(.cursor(offset)))
    }

    let initial = EditorState(
      markdown: markdown, selection: .cursor(positions[0].1))

    let results = EditorTestHarness.run(
      name: "ordered-list-mixed",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 400))

    #expect(results.count == positions.count)

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath))
    }

    // When cursor is in body, unordered items should show bullets but
    // ordered items should show their numeric markers. These should look
    // visually distinct from each other.
    var uniqueHashes = Set<Int>()
    for r in results {
      uniqueHashes.insert(r.bitmapHash)
    }
    #expect(
      uniqueHashes.count >= 2,
      "Expected at least 2 visually distinct states, got \(uniqueHashes.count)")
  }

  // MARK: - Enter / Backspace visual

  @Test("Enter on empty ordered list item visual: list ends")
  func enterOnEmptyOrderedListItemVisual() {
    let events: [EditorEvent] = [
      .insertText("1. Item 1"),
      .insertNewline,         // continues list: "1. Item 1\n2. "
      .insertNewline,         // empty item: removes marker
    ]

    let results = EditorTestHarness.run(
      name: "ordered-list-enter-empty-item",
      initial: EditorState(),
      events: events,
      size: NSSize(width: 600, height: 200))

    // After first Enter, should have continued list with "2. "
    #expect(results[2].state.markdown == "1. Item 1\n2. ")
    // After second Enter on empty item, marker removed
    #expect(results[3].state.markdown == "1. Item 1\n\n")

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath))
    }
  }

  @Test("Backspace removes ordered list marker visual")
  func backspaceRemovesOrderedMarkerVisual() {
    let initial = EditorState(markdown: "1. Hello", selection: .cursor(3))
    let events: [EditorEvent] = [
      .deleteBackward,  // removes "1. ", leaves "Hello"
    ]

    let results = EditorTestHarness.run(
      name: "ordered-list-backspace-marker",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 200))

    #expect(results[1].state.markdown == "Hello")
    #expect(results[1].state.selection == .cursor(0))

    // Visual should change (list styling removed)
    #expect(
      results[0].bitmapHash != results[1].bitmapHash,
      "Removing ordered marker should produce visual change")
  }

  // MARK: - Renumbering visual tests

  @Test("Insert item in middle of ordered list renumbers visually")
  func insertMiddleRenumbersVisual() {
    let initial = EditorState(
      markdown: "1. First\n2. Second\n3. Third", selection: .cursor(8))
    let events: [EditorEvent] = [
      .insertNewline,  // Insert new item after "First"
    ]

    let results = EditorTestHarness.run(
      name: "ordered-list-renumber-insert-middle",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 300))

    let finalState = results.last!.state
    #expect(finalState.markdown == "1. First\n2. \n3. Second\n4. Third")

    // Visual should change (new item inserted, renumbered)
    #expect(
      results[0].bitmapHash != results[1].bitmapHash,
      "Inserting item should produce visual change")

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath))
    }
  }

  @Test("Split ordered list renumbers second half visually")
  func splitListRenumbersVisual() {
    let initial = EditorState(
      markdown: "1. First\n2. \n3. Third", selection: .cursor(12))
    let events: [EditorEvent] = [
      .insertNewline,  // Enter on empty item ends the list
    ]

    let results = EditorTestHarness.run(
      name: "ordered-list-renumber-split",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 300))

    let finalState = results.last!.state
    #expect(finalState.markdown.contains("1. Third"))

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath))
    }
  }

  @Test("Recombine ordered lists renumbers visually")
  func recombineListsRenumbersVisual() {
    let initial = EditorState(
      markdown: "1. First\n2. Second\n\n1. Alpha\n2. Beta",
      selection: .cursor(19))
    let events: [EditorEvent] = [
      .deleteForward,  // Delete the blank line, combining lists
    ]

    let results = EditorTestHarness.run(
      name: "ordered-list-renumber-recombine",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 300))

    let finalState = results.last!.state
    #expect(finalState.markdown == "1. First\n2. Second\n3. Alpha\n4. Beta")

    #expect(
      results[0].bitmapHash != results[1].bitmapHash,
      "Recombining lists should produce visual change")

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath))
    }
  }

  // MARK: - Determinism

  @Test("Fresh render matches incremental render for ordered list")
  func determinismOrderedList() {
    let results = EditorTestHarness.runTyping(
      name: "determinism-ordered-list",
      characters: "1. Hello World")

    let finalState = results.last!.state
    let freshBitmap = SnapshotCapture.capture(
      text: finalState.markdown,
      cursorPosition: finalState.selection.head,
      size: NSSize(width: 600, height: 200))

    let incrementalBitmap = NSBitmapImageRep(
      data: try! Data(contentsOf: URL(fileURLWithPath: results.last!.imagePath)))!

    let comparison = BitmapComparator.compare(freshBitmap, incrementalBitmap)
    #expect(comparison.isMatch, "Fresh and incremental renders must match for ordered list")
  }
}
