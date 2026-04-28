import AppKit
import Foundation
import Testing

@testable import MarkdownEditor

/// Visual integration tests for unordered list rendering.
@Suite("Unordered List Visual Tests")
@MainActor
struct UnorderedListVisualTests {

  // MARK: - Typing flow

  @Test("Type '- Hello' character by character")
  func typeListItem() {
    let results = EditorTestHarness.runTyping(
      name: "list-typing-hello",
      characters: "- Hello")

    // Initial + 7 chars = 8 steps
    #expect(results.count == 8)

    let finalState = results.last!.state
    #expect(finalState.markdown == "- Hello")
    #expect(finalState.selection == .cursor(7))

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }

    // Note: "- " alone (without content) may not be recognized as a list by
    // swift-markdown until content is added. The visual change happens when
    // the first content character is typed (e.g., "- H").
  }

  @Test("Type multi-item list with Enter continuation")
  func typeMultiItemList() {
    // Type "- Item 1", Enter, "Item 2", Enter, "Item 3"
    var events: [EditorEvent] = []
    for c in "- Item 1" { events.append(.insertText(String(c))) }
    events.append(.insertNewline)
    for c in "Item 2" { events.append(.insertText(String(c))) }
    events.append(.insertNewline)
    for c in "Item 3" { events.append(.insertText(String(c))) }

    let results = EditorTestHarness.run(
      name: "list-multi-item-typing",
      initial: EditorState(),
      events: events,
      size: NSSize(width: 600, height: 300))

    let finalState = results.last!.state
    #expect(finalState.markdown == "- Item 1\n- Item 2\n- Item 3")

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }
  }

  // MARK: - Cursor movement: inside vs outside

  @Test("Cursor inside list item reveals delimiter, cursor outside hides it")
  func listDelimiterVisibility() {
    let markdown = "- Hello\n\nBody text"
    let initial = EditorState(markdown: markdown, selection: .cursor(4))  // inside list

    let events: [EditorEvent] = [
      .setSelection(.cursor(10)),  // move to body
      .setSelection(.cursor(4)),   // move back to list
    ]

    let results = EditorTestHarness.run(
      name: "list-delimiter-visibility",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 200))

    #expect(results.count == 3)

    // Step 0 (cursor in list) vs step 1 (cursor in body) should differ
    #expect(
      results[0].bitmapHash != results[1].bitmapHash,
      "Cursor inside vs outside list should look different (delimiter vs bullet)")

    // Step 2 (cursor back in list) should match step 0
    #expect(
      results[0].bitmapHash == results[2].bitmapHash,
      "Same cursor position should produce same visual")
  }

  @Test("Cursor at various positions around list item")
  func cursorAtVariousPositions() {
    let markdown = "- Hello World\n\nBody text here"
    // Test many cursor positions
    let positions: [(String, Int)] = [
      ("start-of-marker", 0),
      ("after-marker", 2),
      ("middle-of-content", 7),
      ("end-of-content", 13),
      ("blank-line", 14),
      ("body-start", 15),
      ("body-middle", 20),
      ("body-end", 29),
    ]

    var events: [EditorEvent] = []
    for (_, offset) in positions.dropFirst() {
      events.append(.setSelection(.cursor(offset)))
    }

    let initial = EditorState(
      markdown: markdown, selection: .cursor(positions[0].1))

    let results = EditorTestHarness.run(
      name: "list-cursor-positions",
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

  // MARK: - Multiple list items with cursor on different items

  @Test("Cursor on different list items shows bullet on others")
  func cursorOnDifferentItems() {
    let markdown = "- Item 1\n- Item 2\n- Item 3\n\nBody"

    let positions: [(String, Int)] = [
      ("first-item", 4),       // inside "Item 1"
      ("second-item", 14),     // inside "Item 2"
      ("third-item", 23),      // inside "Item 3"
      ("body", 30),            // in body (all items have bullets)
    ]

    var events: [EditorEvent] = []
    for (_, offset) in positions.dropFirst() {
      events.append(.setSelection(.cursor(offset)))
    }

    let initial = EditorState(
      markdown: markdown, selection: .cursor(positions[0].1))

    let results = EditorTestHarness.run(
      name: "list-cursor-different-items",
      initial: initial,
      events: events,
      size: NSSize(width: 600, height: 300))

    #expect(results.count == positions.count)

    // When cursor is in body (step 3), all items should show bullets.
    // When cursor is on an item, that item shows delimiter, others show bullet.
    // So all four should look different from each other.
    var uniqueHashes = Set<Int>()
    for r in results {
      uniqueHashes.insert(r.bitmapHash)
    }
    #expect(
      uniqueHashes.count >= 3,
      "Expected at least 3 distinct visuals for different cursor positions, got \(uniqueHashes.count)")
  }

  // MARK: - Enter / Backspace visual

  @Test("Enter on empty list item visual: list ends")
  func enterOnEmptyListItemVisual() {
    let events: [EditorEvent] = [
      .insertText("- Item 1"),
      .insertNewline,         // continues list: "- Item 1\n- "
      .insertNewline,         // empty item: removes marker
    ]

    let results = EditorTestHarness.run(
      name: "list-enter-empty-item",
      initial: EditorState(),
      events: events,
      size: NSSize(width: 600, height: 200))

    // After first Enter, should have continued list
    #expect(results[2].state.markdown == "- Item 1\n- ")
    // After second Enter on empty item, marker removed
    #expect(results[3].state.markdown == "- Item 1\n\n")

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath))
    }
  }

  @Test("Backspace removes list marker visual")
  func backspaceRemovesMarkerVisual() {
    let initial = EditorState(markdown: "- Hello", selection: .cursor(2))
    let events: [EditorEvent] = [
      .deleteBackward,  // removes "- ", leaves "Hello"
    ]

    let results = EditorTestHarness.run(
      name: "list-backspace-marker",
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

  @Test("Fresh render matches incremental render for list")
  func determinismList() {
    let results = EditorTestHarness.runTyping(
      name: "determinism-list",
      characters: "- Hello World")

    let finalState = results.last!.state
    let freshBitmap = SnapshotCapture.capture(
      text: finalState.markdown,
      cursorPosition: finalState.selection.head,
      size: NSSize(width: 600, height: 200))

    let incrementalBitmap = NSBitmapImageRep(
      data: try! Data(contentsOf: URL(fileURLWithPath: results.last!.imagePath)))!

    let comparison = BitmapComparator.compare(freshBitmap, incrementalBitmap)
    #expect(comparison.isMatch, "Fresh and incremental renders must match for list")
  }
}
