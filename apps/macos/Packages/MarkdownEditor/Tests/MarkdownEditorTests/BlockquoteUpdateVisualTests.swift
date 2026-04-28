import AppKit
import Foundation
import Testing

@testable import MarkdownEditor

/// Visual tests for blockquote keyboard behavior (Enter continuation flow).
@Suite("Blockquote Update Visual Tests")
@MainActor
struct BlockquoteUpdateVisualTests {

  @Test("Type blockquote, Enter to continue, Enter on empty to end")
  func blockquoteEnterContinuationFlow() {
    // Type "> Hello", Enter (continues), "World", Enter (continues), Enter (ends)
    var events: [EditorEvent] = []
    for c in "> Hello" { events.append(.insertText(String(c))) }
    events.append(.insertNewline)
    for c in "World" { events.append(.insertText(String(c))) }
    events.append(.insertNewline)
    events.append(.insertNewline)

    let results = EditorTestHarness.run(
      name: "blockquote-enter-continuation",
      initial: EditorState(),
      events: events,
      size: NSSize(width: 600, height: 300))

    // After typing "> Hello" and Enter, should have "> Hello\n> "
    let afterFirstEnter = results[8]  // 1 initial + 7 chars + 1 enter = index 8
    #expect(afterFirstEnter.state.markdown == "> Hello\n> ")

    // After typing "World" and Enter, should have "> Hello\n> World\n> "
    let afterSecondEnter = results[14]  // +5 chars + 1 enter = index 14
    #expect(afterSecondEnter.state.markdown == "> Hello\n> World\n> ")

    // After final Enter on empty blockquote, should end the blockquote
    let afterThirdEnter = results[15]  // +1 enter = index 15
    #expect(afterThirdEnter.state.markdown == "> Hello\n> World\n\n")

    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }
  }
}
