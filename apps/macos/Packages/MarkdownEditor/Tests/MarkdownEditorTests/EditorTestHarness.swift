import AppKit
import Foundation
import Testing

@testable import MarkdownEditor

/// Test harness for the Elm-architecture editor.
///
/// Accepts an initial state and a sequence of events, runs the update loop,
/// and captures both state and bitmap snapshots at each step. Designed for
/// agent-driven development: agents define scenarios, run tests, and review
/// the resulting images to verify correctness.
@MainActor
enum EditorTestHarness {

  /// Root directory for test artifacts, relative to the package root.
  private static let artifactsRoot: String = {
    // __FILE__ is in Tests/MarkdownEditorTests/, walk up to the package root.
    let thisFile = #filePath
    let testsDir = (thisFile as NSString).deletingLastPathComponent  // MarkdownEditorTests/
    let testRoot = (testsDir as NSString).deletingLastPathComponent  // Tests/
    let packageRoot = (testRoot as NSString).deletingLastPathComponent  // MarkdownEditor/
    return (packageRoot as NSString).appendingPathComponent("test-artifacts")
  }()

  /// The result of processing one event.
  struct StepResult {
    /// The event that was processed.
    let event: EditorEvent
    /// The editor state after the event.
    let state: EditorState
    /// Path to the bitmap snapshot image (PNG).
    let imagePath: String
    /// Hash of the bitmap data for change detection.
    let bitmapHash: Int
  }

  /// Run a sequence of events through the editor and capture state + visuals at each step.
  ///
  /// - Parameters:
  ///   - name: Test name (used for directory naming).
  ///   - initial: Starting editor state.
  ///   - events: Sequence of user events to process.
  ///   - size: Size of the rendered view.
  /// - Returns: Array of step results, one per event.
  static func run(
    name: String,
    initial: EditorState,
    events: [EditorEvent],
    size: NSSize = NSSize(width: 600, height: 400)
  ) -> [StepResult] {
    let dir = "\(artifactsRoot)/\(name)"
    let fm = FileManager.default
    try? fm.removeItem(atPath: dir)
    try? fm.createDirectory(atPath: dir, withIntermediateDirectories: true)

    let components = MarkdownTextViewFactory.create(size: size)
    var results: [StepResult] = []
    var currentState = initial

    // Capture initial state (step 0)
    let initialBitmap = captureState(currentState, components: components, size: size)
    let initialPath = saveBitmap(initialBitmap, name: "step-000-initial", directory: dir)
    let initialHash = bitmapHash(initialBitmap)
    results.append(StepResult(
      event: .setSelection(initial.selection),
      state: currentState,
      imagePath: initialPath,
      bitmapHash: initialHash))

    // Process each event
    for (i, event) in events.enumerated() {
      currentState = EditorUpdate.update(currentState, event: event)
      let bitmap = captureState(currentState, components: components, size: size)
      let path = saveBitmap(
        bitmap, name: String(format: "step-%03d-%@", i + 1, eventName(event)), directory: dir)
      let hash = bitmapHash(bitmap)
      results.append(StepResult(
        event: event, state: currentState, imagePath: path, bitmapHash: hash))
    }

    // Also capture a fresh render of the final state (for determinism check)
    let freshBitmap = captureFresh(currentState, size: size)
    saveBitmap(freshBitmap, name: "final-fresh", directory: dir)

    // Write manifest
    writeManifest(name: name, results: results, directory: dir)

    return results
  }

  /// Run a typing sequence (convenience for character-by-character input).
  static func runTyping(
    name: String,
    characters: String,
    initial: EditorState = EditorState(),
    size: NSSize = NSSize(width: 600, height: 200)
  ) -> [StepResult] {
    let events = characters.map { char -> EditorEvent in
      if char == "\n" {
        return .insertNewline
      } else {
        return .insertText(String(char))
      }
    }
    return run(name: name, initial: initial, events: events, size: size)
  }

  // MARK: - Private

  private static func captureState(
    _ state: EditorState,
    components: MarkdownTextViewFactory.Components,
    size: NSSize
  ) -> NSBitmapImageRep {
    SnapshotCapture.apply(
      text: state.markdown,
      cursorPosition: state.selection.head,
      to: components)
    return SnapshotCapture.renderBitmap(from: components, size: size)
  }

  private static func captureFresh(
    _ state: EditorState, size: NSSize
  ) -> NSBitmapImageRep {
    SnapshotCapture.capture(
      text: state.markdown,
      cursorPosition: state.selection.head,
      size: size)
  }

  @discardableResult
  private static func saveBitmap(
    _ bitmap: NSBitmapImageRep, name: String, directory: String
  ) -> String {
    SnapshotCapture.saveToDisk(bitmap, name: name, directory: directory)
  }

  private static func bitmapHash(_ bitmap: NSBitmapImageRep) -> Int {
    bitmap.representation(using: .png, properties: [:])?.hashValue ?? 0
  }

  private static func eventName(_ event: EditorEvent) -> String {
    switch event {
    case .insertText(let text):
      if text == " " { return "space" }
      if text.count == 1, let c = text.first, c.isLetter || c.isNumber {
        return String(c)
      }
      return "text"
    case .insertNewline: return "newline"
    case .deleteBackward: return "backspace"
    case .deleteForward: return "delete"
    case .setSelection: return "select"
    case .paste: return "paste"
    }
  }

  private static func writeManifest(
    name: String, results: [StepResult], directory: String
  ) {
    var manifest = "# Test: \(name)\n\n"
    manifest += "| Step | Event | Markdown | Cursor | Image |\n"
    manifest += "|------|-------|----------|--------|-------|\n"

    for (i, r) in results.enumerated() {
      let eventStr = describeEvent(r.event)
      let mdEscaped = r.state.markdown
        .replacingOccurrences(of: "\n", with: "\\n")
        .replacingOccurrences(of: "|", with: "\\|")
      let cursorStr: String
      switch r.state.selection {
      case .cursor(let pos): cursorStr = "\(pos)"
      case .range(let a, let h): cursorStr = "\(a)..\(h)"
      }
      let imgFile = URL(fileURLWithPath: r.imagePath).lastPathComponent
      manifest += "| \(i) | \(eventStr) | `\(mdEscaped)` | \(cursorStr) | \(imgFile) |\n"
    }

    manifest += "\nImages directory: `\(directory)/`\n"
    try? manifest.write(
      toFile: "\(directory)/manifest.md", atomically: true, encoding: .utf8)
  }

  private static func describeEvent(_ event: EditorEvent) -> String {
    switch event {
    case .insertText(let text):
      if text == " " { return "Space" }
      if text.count <= 3 { return "`\(text)`" }
      return "Insert(\(text.count) chars)"
    case .insertNewline: return "Enter"
    case .deleteBackward: return "Backspace"
    case .deleteForward: return "Delete"
    case .setSelection(let sel):
      switch sel {
      case .cursor(let pos): return "Cursor(\(pos))"
      case .range(let a, let h): return "Select(\(a)..\(h))"
      }
    case .paste(let text): return "Paste(\(text.count) chars)"
    }
  }
}
