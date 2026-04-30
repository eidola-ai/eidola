import AppKit
import Foundation
import MarkdownEditor
import SwiftUI

// MARK: - Script schema

struct ScriptDocument: Decodable {
  let markdown: String
  let size: [Double]?
  let outDir: String
  let events: [ScriptEvent]
  /// Event-injection mechanism. One of `direct`, `sendEvent`, `cgEvent`.
  /// Defaults to `sendEvent` — the documented bug-reproducing path.
  let injection: String?
  /// Milliseconds to pump the run loop between script events. Default 50.
  let stepDelayMs: Int?
}

struct ScriptEvent: Decodable {
  let type: String
  // Common payloads, all optional and decoded by `type`.
  let name: String?
  let position: Int?
  let location: Int?
  let length: Int?
  let text: String?
  let x: Double?
  let y: Double?
  let ms: Int?
}

// MARK: - JSONL trace appender (script-level events)

@MainActor
enum ScriptTrace {
  static var path: String?

  /// Wall-clock origin for relative timestamps. Captured on first `write`.
  /// Stored as a non-static var to dodge a Swift 6 issue where a global
  /// `static let DispatchTime.now()` initializer triggers a runtime trap on
  /// first access. Date is fine.
  private static var origin: Date?

  static func write(_ event: String, _ payload: [String: Any] = [:]) {
    if origin == nil { origin = Date() }
    let ms: Double = Date().timeIntervalSince(origin ?? Date()) * 1000.0
    var obj: [String: Any] = [
      "event": event,
      "time_ms": ms,
    ]
    if !payload.isEmpty {
      obj["payload"] = payload
    }
    guard let data = try? JSONSerialization.data(
      withJSONObject: obj, options: [.sortedKeys]),
      let str = String(data: data, encoding: .utf8) else {
      FileHandle.standardError.write(Data("{\"event\":\"\(event)\",\"_error\":\"serialization-failed\"}\n".utf8))
      return
    }
    let line = str + "\n"
    if let path {
      if !FileManager.default.fileExists(atPath: path) {
        FileManager.default.createFile(atPath: path, contents: nil)
      }
      if let handle = FileHandle(forWritingAtPath: path) {
        _ = try? handle.seekToEnd()
        if let bytes = line.data(using: .utf8) {
          handle.write(bytes)
        }
        _ = try? handle.close()
      }
    } else {
      FileHandle.standardError.write(Data(line.utf8))
    }
  }
}

// MARK: - Run loop helpers

@MainActor
enum RunLoopPump {
  /// Process pending NSApp events for `ms` milliseconds (or until quiescence).
  static func pump(forMs ms: Int) {
    let deadline = Date().addingTimeInterval(Double(ms) / 1000.0)
    while Date() < deadline {
      // Drain any pending AppKit events with a short timeout each iter.
      let next = Date().addingTimeInterval(0.01)
      while let event = NSApp.nextEvent(
        matching: .any, until: next, inMode: .default, dequeue: true)
      {
        NSApp.sendEvent(event)
      }
      // Also pump the foundation run loop briefly so timers / observers fire.
      RunLoop.current.run(mode: .default, before: Date().addingTimeInterval(0.005))
    }
  }
}

// MARK: - Editor host

/// Hosts a `MarkdownEditor` view in a real `NSWindow` and exposes the
/// underlying `NSTextView` for scripted manipulation.
@MainActor
final class EditorHost {
  let window: NSWindow
  let scrollView: NSScrollView
  /// Updated by polling after each event; the SwiftUI binding flows through
  /// the Coordinator's two-way state binding.
  private var stateBox: StateBox

  /// Shared backing store for the SwiftUI binding.
  @MainActor
  final class StateBox {
    var state: EditorState
    init(_ s: EditorState) { state = s }
  }

  /// Resolve the inner text view inside the scroll view's document hierarchy.
  var textView: NSTextView? {
    scrollView.documentView as? NSTextView
  }

  init(initialMarkdown: String, size: NSSize) {
    let stateBox = StateBox(EditorState(markdown: initialMarkdown, selection: .cursor(0)))
    self.stateBox = stateBox

    let binding = Binding<EditorState>(
      get: { stateBox.state }, set: { stateBox.state = $0 })

    // Build the SwiftUI view, wrap it in an NSHostingView, install in window.
    let host = NSHostingView(rootView: MarkdownEditor(state: binding))
    host.frame = NSRect(origin: .zero, size: size)

    let window = NSWindow(
      contentRect: NSRect(origin: .zero, size: size),
      styleMask: [.titled, .closable],
      backing: .buffered,
      defer: false)
    window.title = "MarkdownEditorScript"
    window.contentView = host
    // Place on-screen so the AppKit layout / event-routing pipeline behaves
    // as it would for a real user. We deliberately do NOT activate the app
    // (policy is .accessory in main()), but we do bring the window fully
    // into a visible state — the TK2 visual-position-preserve heuristic
    // empirically only fires when the layout has been displayed at least
    // once against a visible backing surface.
    window.setFrameOrigin(NSPoint(x: 100, y: 100))
    window.orderFrontRegardless()
    window.makeKeyAndOrderFront(nil)

    self.window = window
    self.scrollView = (host.subviews.first(where: { $0 is NSScrollView }) as? NSScrollView)
      ?? NSScrollView()
  }

  /// After construction, the SwiftUI hosting hierarchy needs at least one
  /// run-loop tick to instantiate `NSScrollView` / `NSTextView`. Pump until
  /// `textView` is non-nil or we time out.
  func waitForTextView(timeoutMs: Int = 2000) {
    let deadline = Date().addingTimeInterval(Double(timeoutMs) / 1000.0)
    while textView == nil && Date() < deadline {
      RunLoopPump.pump(forMs: 20)
    }
    // Also drill via subview tree if SwiftUI hosted differently:
    if textView == nil, let host = window.contentView {
      drillForTextView(in: host)
    }
  }

  private func drillForTextView(in view: NSView) {
    if let tv = view as? NSTextView {
      _resolvedTextView = tv
      return
    }
    for sub in view.subviews {
      drillForTextView(in: sub)
      if _resolvedTextView != nil { return }
    }
  }

  /// Cached after drill; preferred over the scrollView lookup if present.
  private var _resolvedTextView: NSTextView?
  var resolvedTextView: NSTextView? {
    _resolvedTextView ?? textView
  }

  /// Force the text view to be first responder so AppKit-level event
  /// routing has a target.
  func makeFirstResponder() {
    if let tv = resolvedTextView {
      window.makeFirstResponder(tv)
    }
  }
}

// MARK: - Event injection

@MainActor
enum EventInjector {

  enum Mode: String {
    case direct
    case sendEvent
    case cgEvent
  }

  static func dispatchKey(
    name: String, host: EditorHost, mode: Mode
  ) {
    switch mode {
    case .direct:
      directKey(name: name, host: host)
    case .sendEvent:
      sendEventKey(name: name, host: host)
    case .cgEvent:
      // CGEvent path requires accessibility permissions; fall back to
      // sendEvent if the post fails or cannot acquire a tap.
      cgEventKey(name: name, host: host)
    }
  }

  private static func directKey(name: String, host: EditorHost) {
    guard let tv = host.resolvedTextView else { return }
    let n = name.lowercased()
    switch n {
    case "right": tv.moveRight(nil)
    case "left": tv.moveLeft(nil)
    case "up": tv.moveUp(nil)
    case "down": tv.moveDown(nil)
    case "shift+right": tv.moveRightAndModifySelection(nil)
    case "shift+left": tv.moveLeftAndModifySelection(nil)
    case "shift+up": tv.moveUpAndModifySelection(nil)
    case "shift+down": tv.moveDownAndModifySelection(nil)
    case "home": tv.moveToBeginningOfLine(nil)
    case "end": tv.moveToEndOfLine(nil)
    case "enter", "return": tv.insertNewline(nil)
    case "shift+enter", "shift+return":
      tv.insertLineBreak(nil)
    case "backspace":
      tv.deleteBackward(nil)
    case "delete":
      tv.deleteForward(nil)
    default:
      ScriptTrace.write("script.unknown_key", ["name": name])
    }
  }

  private static func sendEventKey(name: String, host: EditorHost) {
    guard let tv = host.resolvedTextView else { return }
    host.makeFirstResponder()
    let (keyCode, chars, modifiers) = keySpec(forName: name)
    guard let keyCode else {
      ScriptTrace.write("script.unknown_key", ["name": name])
      return
    }
    // Build a key-down then key-up event. NSApp.sendEvent dispatches them
    // through the normal AppKit responder chain — moveRight goes through
    // doCommandBy / interpretKeyEvents. This is the closest path to a real
    // user keypress short of an OS-level CGEvent.
    let now = ProcessInfo.processInfo.systemUptime
    if let down = NSEvent.keyEvent(
      with: .keyDown,
      location: .zero,
      modifierFlags: modifiers,
      timestamp: now,
      windowNumber: host.window.windowNumber,
      context: nil,
      characters: chars,
      charactersIgnoringModifiers: chars,
      isARepeat: false,
      keyCode: keyCode)
    {
      // Ensure the window is key so the event lands on the first responder.
      // Without this, NSApp.sendEvent posts to a nil keyWindow and the
      // event is dropped silently.
      host.window.makeKeyAndOrderFront(nil)
      _ = tv  // silence unused
      NSApp.sendEvent(down)
    }
    if let up = NSEvent.keyEvent(
      with: .keyUp,
      location: .zero,
      modifierFlags: modifiers,
      timestamp: now,
      windowNumber: host.window.windowNumber,
      context: nil,
      characters: chars,
      charactersIgnoringModifiers: chars,
      isARepeat: false,
      keyCode: keyCode)
    {
      NSApp.sendEvent(up)
    }
  }

  private static func cgEventKey(name: String, host: EditorHost) {
    let (keyCode, _, modifiers) = keySpec(forName: name)
    guard let keyCode else { return }
    host.makeFirstResponder()
    host.window.makeKeyAndOrderFront(nil)
    let src = CGEventSource(stateID: .hidSystemState)
    if modifiers.contains(.shift) {
      let modDown = CGEvent(
        keyboardEventSource: src, virtualKey: 0x38 /* shift */, keyDown: true)
      modDown?.post(tap: .cgAnnotatedSessionEventTap)
    }
    let down = CGEvent(
      keyboardEventSource: src, virtualKey: CGKeyCode(keyCode), keyDown: true)
    down?.flags = cgFlags(modifiers)
    down?.post(tap: .cgAnnotatedSessionEventTap)
    let up = CGEvent(
      keyboardEventSource: src, virtualKey: CGKeyCode(keyCode), keyDown: false)
    up?.flags = cgFlags(modifiers)
    up?.post(tap: .cgAnnotatedSessionEventTap)
    if modifiers.contains(.shift) {
      let modUp = CGEvent(
        keyboardEventSource: src, virtualKey: 0x38, keyDown: false)
      modUp?.post(tap: .cgAnnotatedSessionEventTap)
    }
  }

  private static func cgFlags(_ flags: NSEvent.ModifierFlags) -> CGEventFlags {
    var out: CGEventFlags = []
    if flags.contains(.shift) { out.insert(.maskShift) }
    if flags.contains(.command) { out.insert(.maskCommand) }
    if flags.contains(.option) { out.insert(.maskAlternate) }
    if flags.contains(.control) { out.insert(.maskControl) }
    return out
  }

  /// Returns (keyCode, chars-for-keyEvent, modifierFlags) for a logical
  /// key name. Key codes match the macOS HID virtual key constants.
  private static func keySpec(
    forName name: String
  ) -> (UInt16?, String, NSEvent.ModifierFlags) {
    let n = name.lowercased()
    var modifiers: NSEvent.ModifierFlags = []
    var rest = n
    if rest.hasPrefix("shift+") {
      modifiers.insert(.shift)
      rest = String(rest.dropFirst("shift+".count))
    }
    switch rest {
    case "right":
      // NSRightArrowFunctionKey = 0xF703; HID virtual keycode = 0x7C.
      return (0x7C, "\u{F703}", modifiers)
    case "left":
      // NSLeftArrowFunctionKey = 0xF702; HID virtual keycode = 0x7B.
      return (0x7B, "\u{F702}", modifiers)
    case "up":
      // NSUpArrowFunctionKey = 0xF700; HID virtual keycode = 0x7E.
      return (0x7E, "\u{F700}", modifiers)
    case "down":
      // NSDownArrowFunctionKey = 0xF701; HID virtual keycode = 0x7D.
      return (0x7D, "\u{F701}", modifiers)
    case "home":
      return (0x73, "\u{F729}", modifiers)
    case "end":
      return (0x77, "\u{F72B}", modifiers)
    case "enter", "return":
      return (0x24, "\r", modifiers)
    case "backspace":
      return (0x33, "\u{0008}", modifiers)
    case "delete":
      return (0x75, "\u{F728}", modifiers)
    default:
      return (nil, "", modifiers)
    }
  }
}

// MARK: - Snapshot capture

@MainActor
enum SnapshotWriter {
  static func capture(host: EditorHost, name: String, outDir: String) -> String? {
    guard let tv = host.resolvedTextView else { return nil }
    if let tlm = tv.textLayoutManager {
      tlm.ensureLayout(for: tlm.documentRange)
    }
    tv.needsDisplay = true
    tv.displayIfNeeded()

    let bounds = tv.bounds
    let w = max(1, Int(bounds.width))
    let h = max(1, Int(bounds.height))
    guard let bitmap = NSBitmapImageRep(
      bitmapDataPlanes: nil,
      pixelsWide: w,
      pixelsHigh: h,
      bitsPerSample: 8,
      samplesPerPixel: 4,
      hasAlpha: true,
      isPlanar: false,
      colorSpaceName: .calibratedRGB,
      bytesPerRow: 0,
      bitsPerPixel: 0)
    else { return nil }
    tv.cacheDisplay(in: bounds, to: bitmap)
    let path = "\(outDir)/\(name).png"
    if let data = bitmap.representation(using: .png, properties: [:]) {
      try? data.write(to: URL(fileURLWithPath: path))
      return path
    }
    return nil
  }
}

// MARK: - Manifest writer

@MainActor
final class ManifestBuilder {
  struct Row {
    let step: Int
    let event: String
    let cursor: String
    let snapshot: String?
  }
  var rows: [Row] = []

  func write(toPath path: String, scriptName: String) {
    var md = "# Script: \(scriptName)\n\n"
    md += "| Step | Event | Cursor | Snapshot |\n"
    md += "|------|-------|--------|----------|\n"
    for r in rows {
      let snap = r.snapshot ?? ""
      md += "| \(r.step) | \(r.event) | \(r.cursor) | \(snap) |\n"
    }
    try? md.write(toFile: path, atomically: true, encoding: .utf8)
  }
}

// MARK: - Driver

@MainActor
func runScript(at scriptPath: String) -> Int32 {
  guard let scriptData = FileManager.default.contents(atPath: scriptPath) else {
    FileHandle.standardError.write(Data("error: cannot read script at \(scriptPath)\n".utf8))
    return 2
  }
  let decoder = JSONDecoder()
  let script: ScriptDocument
  do {
    script = try decoder.decode(ScriptDocument.self, from: scriptData)
  } catch {
    FileHandle.standardError.write(Data("error: cannot decode script: \(error)\n".utf8))
    return 2
  }

  // Prepare output dir.
  try? FileManager.default.createDirectory(
    atPath: script.outDir, withIntermediateDirectories: true)
  // Truncate trace.jsonl so re-runs start fresh.
  let traceFile = "\(script.outDir)/trace.jsonl"
  FileManager.default.createFile(atPath: traceFile, contents: nil)

  // Wire the production-side DebugTrace + the script-level trace into the
  // same file so ordering is preserved.
  DebugTrace.outputFilePath = traceFile
  DebugTrace.enabled = true
  ScriptTrace.path = traceFile

  let mode = EventInjector.Mode(rawValue: script.injection ?? "sendEvent") ?? .sendEvent
  let stepDelay = script.stepDelayMs ?? 50

  let size: NSSize
  if let s = script.size, s.count == 2 {
    size = NSSize(width: s[0], height: s[1])
  } else {
    size = NSSize(width: 800, height: 400)
  }

  ScriptTrace.write("script.start", [
    "markdown": script.markdown,
    "out_dir": script.outDir,
    "injection": mode.rawValue,
    "step_delay_ms": stepDelay,
    "events_count": script.events.count,
  ])

  let host = EditorHost(initialMarkdown: script.markdown, size: size)
  // Pump until the SwiftUI host instantiates the text view.
  host.waitForTextView()
  host.makeFirstResponder()
  // Allow first render to settle.
  RunLoopPump.pump(forMs: 100)
  let isFR = host.window.firstResponder === host.resolvedTextView
  let hasTLM = host.resolvedTextView?.textLayoutManager != nil
  ScriptTrace.write("script.host_ready", [
    "is_first_responder": isFR,
    "has_layout_manager": hasTLM,
  ])

  let manifest = ManifestBuilder()
  var stepIndex = 0

  func cursorString() -> String {
    guard let tv = host.resolvedTextView else { return "?" }
    let r = tv.selectedRange()
    return r.length == 0 ? "\(r.location)" : "\(r.location)..\(r.location + r.length)"
  }

  for ev in script.events {
    let kind = ev.type.lowercased()
    switch kind {
    case "snapshot":
      let name = ev.name ?? String(format: "step-%03d-snapshot", stepIndex)
      let path = SnapshotWriter.capture(host: host, name: name, outDir: script.outDir)
      ScriptTrace.write("script.snapshot", [
        "name": name,
        "path": path ?? "",
        "cursor": cursorString(),
      ])
      manifest.rows.append(.init(
        step: stepIndex, event: "snapshot:\(name)",
        cursor: cursorString(), snapshot: "\(name).png"))
    case "set_cursor":
      let pos = ev.position ?? 0
      ScriptTrace.write("script.set_cursor", ["position": pos])
      host.resolvedTextView?.setSelectedRange(NSRange(location: pos, length: 0))
      manifest.rows.append(.init(
        step: stepIndex, event: "set_cursor:\(pos)",
        cursor: cursorString(), snapshot: nil))
    case "select_range":
      let loc = ev.location ?? 0
      let len = ev.length ?? 0
      ScriptTrace.write("script.select_range", ["location": loc, "length": len])
      host.resolvedTextView?.setSelectedRange(NSRange(location: loc, length: len))
      manifest.rows.append(.init(
        step: stepIndex, event: "select_range:\(loc)..\(loc+len)",
        cursor: cursorString(), snapshot: nil))
    case "key":
      let name = ev.name ?? "right"
      ScriptTrace.write("script.key", ["name": name, "injection": mode.rawValue])
      EventInjector.dispatchKey(name: name, host: host, mode: mode)
      manifest.rows.append(.init(
        step: stepIndex, event: "key:\(name)",
        cursor: cursorString(), snapshot: nil))
    case "type":
      let text = ev.text ?? ""
      ScriptTrace.write("script.type", ["text": text])
      host.resolvedTextView?.insertText(text, replacementRange: NSRange(location: NSNotFound, length: 0))
      manifest.rows.append(.init(
        step: stepIndex, event: "type:\(text)",
        cursor: cursorString(), snapshot: nil))
    case "click":
      let x = ev.x ?? 0
      let y = ev.y ?? 0
      ScriptTrace.write("script.click", ["x": x, "y": y])
      if let tv = host.resolvedTextView {
        let pt = NSPoint(x: x, y: y)
        let down = NSEvent.mouseEvent(
          with: .leftMouseDown, location: pt,
          modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
          windowNumber: host.window.windowNumber, context: nil,
          eventNumber: 0, clickCount: 1, pressure: 1)
        let up = NSEvent.mouseEvent(
          with: .leftMouseUp, location: pt,
          modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
          windowNumber: host.window.windowNumber, context: nil,
          eventNumber: 0, clickCount: 1, pressure: 0)
        if let down { tv.mouseDown(with: down) }
        if let up { tv.mouseUp(with: up) }
      }
      manifest.rows.append(.init(
        step: stepIndex, event: "click:\(x),\(y)",
        cursor: cursorString(), snapshot: nil))
    case "wait":
      let ms = ev.ms ?? 50
      ScriptTrace.write("script.wait", ["ms": ms])
      RunLoopPump.pump(forMs: ms)
      manifest.rows.append(.init(
        step: stepIndex, event: "wait:\(ms)ms",
        cursor: cursorString(), snapshot: nil))
    default:
      ScriptTrace.write("script.unknown_event", ["type": ev.type])
    }
    // Force layout + display so any TK2 visual-position-preserve heuristic
    // tied to actual paint cycles has a chance to fire. The off-screen
    // version of the runner observed a clean walk because the layout
    // controller never re-laid the viewport between keypresses; pushing
    // ensureLayout + displayIfNeeded here mirrors what the live demo gets
    // for free from its 60Hz draw loop.
    if let tv = host.resolvedTextView {
      if let tlm = tv.textLayoutManager {
        tlm.textViewportLayoutController.layoutViewport()
        tlm.ensureLayout(for: tlm.documentRange)
      }
      tv.needsDisplay = true
      tv.displayIfNeeded()
    }
    // Pump the run loop so any deferred AppKit work (selection changes,
    // layout invalidation, etc.) finishes before the next event.
    RunLoopPump.pump(forMs: stepDelay)
    stepIndex += 1
  }

  // Final manifest write.
  manifest.write(
    toPath: "\(script.outDir)/manifest.md",
    scriptName: (scriptPath as NSString).lastPathComponent)
  ScriptTrace.write("script.end", ["final_cursor": cursorString()])
  DebugTrace.flush()
  return 0
}

// MARK: - Entry point

@MainActor
func main() -> Int32 {
  let args = ProcessInfo.processInfo.arguments
  guard args.count >= 2 else {
    FileHandle.standardError.write(Data("usage: MarkdownEditorScript <script.json>\n".utf8))
    return 1
  }
  let scriptPath = args[1]

  // Build the NSApp first so SwiftUI hosting + AppKit events work.
  let app = NSApplication.shared
  app.setActivationPolicy(.accessory)
  // Don't show the app — we keep the window off-screen but with a real
  // backing surface so layout / event routing work.
  return runScript(at: scriptPath)
}

// Bridge into the main actor — `exit` is a pure C function so we have to
// hop. Use a synchronous pump via NSApp event loop.
let exitCode = MainActor.assumeIsolated { main() }
exit(exitCode)
