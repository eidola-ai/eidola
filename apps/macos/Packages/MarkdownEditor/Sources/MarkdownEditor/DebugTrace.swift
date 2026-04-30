import Foundation

/// Lightweight, opt-in tracing for diagnosing TK2 cursor / apply pipeline
/// issues. All `log` calls compile to a single check on `DebugTrace.enabled`
/// when tracing is off, so it's safe to leave sprinkled throughout
/// production code paths.
///
/// Two ways to enable:
///   1. Set `DebugTrace.enabled = true` programmatically (used by the
///      `MarkdownEditorScript` runner).
///   2. Set the env var `MARKDOWN_EDITOR_TRACE=1` before launching.
///
/// Output target precedence:
///   1. If `outputFilePath` is set, append JSONL there.
///   2. Else if `MARKDOWN_EDITOR_TRACE_FILE` env var is set, append there.
///   3. Else write to stderr.
///
/// Each log line is a single JSON object terminated with `\n`:
///   `{"event":"apply.start","time_ms":<ms>,"payload":{...}}`
///
/// `time_ms` is monotonic-ish — milliseconds since the first `log` call in
/// the process — so the runner's snapshot/event ordering is preserved
/// without depending on wallclock skew.
public enum DebugTrace {
  /// Master switch. Default is `false`. Set this to `true` to start
  /// emitting trace lines. Backed by a thread-safe atomic-ish accessor
  /// (single Bool guarded by `lock`) so the storage is safe under Swift 6
  /// strict concurrency without forcing a global actor on a logging API
  /// that's called from many isolation contexts.
  public static var enabled: Bool {
    get {
      lock.lock()
      defer { lock.unlock() }
      return _enabled
    }
    set {
      lock.lock()
      defer { lock.unlock() }
      _enabled = newValue
    }
  }
  nonisolated(unsafe) private static var _enabled: Bool = {
    if let v = ProcessInfo.processInfo.environment["MARKDOWN_EDITOR_TRACE"],
      v == "1" || v.lowercased() == "true"
    {
      return true
    }
    return false
  }()

  /// Optional file path to append trace lines to. If `nil`, trace lines
  /// go to stderr. Set programmatically by `MarkdownEditorScript`.
  public static var outputFilePath: String? {
    get {
      lock.lock()
      defer { lock.unlock() }
      return _outputFilePath
    }
    set {
      lock.lock()
      defer { lock.unlock() }
      _outputFilePath = newValue
    }
  }
  nonisolated(unsafe) private static var _outputFilePath: String? = ProcessInfo
    .processInfo.environment["MARKDOWN_EDITOR_TRACE_FILE"]

  /// Lazily-resolved file handle. Reset by setting `outputFilePath`.
  nonisolated(unsafe) private static var cachedHandle: FileHandle?
  nonisolated(unsafe) private static var cachedHandlePath: String?

  /// Process-start `Date` for relative timestamps; resolved on first log.
  /// `Date` is used instead of `DispatchTime.now().uptimeNanoseconds`
  /// because under Swift 6 strict-concurrency the latter as a `static let`
  /// initializer trips a runtime check on first access.
  nonisolated(unsafe) private static var origin: Date?
  /// Lock for serializing writes from any thread.
  private static let lock = NSLock()

  /// Emit a trace line with an event name and an optional payload.
  ///
  /// The payload values must be JSON-serializable (`String`, `Int`,
  /// `Double`, `Bool`, `[Any]`, `[String: Any]`). Arrays / dicts may be
  /// nested.
  public static func log(_ event: String, _ payload: [String: Any] = [:]) {
    lock.lock()
    defer { lock.unlock() }
    guard _enabled else { return }

    if origin == nil {
      origin = Date()
    }
    let elapsedMs = Date().timeIntervalSince(origin ?? Date()) * 1000.0

    var obj: [String: Any] = [
      "event": event,
      "time_ms": elapsedMs,
    ]
    if !payload.isEmpty {
      obj["payload"] = payload
    }

    let line: String
    if let data = try? JSONSerialization.data(
      withJSONObject: obj, options: [.fragmentsAllowed, .sortedKeys]),
      let s = String(data: data, encoding: .utf8)
    {
      line = s + "\n"
    } else {
      line = "{\"event\":\"\(event)\",\"time_ms\":\(elapsedMs),\"_error\":\"serialization-failed\"}\n"
    }

    writeLocked(line)
  }

  /// Force-flush any buffered output. Important to call between snapshot
  /// boundaries in the runner so the trace ordering is observable.
  public static func flush() {
    lock.lock()
    defer { lock.unlock() }
    cachedHandle?.synchronizeFile()
  }

  // MARK: - Private

  /// `lock` must be held.
  private static func writeLocked(_ line: String) {
    if let path = _outputFilePath {
      let handle: FileHandle?
      if cachedHandlePath == path, let h = cachedHandle {
        handle = h
      } else {
        // Truncate or create on first open per path.
        let fm = FileManager.default
        if !fm.fileExists(atPath: path) {
          fm.createFile(atPath: path, contents: nil)
        }
        handle = FileHandle(forWritingAtPath: path)
        _ = try? handle?.seekToEnd()
        cachedHandle = handle
        cachedHandlePath = path
      }
      if let data = line.data(using: .utf8) {
        handle?.write(data)
      }
    } else {
      // stderr
      FileHandle.standardError.write(Data(line.utf8))
    }
  }
}
