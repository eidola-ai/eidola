import AppKit
import ObjectiveC

/// Process-wide registry of block-renderer factories plus a per-text-view
/// host registry.
///
/// Two responsibilities live here, intentionally bundled because they share
/// the same dispatch keys (`BlockTypeTag`):
///
/// 1. **Factory map.** `register(_:factory:)` associates a tag with a
///    closure that builds a fresh `BlockRenderer`. The applicator looks up
///    the factory when materializing a host for a new spec.
/// 2. **Live host registry.** Per-text-view, the registry tracks every
///    live `BlockRenderHost`. `reconcile(for:specs:)` is the per-`apply`
///    entry point: it creates hosts for new spec ranges, retires hosts
///    whose ranges no longer appear, and forwards spec changes to existing
///    hosts via `BlockRenderHost.updateSpec(_:)`.
///
/// Hosts are keyed by `(textView, range.location)`. Two specs with the
/// same starting offset are considered the same logical block — even if
/// the length / payload changed — so the host (and its embedded view) are
/// reused. This stabilizes view identity across ordinary edits inside a
/// block (e.g. typing a character into a code block grows the range by one
/// without retiring the host).
@MainActor
public final class BlockRendererRegistry {

  /// Process-wide singleton. Renderers register at module load time;
  /// callers reconcile per `apply`.
  public static let shared = BlockRendererRegistry()

  private var factories: [BlockTypeTag: () -> BlockRenderer] = [:]

  /// Per-text-view host tables. Outer key is an `ObjectIdentifier` of the
  /// `NSTextView`; inner key is the host's `spec.range.location` (the
  /// stable identity within a text view).
  private var hostsByTextView: [ObjectIdentifier: [Int: BlockRenderHost]] = [:]

  /// Per-text-view cache of the last spec list `reconcile` actually
  /// processed. The applicator consults this for an equality check before
  /// calling `reconcile` so a no-op apply (selection-only update where the
  /// spec list hasn't changed) doesn't churn the host table or notify
  /// renderers spuriously.
  private var lastReconciledSpecsByTextView: [ObjectIdentifier: [BlockRendererSpec]] = [:]

  init() {
    // The real `CodeBlockRenderer` (Phase 2.2) hosts an embedded
    // `NSScrollView { NSTextView }` for the code-block source. Registered
    // at singleton-init time so the registry is ready before any text
    // view exists. Tests / spikes that need to override this re-register
    // their own factory and restore in `defer`.
    register(.codeBlock) { CodeBlockRenderer() }
  }

  // MARK: - Factory registration

  /// Register a renderer factory under `tag`. The factory is invoked once
  /// per host (one host per attachment). Re-registration overwrites — the
  /// demo and tests rely on this to install bespoke renderers for spike
  /// scenarios.
  public func register(_ tag: BlockTypeTag, factory: @escaping () -> BlockRenderer) {
    factories[tag] = factory
  }

  /// Look up the factory for `tag`, or `nil` if unregistered.
  public func factory(for tag: BlockTypeTag) -> (() -> BlockRenderer)? {
    factories[tag]
  }

  // MARK: - Host reconciliation

  /// Diagnostic counter incremented each time `reconcile(for:specs:)`
  /// actually executes its host-table mutation pass. Tests use this to
  /// assert that the dedup short-circuit in
  /// `reconcileIfChanged(for:specs:)` is firing.
  private(set) var reconcileExecutionCount: Int = 0

  /// Reconcile only when `specs` differ from the last-applied list for
  /// `textView`. Otherwise no-op. The applicator calls this on every
  /// `apply()` (including selection-only updates that don't actually
  /// change the spec list) so the equality check needs to be cheap —
  /// hence comparison on (range, tag, mode, reservedHeight) rather than
  /// the renderer-opaque `payload`.
  @discardableResult
  public func reconcileIfChanged(
    for textView: NSTextView, specs: [BlockRendererSpec]
  ) -> [BlockRenderHost] {
    let key = ObjectIdentifier(textView)
    if let last = lastReconciledSpecsByTextView[key],
      Self.specListsEqual(last, specs)
    {
      // No change → return the existing hosts without touching the table.
      let table = hostsByTextView[key] ?? [:]
      return table.keys.sorted().compactMap { table[$0] }
    }
    lastReconciledSpecsByTextView[key] = specs
    return reconcile(for: textView, specs: specs)
  }

  private static func specListsEqual(
    _ a: [BlockRendererSpec], _ b: [BlockRendererSpec]
  ) -> Bool {
    guard a.count == b.count else { return false }
    for (l, r) in zip(a, b) {
      guard l.range == r.range,
        l.blockTypeTag == r.blockTypeTag,
        l.mode == r.mode,
        l.reservedHeight == r.reservedHeight
      else { return false }
    }
    return true
  }

  /// Reconcile the registry's live hosts for `textView` against `specs`.
  ///
  /// - Hosts whose `spec.range.location` no longer appears in `specs` are
  ///   disposed and removed.
  /// - Hosts whose location still appears get `updateSpec(_:)` called with
  ///   the new spec (range length / payload may have changed).
  /// - New locations get a fresh host built from the registered factory.
  ///   If no factory is registered for the spec's tag, the spec is
  ///   silently skipped — emitting a spec for an unregistered tag is a
  ///   bug in the renderer, but failing here would crash the editor.
  ///
  /// Returns the (now reconciled) host table for the text view, ordered
  /// by location. Useful for tests; production callers can ignore.
  @discardableResult
  public func reconcile(
    for textView: NSTextView, specs: [BlockRendererSpec]
  ) -> [BlockRenderHost] {
    reconcileExecutionCount += 1
    let key = ObjectIdentifier(textView)
    var hosts = hostsByTextView[key] ?? [:]

    let incomingLocations = Set(specs.map { $0.range.location })

    // 1. Retire hosts whose range is gone.
    for (location, host) in hosts where !incomingLocations.contains(location) {
      host.dispose()
      hosts.removeValue(forKey: location)
    }

    // 2. Update existing, create new.
    for spec in specs {
      if let existing = hosts[spec.range.location] {
        existing.updateSpec(spec)
      } else if let factory = factories[spec.blockTypeTag] {
        let host = BlockRenderHost(spec: spec, textView: textView, rendererFactory: factory)
        hosts[spec.range.location] = host
      }
    }

    hostsByTextView[key] = hosts
    return hosts.keys.sorted().compactMap { hosts[$0] }
  }

  /// Look up the host for a specific text view + range start. Used by
  /// `BlockAttachment` to resolve its host when the view provider asks.
  public func host(for textView: NSTextView, atSourceOffset offset: Int) -> BlockRenderHost? {
    hostsByTextView[ObjectIdentifier(textView)]?[offset]
  }

  /// All live hosts for `textView`, sorted by source offset.
  public func hosts(for textView: NSTextView) -> [BlockRenderHost] {
    let table = hostsByTextView[ObjectIdentifier(textView)] ?? [:]
    return table.keys.sorted().compactMap { table[$0] }
  }

  /// Discard all hosts for a text view (e.g. when the view is going away).
  public func dropAll(for textView: NSTextView) {
    let key = ObjectIdentifier(textView)
    if let hosts = hostsByTextView[key] {
      for host in hosts.values { host.dispose() }
    }
    hostsByTextView.removeValue(forKey: key)
    lastReconciledSpecsByTextView.removeValue(forKey: key)
  }

  // MARK: - Selection notifications

  /// Called by the Coordinator on `textViewDidChangeSelection`. Walks the
  /// live hosts for `textView` and fires `cursorPresenceChanged(_:host:)`
  /// only on hosts whose inside/outside state flipped.
  public func notifySelectionChanged(textView: NSTextView, newRange: NSRange) {
    let hosts = hostsByTextView[ObjectIdentifier(textView)]?.values ?? Dictionary<Int, BlockRenderHost>().values
    for host in hosts {
      let nowInside = BlockRenderHost.rangeOverlapsCursor(host.spec.range, cursor: newRange)
      if nowInside != host.lastInside {
        host.lastInside = nowInside
        host.renderer?.cursorPresenceChanged(nowInside, host: host)
      }
    }
  }

  /// Predicate exposed for `TextKit2MarkdownTextView`'s selection-snap
  /// helper: returns the spec range of any live `cursorConditional` host
  /// for `textView` whose range overlaps `proposed`. Phase 2.1 has no
  /// cursor-conditional renderers in flight (the no-op renderer is
  /// `editInPlace`), so this returns `nil` in practice — but the snap
  /// helper is wired to consume it for Phase 2.2 / image renderer.
  public func cursorConditionalRange(
    for textView: NSTextView, overlapping proposed: NSRange
  ) -> NSRange? {
    let table = hostsByTextView[ObjectIdentifier(textView)] ?? [:]
    for host in table.values {
      guard host.spec.mode == .cursorConditional else { continue }
      if NSIntersectionRange(host.spec.range, proposed).length > 0
        || (proposed.length == 0
          && (proposed.location == host.spec.range.location
            || proposed.location == host.spec.range.location + host.spec.range.length))
      {
        return host.spec.range
      }
    }
    return nil
  }
}
