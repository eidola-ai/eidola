import AppKit

/// Identifies a block-renderer family. Renderers register a factory under a
/// tag with `BlockRendererRegistry`; the renderer emits the same tag in the
/// `BlockRendererSpec` so the applicator can dispatch back to the right
/// factory when reconciling specs against live hosts.
///
/// Tags are intentionally `String`-backed (rather than enum-typed) so future
/// renderers can be added without touching this file. The handful of
/// well-known tags used by the in-tree renderers are exposed as static
/// constants below.
public struct BlockTypeTag: Hashable, RawRepresentable, Sendable {
  public let rawValue: String
  public init(rawValue: String) { self.rawValue = rawValue }
}

extension BlockTypeTag {
  /// A fenced code block. Source IS the display.
  public static let codeBlock = BlockTypeTag(rawValue: "codeBlock")
  /// An inline image. Visual when cursor outside; source revealed when inside.
  public static let image = BlockTypeTag(rawValue: "image")
  /// A math block. Visual when cursor outside; source revealed when inside.
  public static let math = BlockTypeTag(rawValue: "math")
}

/// How a renderer behaves with respect to the cursor moving into its range.
public enum BlockRevealMode: Sendable, Equatable {
  /// Source IS the display. Typing edits in place. The renderer surfaces
  /// the same characters that exist in the main storage; cursor entering
  /// the range does not flip the visual. Used by code blocks.
  case editInPlace
  /// Visual when cursor is outside the range; raw markdown source (rendered
  /// in code-block style) when the cursor is inside. Used by images, math,
  /// diagrams, embeds — anything whose visual is non-textual.
  case cursorConditional
}

/// One per block in the source that wants a custom-view renderer.
/// Produced by `MarkdownRenderer.render` and consumed by
/// `TextKit2RenderApplicator.apply` to drive `BlockRendererRegistry`
/// reconciliation.
public struct BlockRendererSpec {
  /// Source range, including any leading/trailing fences/delimiters.
  public let range: NSRange
  /// Tag used to look up the renderer factory in `BlockRendererRegistry`.
  public let blockTypeTag: BlockTypeTag
  /// Reveal mode the renderer should adopt for this block.
  public let mode: BlockRevealMode
  /// Pre-computed reserved height (in points) for the embedded view's
  /// vertical region. The applicator computes this from font metrics so
  /// AppKit can perform layout without consulting the live renderer.
  public let reservedHeight: CGFloat
  /// Renderer-specific opaque payload (e.g. resolved image URL, language
  /// hint, parsed math AST). Carried verbatim through the pipeline.
  public let payload: Any?

  public init(
    range: NSRange,
    blockTypeTag: BlockTypeTag,
    mode: BlockRevealMode,
    reservedHeight: CGFloat,
    payload: Any? = nil
  ) {
    self.range = range
    self.blockTypeTag = blockTypeTag
    self.mode = mode
    self.reservedHeight = reservedHeight
    self.payload = payload
  }
}

/// Implemented by each block-type adapter. One instance lives per host;
/// the registry's factory builds a fresh one per attachment instance.
///
/// Renderers never own canonical markdown state — they read source through
/// `host.sourceText()` and forward mutations back through the host's
/// helpers so the main `NSTextStorage` remains the single source of truth.
@MainActor
public protocol BlockRenderer: AnyObject {
  /// Build (once) the AppKit view that AppKit will hand to TK2 as the
  /// attachment's view. Called from the view provider's `loadView()`.
  func makeView(host: BlockRenderHost) -> NSView

  /// Called whenever the spec for this attachment changes (typically
  /// because the source range moved or the payload changed). The renderer
  /// uses `host.sourceText()` to read current source and reflows.
  func update(spec: BlockRendererSpec, host: BlockRenderHost)

  /// Called when the cursor enters or leaves the renderer's range.
  /// `cursorConditional` renderers flip between rendered and raw mode here.
  /// `editInPlace` renderers may use it for chrome (e.g. show a "copy" button).
  func cursorPresenceChanged(_ inside: Bool, host: BlockRenderHost)

  /// Desired bounds the renderer will draw within. Reported back to TK2 via
  /// `NSTextAttachment.attachmentBounds(for:textContainer:proposedLineFragment:position:)`.
  /// Should remain stable across cursor enter/leave per the "fixed visual
  /// region" rule in the design doc.
  func desiredBounds(host: BlockRenderHost) -> CGRect

  /// Cleanup. Called when the host is being retired (range no longer in
  /// any spec, or text view going away).
  func tearDown()
}
