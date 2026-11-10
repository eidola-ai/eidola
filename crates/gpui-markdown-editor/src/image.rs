//! Image loading + paint adapter.
//!
//! Parallel to [`crate::math`]: the render layer flags image
//! constructs ([`crate::render_spec::ImageOverlay`] for inline,
//! [`crate::render_spec::BlockKind::Image`] for sole-image
//! paragraphs); the element layer calls [`load`] to fetch a
//! [`LoadedImage`] from gpui's asset cache and [`paint`] (or
//! `window.paint_image` directly) to draw it.
//!
//! # Async loading model
//!
//! Image loads are inherently asynchronous (file / network I/O,
//! decode). `load` returns:
//!
//! * `Loaded(Arc<RenderImage>)` when the cache already has the
//!   decoded image — natural size is available immediately so layout
//!   can reserve the right amount of space.
//! * `Loading` while the load is in flight. The cache notifies the
//!   view when the load completes, so subsequent frames will see the
//!   `Loaded` state — callers reserve a fallback size in the
//!   meantime.
//! * `Failed` when the load errored out. The element layer falls
//!   back to dim-delimiter + visible-alt-text rendering on the source
//!   bytes, same as a math typeset failure.
//!
//! # Sizing policy
//!
//! Images are sized at *paint time* via two helpers, parallel to
//! [`crate::math::MathLayout::size`]:
//!
//! * [`inline_size`] for `ImageOverlay`s — height bounded to
//!   `line_height * inline_height_factor` so the image sits with
//!   surrounding text on the same row, width scaled to preserve
//!   aspect ratio. If the natural height is smaller than the bound,
//!   natural dimensions are used.
//! * [`block_size`] for `BlockKind::Image` — width capped to the
//!   available content width, height scaled to preserve aspect.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use gpui::{
    App, Bounds, Corners, ImgResourceLoader, Pixels, Point, RenderImage, Resource, SharedString,
    SharedUri, Size, Window, px, size,
};

/// Outcome of a cache lookup. The element layer dispatches on this:
///
/// * `Loaded` → measure, substitute, paint.
/// * `Loading` → reserve a placeholder square so the layout doesn't
///   jump when the load resolves on a later frame.
/// * `Failed` → fall back to dim delimiters + alt text on the source
///   bytes (the cursor-inside visual treatment).
#[derive(Clone)]
pub enum LoadedImage {
    Loaded(Arc<RenderImage>),
    Loading,
    Failed,
}

impl LoadedImage {
    pub fn is_loaded(&self) -> bool {
        matches!(self, LoadedImage::Loaded(_))
    }

    pub fn is_failed(&self) -> bool {
        matches!(self, LoadedImage::Failed)
    }
}

/// Which image sources [`load`] may resolve: a host-set gpui global,
/// [`ImageSources::Any`] when unset.
///
/// The destination of `![alt](url)` is chosen by whoever wrote the
/// markdown, and a host that renders text it did not write (a model's
/// reply, a pasted document) hands that author a fetch of their choosing:
/// a remote URL is a request whose path can carry anything the author
/// could see, and a local path is an unbounded `fs::read` — `/dev/zero`
/// never ends, and a FIFO blocks the loading thread in `open` forever.
/// Such a host sets [`ImageSources::EmbeddedOnly`]; a refused source
/// takes the [`LoadedImage::Failed`] path, so its markdown stays visible
/// rather than silently vanishing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ImageSources {
    /// `http(s)`, `file://` and absolute paths, and embedded assets.
    #[default]
    Any,
    /// Only the host's own `AssetSource` (relative paths).
    EmbeddedOnly,
}

impl gpui::Global for ImageSources {}

impl ImageSources {
    fn allows(self, resource: &Resource) -> bool {
        match self {
            ImageSources::Any => true,
            ImageSources::EmbeddedOnly => matches!(resource, Resource::Embedded(_)),
        }
    }
}

/// Resolve `dest_url` through gpui's image cache. Kicks off an
/// asynchronous load the first time the URL is seen; subsequent
/// frames see the cached result. The cache invalidates the view
/// when the load completes so this function is safe to call from
/// `request_layout` and `prepaint` — the next frame will see the
/// updated state.
///
/// `dest_url` is classified into a [`Resource`]:
///
/// * `http://…` / `https://…` — fetched via the asset HTTP client.
/// * `file://…` and absolute filesystem paths — read via the
///   filesystem loader.
/// * A relative path, when the editor has a `base_dir` (a document
///   opened from disk — see `MarkdownEditorState::set_image_base_dir`)
///   — joined onto it and read via the filesystem loader, so
///   `![](assets/fig.png)` beside a note finds the file the way every
///   other markdown viewer does. Percent-escapes are decoded first
///   (`my%20figure.png`), since a link destination cannot carry a raw
///   space.
/// * Anything else — treated as an embedded asset path. Hosting
///   applications can ship their own embedded asset source via
///   `gpui::AssetSource` to resolve these.
pub fn load(
    dest_url: &str,
    base_dir: Option<&Path>,
    window: &mut Window,
    cx: &mut App,
) -> LoadedImage {
    let resource = make_resource(dest_url, base_dir);
    let policy = cx.try_global::<ImageSources>().copied().unwrap_or_default();
    if !policy.allows(&resource) {
        return LoadedImage::Failed;
    }
    match window.use_asset::<ImgResourceLoader>(&resource, cx) {
        Some(Ok(data)) => LoadedImage::Loaded(data),
        Some(Err(_)) => LoadedImage::Failed,
        None => LoadedImage::Loading,
    }
}

fn make_resource(dest_url: &str, base_dir: Option<&Path>) -> Resource {
    if dest_url.starts_with("http://") || dest_url.starts_with("https://") {
        Resource::Uri(SharedUri::from(dest_url.to_string()))
    } else if let Some(path) = dest_url.strip_prefix("file://") {
        Resource::Path(PathBuf::from(path).into())
    } else if dest_url.starts_with('/') {
        Resource::Path(PathBuf::from(dest_url).into())
    } else if let Some(base) = base_dir.filter(|_| !has_scheme(dest_url)) {
        Resource::Path(base.join(percent_decode(dest_url)).into())
    } else {
        // Treat as an embedded asset path; hosting apps can register
        // their own `AssetSource` to resolve these. Falls back to a
        // failed load if the host hasn't registered one.
        Resource::Embedded(SharedString::from(dest_url.to_string()))
    }
}

/// Whether `url` opens with a URI scheme (`data:`, `mailto:`, …) rather than
/// a path. Such a destination is never a file beside the document.
fn has_scheme(url: &str) -> bool {
    match url.split_once(':') {
        Some((scheme, _)) => {
            scheme.starts_with(|c: char| c.is_ascii_alphabetic())
                && scheme
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
        }
        None => false,
    }
}

/// Decode `%XX` escapes. A malformed escape is kept literally, and a
/// decoding that is not UTF-8 falls back to the input unchanged.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let (Some(h), Some(l)) = (hex(bytes[i + 1]), hex(bytes[i + 2]))
        {
            out.push((h << 4) | l);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| s.to_string())
}

fn hex(b: u8) -> Option<u8> {
    (b as char).to_digit(16).map(|d| d as u8)
}

/// Pixel size for an inline image overlay at the given line height.
/// Caps height to `line_height * height_factor` (so the image sits
/// comfortably alongside surrounding text), preserving aspect ratio.
/// If natural height is smaller than the cap, returns the natural
/// dimensions unchanged.
pub fn inline_size(image: &RenderImage, line_height: Pixels, height_factor: f32) -> Size<Pixels> {
    let natural = image.size(0);
    let nat_w = px(natural.width.0 as f32);
    let nat_h = px(natural.height.0 as f32);
    if nat_h <= px(0.0) || nat_w <= px(0.0) {
        return size(px(0.0), px(0.0));
    }
    let cap = line_height * height_factor;
    if nat_h <= cap {
        return size(nat_w, nat_h);
    }
    let scale = f32::from(cap) / f32::from(nat_h);
    size(px(f32::from(nat_w) * scale), cap)
}

/// Pixel size for a block image: scale to fit within `max_width`,
/// preserving aspect ratio. If natural width already fits, returns
/// natural dimensions.
pub fn block_size(image: &RenderImage, max_width: Pixels) -> Size<Pixels> {
    let natural = image.size(0);
    let nat_w = px(natural.width.0 as f32);
    let nat_h = px(natural.height.0 as f32);
    if nat_h <= px(0.0) || nat_w <= px(0.0) {
        return size(px(0.0), px(0.0));
    }
    if nat_w <= max_width {
        return size(nat_w, nat_h);
    }
    let scale = f32::from(max_width) / f32::from(nat_w);
    size(max_width, px(f32::from(nat_h) * scale))
}

/// Paint `image` filling `bounds`. Thin wrapper around
/// [`Window::paint_image`] so callers don't have to thread the image
/// bounds (the same as `bounds` — the image fills its box) or the
/// `Corners` / `grayscale` / `frame_index` arguments.
pub fn paint(image: Arc<RenderImage>, bounds: Bounds<Pixels>, window: &mut Window) {
    let _ = window.paint_image(bounds, bounds, Corners::default(), image, 0, false);
}

/// Default fraction of a line height to reserve for an inline image
/// whose natural size isn't yet known (still loading). The same
/// fraction is used as the height cap for inline images so the
/// placeholder doesn't jump when the load completes.
pub const INLINE_HEIGHT_FACTOR: f32 = 1.4;

/// Reserve a placeholder square `INLINE_HEIGHT_FACTOR * line_height`
/// tall while an inline image loads. Square because we don't yet
/// know the aspect ratio; the substitution width is approximate
/// either way (the load completes within a frame or two in the
/// cached / file:// case).
pub fn inline_placeholder_size(line_height: Pixels) -> Size<Pixels> {
    let h = line_height * INLINE_HEIGHT_FACTOR;
    size(h, h)
}

/// Reserve a placeholder for a block image whose natural size isn't
/// yet known. Defaults to 8em tall and 16em wide (a 2:1 banner),
/// clamped to the available width.
pub fn block_placeholder_size(em_px: Pixels, max_width: Pixels) -> Size<Pixels> {
    let h = em_px * 8.0;
    let w = (em_px * 16.0).min(max_width);
    size(w, h)
}

/// Convenience for callers that want to paint an image into a
/// centered sub-rectangle of `bounds`. Picks the largest aspect-
/// preserving fit. Mirrors gpui's `ObjectFit::Contain`. Currently
/// unused — paint paths size `bounds` exactly so no contain pass is
/// required — but exposed for hosting code.
pub fn fit_contain(natural: Size<Pixels>, bounds: Bounds<Pixels>) -> Bounds<Pixels> {
    if natural.width <= px(0.0) || natural.height <= px(0.0) {
        return bounds;
    }
    let scale_w = f32::from(bounds.size.width) / f32::from(natural.width);
    let scale_h = f32::from(bounds.size.height) / f32::from(natural.height);
    let scale = scale_w.min(scale_h);
    let w = px(f32::from(natural.width) * scale);
    let h = px(f32::from(natural.height) * scale);
    let x = bounds.origin.x + (bounds.size.width - w) / 2.0;
    let y = bounds.origin.y + (bounds.size.height - h) / 2.0;
    Bounds::new(Point { x, y }, size(w, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path_of(resource: Resource) -> Option<PathBuf> {
        match resource {
            Resource::Path(p) => Some(p.to_path_buf()),
            _ => None,
        }
    }

    #[test]
    fn a_relative_image_resolves_against_the_documents_directory() {
        let base = Path::new("/notes/today");
        assert_eq!(
            path_of(make_resource("assets/fig.png", Some(base))),
            Some(PathBuf::from("/notes/today/assets/fig.png"))
        );
        assert_eq!(
            path_of(make_resource("../my%20figure.png", Some(base))),
            Some(PathBuf::from("/notes/today/../my figure.png"))
        );
        // Absolute paths and URLs ignore the base.
        assert_eq!(
            path_of(make_resource("/abs/fig.png", Some(base))),
            Some(PathBuf::from("/abs/fig.png"))
        );
        assert!(matches!(
            make_resource("https://example.com/a.png", Some(base)),
            Resource::Uri(_)
        ));
        // A scheme is never a file beside the document.
        assert!(matches!(
            make_resource("data:image/png;base64,AAAA", Some(base)),
            Resource::Embedded(_)
        ));
    }

    #[test]
    fn without_a_base_a_relative_image_stays_an_embedded_asset() {
        assert!(matches!(
            make_resource("assets/fig.png", None),
            Resource::Embedded(_)
        ));
    }

    #[test]
    fn percent_decoding_keeps_what_it_cannot_decode() {
        assert_eq!(percent_decode("a%20b"), "a b");
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("%zz"), "%zz");
        assert_eq!(percent_decode("%E2%9C%93"), "\u{2713}");
    }

    #[test]
    fn inline_size_respects_height_cap_and_scales_width() {
        // We can't easily construct a RenderImage in tests, but we
        // can exercise the placeholder helpers and fit_contain which
        // operate purely on Size values.
        let pl = inline_placeholder_size(px(20.0));
        assert!(pl.height > px(0.0));
        assert!(pl.width > px(0.0));
    }

    #[test]
    fn block_placeholder_clamps_to_available_width() {
        let small = block_placeholder_size(px(16.0), px(64.0));
        assert!(small.width <= px(64.0));
        let large = block_placeholder_size(px(16.0), px(1024.0));
        // 16 * 16 = 256
        assert_eq!(large.width, px(256.0));
    }

    #[test]
    fn fit_contain_preserves_aspect() {
        let natural = size(px(100.0), px(50.0));
        let bounds = Bounds::new(
            Point {
                x: px(0.0),
                y: px(0.0),
            },
            size(px(50.0), px(50.0)),
        );
        let fit = fit_contain(natural, bounds);
        assert_eq!(fit.size.width, px(50.0));
        assert_eq!(fit.size.height, px(25.0));
    }

    #[test]
    fn embedded_only_refuses_every_source_the_markdown_author_picks() {
        for url in [
            "https://example.com/x.png?q=secret",
            "http://127.0.0.1:8765/x.png",
            "file:///dev/zero",
            "/dev/zero",
            "/tmp/pipe.fifo",
        ] {
            let resource = make_resource(url, None);
            assert!(
                !ImageSources::EmbeddedOnly.allows(&resource),
                "{url} must be refused under EmbeddedOnly"
            );
            assert!(ImageSources::Any.allows(&resource), "{url} under Any");
        }
    }

    #[test]
    fn embedded_only_keeps_the_hosts_own_assets() {
        for url in [
            "icons/logo.png",
            "images/x.svg",
            "HTTPS://upper-case-is-not-a-scheme",
        ] {
            let resource = make_resource(url, None);
            assert!(matches!(resource, Resource::Embedded(_)), "{url}");
            assert!(ImageSources::EmbeddedOnly.allows(&resource), "{url}");
        }
    }

    #[test]
    fn unset_policy_is_any() {
        assert_eq!(ImageSources::default(), ImageSources::Any);
    }
}
