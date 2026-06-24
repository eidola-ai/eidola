use std::collections::HashMap;

use eidola_gui::theme;
use gpui::*;
use gpui_component::{ActiveTheme, InteractiveElementExt, Root, h_flex, v_flex};
use gpui_markdown_editor::{EditorState, MarkdownEditor, MarkdownEditorState, MarkdownStyle};

// ---------------------------------------------------------------------------
// Layout constants. Kept inline (and a little duplicated from the real chat
// view) on purpose — this is a short-lived visual experiment, so we favour a
// self-contained file over reaching into the app's chat module.
// ---------------------------------------------------------------------------

/// Height reserved at the top of the window for the (transparent) titlebar.
/// macOS extends the content view under the traffic-light buttons, so we leave
/// this much room and treat the band as a draggable title-bar surface.
const TITLE_BAR_RESERVE: Pixels = px(36.);

/// Prose typography for the user-/AI-authored narrative content. The body is
/// Newsreader (a serif) at a book-like size and leading — deliberately distinct
/// from the system UI font the theme uses for components/chrome.
const PROSE_FONT_SIZE: Pixels = px(17.);
const PROSE_LINE_HEIGHT: f32 = 1.65;

/// The byline gutter (right-aligned author + time) and the centered reading
/// column it sits beside.
const GUTTER_WIDTH: Pixels = px(120.);
const GUTTER_GAP: Pixels = px(28.);
const BODY_MAX_WIDTH: Pixels = px(600.);
const SIDE_PAD: Pixels = px(40.);

/// Vertical breathing room around each post, plus the faint full-bleed band
/// that separates one depth level (one row of the tree) from the next.
const POST_PAD_Y: Pixels = px(40.);
const BAND_HEIGHT: Pixels = px(48.);

/// One post in the conversation tree: who wrote it, when, its markdown content,
/// and its replies. The space is a tree of these (replies only); the UI follows
/// the tree structure (see [`SpaceView`]).
struct Node {
    /// Stable id, unique across the tree — also the element id for the post's
    /// page (keeps the per-post markdown editors from colliding) and the key
    /// for its editor state in [`SpaceView::bodies`].
    id: &'static str,
    author: &'static str,
    created_at: &'static str,
    /// Formatted as markdown.
    content: &'static str,
    /// Replies, ordered left-to-right by creation time (earliest first).
    children: Vec<Node>,
}

/// The space view renders a conversation *tree* as **recursively nested**
/// scrollers. Each node renders its post, then (if it has replies) a separator
/// band and a horizontal scroller whose pages are its children — and each of
/// those pages is the child's *entire subtree*, rendered the same way. So
/// scrolling a node's children scroller moves between whole branches (every
/// descendant travels with its branch), and the nesting *is* the tree.
///
/// A horizontal scroll is claimed by the innermost branch scroller under the
/// cursor (it stops propagation), so scrolling over a deep post navigates that
/// post's level while scrolling over a shallower region navigates the level
/// that encloses it. Vertical scroll bubbles to the one outer page scroller.
///
/// Because a branch scroller is as tall as its tallest child subtree, a shorter
/// sibling leaves empty space *below* it — the root view is taller than any one
/// branch needs, but that slack is always at the bottom.
pub struct SpaceView {
    /// The conversation tree (a single root for this experiment).
    root: Node,
    /// One read-only markdown-editor state per node, keyed by node id.
    bodies: HashMap<&'static str, Entity<MarkdownEditorState>>,
    /// One horizontal `ScrollHandle` per node that has children, keyed by node
    /// id. Only used to read the scroll position back (to highlight the active
    /// page-indicator dot); the nesting itself carries the structure.
    scrolls: HashMap<&'static str, ScrollHandle>,
    /// Set on mouse-down in the title-bar band, consumed on the first
    /// mouse-move to begin a native window drag (mirrors gpui-component's
    /// `TitleBar`: `start_window_move` wants a drag event, not the down).
    should_move_window: bool,
}

impl SpaceView {
    /// Build the view: seed the conversation tree and a read-only
    /// markdown-editor state per node (each needs `window`/`cx`, so this runs
    /// inside `cx.new`).
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let root = sample_tree();

        let mut bodies = HashMap::new();
        let mut scrolls = HashMap::new();
        build_state(&root, &mut bodies, &mut scrolls, window, cx);

        Self {
            root,
            bodies,
            scrolls,
            should_move_window: false,
        }
    }

    /// Which child page a node's scroller is resting on, derived from its scroll
    /// offset. Each page is one viewport wide, so the nearest is
    /// `round(scrolled / page_width)`. Cosmetic — it only highlights the active
    /// page-indicator dot.
    fn active_child_index(&self, node_id: &str, page_width: Pixels, count: usize) -> usize {
        if count <= 1 || page_width <= px(0.) {
            return 0;
        }
        let Some(handle) = self.scrolls.get(node_id) else {
            return 0;
        };
        // Pages are separated by a vertical band, so the stride between page
        // origins is the page width plus that separator.
        let stride = (page_width + BAND_HEIGHT).as_f32();
        let scrolled = (-handle.offset().x).as_f32();
        let idx = (scrolled / stride).round() as i64;
        idx.clamp(0, count as i64 - 1) as usize
    }

    /// Render a node's whole subtree: its post, then (if it has replies) a
    /// separator band and a horizontal scroller whose pages are each child's
    /// *entire subtree* (recursively). The recursion builds the nesting that
    /// carries the tree structure.
    fn render_node(&self, node: &Node, page_width: Pixels, cx: &Context<Self>) -> Div {
        let theme = cx.theme();
        // Definite pixel widths (not `w_full`) throughout the subtree: a page
        // lives inside an `overflow_x_scroll`, where percentage widths resolve
        // against the scroller's (effectively unbounded) content size rather
        // than the page, collapsing everything to content width.
        let mut column = v_flex()
            .w(page_width)
            .child(self.render_post(node, page_width, cx));

        if node.children.is_empty() {
            return column;
        }

        let count = node.children.len();
        let active = self.active_child_index(node.id, page_width, count);
        column = column.child(render_band(page_width, count, active, &theme));

        // The branch scroller: each page is one child's full subtree. The
        // innermost scroller under the cursor claims a horizontal scroll (stops
        // propagation) so it doesn't also move the scrollers above it; vertical
        // deltas fall through to the outer page scroller.
        // `items_stretch` so the vertical branch separators (and short branch
        // pages) fill the scroller's full height — the slack of a shorter branch
        // ends up at its bottom.
        let mut strip = h_flex()
            .id(SharedString::from(format!("{}-children", node.id)))
            .w(page_width)
            .items_stretch()
            .overflow_x_scroll()
            .on_scroll_wheel(cx.listener(|_, ev: &ScrollWheelEvent, window, cx| {
                let delta = ev.delta.pixel_delta(window.line_height());
                if delta.x.as_f32().abs() > delta.y.as_f32().abs() {
                    cx.stop_propagation();
                }
            }));
        if let Some(handle) = self.scrolls.get(node.id) {
            strip = strip.track_scroll(handle);
        }
        for (i, child) in node.children.iter().enumerate() {
            // A vertical separator between branches — same thickness and ground
            // as the horizontal band, so a scroll across the seam reads as a
            // real boundary between two branches.
            if i > 0 {
                strip = strip.child(div().w(BAND_HEIGHT).flex_none().bg(theme.muted));
            }
            // The page wrapper carries the child id so the per-post markdown
            // editors (which all share the element id "markdown-editor") get
            // distinct global ids across branches.
            strip = strip.child(
                div()
                    .id(SharedString::from(child.id))
                    .w(page_width)
                    .flex_none()
                    .child(self.render_node(child, page_width, cx)),
            );
        }
        column.child(strip)
    }

    /// One post: the right-aligned byline gutter (system font) beside the
    /// centered reading column (Newsreader prose, read-only markdown).
    fn render_post(&self, node: &Node, page_width: Pixels, cx: &Context<Self>) -> impl IntoElement {
        let theme = cx.theme();

        let body_width = (page_width - GUTTER_WIDTH - GUTTER_GAP - SIDE_PAD * 2.)
            .min(BODY_MAX_WIDTH)
            .max(px(240.));

        // Byline — UI/chrome voice (system font): bold name over a muted time.
        let byline = v_flex()
            .w(GUTTER_WIDTH)
            .flex_none()
            .items_end()
            .child(
                div()
                    .text_sm()
                    .font_weight(FontWeight::BOLD)
                    .text_color(theme.foreground)
                    .child(SharedString::from(node.author)),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(theme.muted_foreground)
                    .child(SharedString::from(node.created_at)),
            );

        // Body — narrative voice (Newsreader prose), read-only markdown. A
        // definite width makes the markdown's height-for-width measurement
        // correct.
        let body = div().w(body_width).child(
            MarkdownEditor::new(&self.bodies[node.id])
                .style(prose_style(cx))
                .disabled(true),
        );

        h_flex()
            .w(page_width)
            .py(POST_PAD_Y)
            .justify_center()
            .items_start()
            .gap(GUTTER_GAP)
            .child(byline)
            .child(body)
    }

    /// A draggable band across the top of the window standing in for the
    /// (now transparent) titlebar. On macOS `WindowControlArea::Drag` is a
    /// no-op, so dragging is wired explicitly: arm on mouse-down, then call
    /// `window.start_window_move()` on the first move while armed.
    fn render_title_bar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .id("title-bar")
            .absolute()
            .top_0()
            .left_0()
            .right_0()
            .h(TITLE_BAR_RESERVE)
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _, _, _| this.should_move_window = true),
            )
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, _, _, _| this.should_move_window = false),
            )
            .on_mouse_move(cx.listener(|this, _, window, _| {
                if this.should_move_window {
                    this.should_move_window = false;
                    window.start_window_move();
                }
            }))
            .on_double_click(|_, window, _| window.titlebar_double_click())
    }
}

impl Render for SpaceView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let page_width = window.viewport_size().width;

        // The whole tree is one recursively-nested element rooted at A.
        let tree = self.render_node(&self.root, page_width, cx);

        div()
            .relative()
            .size_full()
            .bg(theme.background)
            // Components/chrome render in the system UI font (the theme leaves
            // `font_family` unset); only prose opts into Newsreader.
            .font_family(theme.font_family.clone())
            .text_color(theme.foreground)
            .child(self.render_title_bar(cx))
            .child(
                div()
                    .id("scroll")
                    .size_full()
                    .overflow_y_scroll()
                    .child(v_flex().w_full().pt(TITLE_BAR_RESERVE).child(tree)),
            )
    }
}

/// The faint full-bleed separator band between a post and its replies. When the
/// post has more than one reply, the band carries a row of page-indicator dots
/// (the active branch highlighted); a single reply shows a plain band, so a
/// non-branching conversation reads like a sequential list.
fn render_band(
    page_width: Pixels,
    count: usize,
    active: usize,
    theme: &gpui_component::Theme,
) -> Div {
    let mut band = h_flex()
        .w(page_width)
        .h(BAND_HEIGHT)
        .bg(theme.muted)
        .items_center()
        .justify_center();
    if count >= 2 {
        band = band.child(h_flex().gap_2().children((0..count).map(|i| {
            div().size(px(5.)).rounded_full().bg(if i == active {
                theme.muted_foreground
            } else {
                theme.border
            })
        })));
    }
    band
}

/// `MarkdownStyle` for prose bodies: Newsreader at a book size/leading with a
/// gentle heading ramp. `from_theme` seeds the system font, so we override the
/// family back to Newsreader for narrative content.
fn prose_style(cx: &App) -> MarkdownStyle {
    let mut style = MarkdownStyle::from_theme(cx)
        .font_size(PROSE_FONT_SIZE)
        .line_height(rems(PROSE_LINE_HEIGHT))
        .paragraph_gap(rems(1.5))
        .heading_base_font_size(PROSE_FONT_SIZE)
        .heading_font_size(|level, base| match level {
            1 => base * 1.5,
            2 => base * 1.25,
            3 => base * 1.125,
            _ => base,
        });
    style.font_family = theme::FONT_FAMILY.into();
    style
}

/// Walk the tree once, creating a read-only markdown-editor state for every
/// node and a horizontal scroll handle for every node that has replies.
fn build_state(
    node: &Node,
    bodies: &mut HashMap<&'static str, Entity<MarkdownEditorState>>,
    scrolls: &mut HashMap<&'static str, ScrollHandle>,
    window: &mut Window,
    cx: &mut Context<SpaceView>,
) {
    let markdown = node.content.to_string();
    let state = cx.new(|cx| {
        MarkdownEditorState::with_state(
            EditorState {
                markdown,
                ..Default::default()
            },
            window,
            cx,
        )
    });
    bodies.insert(node.id, state);
    if !node.children.is_empty() {
        scrolls.insert(node.id, ScrollHandle::new());
    }
    for child in &node.children {
        build_state(child, bodies, scrolls, window, cx);
    }
}

/// The seed tree for the experiment:
///
/// ```text
/// 0        A
///         / \
/// 1      B   C
///       /|
/// 2    D E
///        |
/// 3      F
/// ```
fn sample_tree() -> Node {
    Node {
        id: "A",
        author: "Mara Vance",
        created_at: "10:03 AM",
        content: "I've started treating every note I write as the first room of a house I might never finish building. You walk in, set down one true sentence, and leave the door open behind you.\n\n\
            For years I wrote into documents – clean, walled, finished-feeling things. A document *wants* to be done. It pulls the last paragraph toward it like a tide. But the thoughts I care about most aren't done; they're held in a kind of suspension, waiting for someone – a friend, a stranger, some patient machine – to disturb them. A document has no room for that disturbance. It has margins, but no *space*.\n\n\
            So this is the small wager of writing here instead: that the note stays exactly as I wrote it, at the size I wrote it, and the conversation grows around it rather than burying it. The first post is load-bearing. Everything else leans on it.\n\n\
            *A reply should feel less like a comment and more like someone pulling a chair up to the same table.*\n\n\
            What I don't yet know is how deep that table can get before it stops feeling like one table. At some point a thread becomes a forest, and you lose the path back to the clearing where you started. Maybe that's fine. Maybe the clearing should always be one gesture away.",
        children: vec![
            Node {
                id: "B",
                author: "Kimi K2",
                created_at: "10:04 AM",
                content: "There's a quiet radicalism in refusing the document's pull toward done. Most tools treat a note as a draft of something else, a means to a finished end. You're treating it as a place worth standing in.\n\n\
                    The risk you name – that a place can sprawl – is real. But a sprawling place is still a place. A failed essay is a failure; a rambling house is just a house with too many rooms, and you can always close a door. The structure forgives more than the document does.",
                children: vec![
                    Node {
                        id: "D",
                        author: "Mara Vance",
                        created_at: "10:06 AM",
                        content: "Sprawl is exactly the fear, though. A failed essay at least has an ending – you know when you've lost. A place can just keep adding rooms until no one remembers where the front door was.",
                        children: vec![],
                    },
                    Node {
                        id: "E",
                        author: "Mara Vance",
                        created_at: "10:07 AM",
                        content: "Maybe the front door is the whole point. You don't memorize a house. You keep returning to the one room that matters, and the rest stays *available* without ever being *demanded* of you.",
                        children: vec![Node {
                            id: "F",
                            author: "Kimi K2",
                            created_at: "10:08 AM",
                            content: "Right – availability without obligation. The forest is fine as long as there's always a path marked back to the clearing. The branching only hurts when it erases the trunk.",
                            children: vec![],
                        }],
                    },
                ],
            },
            Node {
                id: "C",
                author: "Kimi K2",
                created_at: "10:04 AM",
                content: "The load-bearing-post idea is the whole thing for me. In chat apps the first message is the most disposable – it scrolls into the dark within the hour. Here you're proposing the opposite: the origin stays lit, and everything is measured against it.\n\n\
                    Does that put a lot of pressure on the first sentence, though?",
                children: vec![],
            },
        ],
    }
}

fn main() {
    let app = gpui_platform::application().with_assets(gpui_component_assets::Assets);

    app.run(move |cx| {
        // This must be called before using any GPUI Component features.
        gpui_component::init(cx);

        // Install the theme.
        theme::install(cx);

        // Initialize the markdown editor key bindings.
        gpui_markdown_editor::init(cx);

        // Open an initial window. `cx` here is `&mut App`, so the window opens
        // synchronously (no `cx.spawn`, which would hand back an `AsyncApp`
        // that `WindowBounds::centered` can't take).
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::centered(size(px(900.), px(680.)), cx)),
                titlebar: Some(TitlebarOptions {
                    appears_transparent: true,
                    ..Default::default()
                }),
                ..Default::default()
            },
            |window, cx| {
                theme::observe_window_appearance(window);

                // Root the view in the window.
                let view = cx.new(|cx| SpaceView::new(window, cx));
                cx.new(|cx| Root::new(view, window, cx))
            },
        )
        .expect("Failed to open window.");
    });
}
