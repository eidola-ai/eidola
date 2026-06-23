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
/// that separates one post from the next.
const POST_PAD_Y: Pixels = px(40.);
const BAND_HEIGHT: Pixels = px(48.);

/// One post in the space: who wrote it, when, and its markdown content.
pub struct SpaceItem {
    author: String,
    created_at: String,

    /// Formatted as markdown.
    content: String,
}

pub struct SpaceView {
    items: Vec<SpaceItem>,
    /// One markdown-editor state per item (index-aligned with `items`), each
    /// rendered read-only (`disabled`) so it displays prose without accepting
    /// input.
    bodies: Vec<Entity<MarkdownEditorState>>,
    /// Set on mouse-down in the title-bar band, consumed on the first
    /// mouse-move to begin a native window drag (mirrors gpui-component's
    /// `TitleBar`: `start_window_move` wants a drag event, not the down).
    should_move_window: bool,
}

impl SpaceView {
    /// Build the view: seed the post list and a read-only markdown-editor state
    /// per post (each needs `window`/`cx`, so this runs inside `cx.new`).
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let items = sample_items();
        let bodies = items
            .iter()
            .map(|item| {
                let markdown = item.content.clone();
                cx.new(|cx| {
                    MarkdownEditorState::with_state(
                        EditorState {
                            markdown,
                            ..Default::default()
                        },
                        window,
                        cx,
                    )
                })
            })
            .collect();
        Self {
            items,
            bodies,
            should_move_window: false,
        }
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

    /// One post: the right-aligned byline gutter (system font) beside the
    /// centered reading column (Newsreader prose, rendered through a read-only
    /// `MarkdownEditor`). The `[gutter | body]` pair is centered as a unit.
    fn render_post(
        &self,
        index: usize,
        body_width: Pixels,
        cx: &Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme();
        let item = &self.items[index];

        // Byline — UI/chrome voice (system font): bold name over a muted time.
        let byline = v_flex()
            .w(GUTTER_WIDTH)
            .flex_none()
            .items_end()
            .child(
                div()
                    .text_sm()
                    .pt_4()
                    .font_weight(FontWeight::BOLD)
                    .text_color(theme.foreground)
                    .text_align(TextAlign::Right)
                    .child(SharedString::from(item.author.clone())),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(theme.muted_foreground)
                    .child(SharedString::from(item.created_at.clone())),
            );

        // Body — narrative voice (Newsreader prose), read-only markdown. A
        // definite width makes the markdown's height-for-width measurement
        // correct.
        let body = div().w(body_width).child(
            MarkdownEditor::new(&self.bodies[index])
                .style(prose_style(cx))
                .disabled(true),
        );

        h_flex()
            .w_full()
            .py(POST_PAD_Y)
            .justify_center()
            .items_start()
            .gap(GUTTER_GAP)
            .child(byline)
            .child(body)
    }
}

impl Render for SpaceView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();

        // The reading column shrinks to fit narrow windows but is capped at the
        // prose measure so long-form text keeps a comfortable ~65-char line.
        let viewport_w = window.viewport_size().width;
        let body_width = (viewport_w - GUTTER_WIDTH - GUTTER_GAP - SIDE_PAD * 2.)
            .min(BODY_MAX_WIDTH)
            .max(px(240.));

        // Posts, interleaved with a faint full-bleed band between them.
        let mut content = v_flex().w_full().pt(TITLE_BAR_RESERVE);
        for index in 0..self.items.len() {
            if index > 0 {
                content = content.child(div().w_full().h(BAND_HEIGHT).bg(theme.muted));
            }
            content = content.child(self.render_post(index, body_width, cx));
        }

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
                    .child(content),
            )
    }
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

/// The seed content for the experiment — a small thread of posts.
fn sample_items() -> Vec<SpaceItem> {
    vec![
        SpaceItem {
            author: "Mike Marcacci".into(),
            created_at: "10:03 AM".into(),
            content: "I've started treating every note I write as the first room of a house I might never finish building. You walk in, set down one true sentence, and leave the door open behind you.\n\n\
                For years I wrote into documents – clean, walled, finished-feeling things. A document *wants* to be done. It pulls the last paragraph toward it like a tide. But the thoughts I care about most aren't done; they're held in a kind of suspension, waiting for someone – a friend, a stranger, some patient machine – to disturb them. A document has no room for that disturbance. It has margins, but no *space*.\n\n\
                So this is the small wager of writing here instead: that the note stays exactly as I wrote it, at the size I wrote it, and the conversation grows around it rather than burying it. The first post is load-bearing. Everything else leans on it.\n\n\
                *A reply should feel less like a comment and more like someone pulling a chair up to the same table.*\n\n\
                What I don't yet know is how deep that table can get before it stops feeling like one table. At some point a thread becomes a forest, and you lose the path back to the clearing where you started. Maybe that's fine. Maybe the clearing should always be one gesture away."
                .into(),
        },
        SpaceItem {
            author: "Kimi K2".into(),
            created_at: "10:04 AM".into(),
            content: "The load-bearing-post idea is the whole thing for me. In chat apps the first message is the most disposable – it scrolls into the dark within the hour. Here you're proposing the opposite: the origin stays lit, and everything is measured against it."
                .into(),
        },
    ]
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
