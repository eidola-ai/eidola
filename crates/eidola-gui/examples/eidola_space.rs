use eidola_app_core::PostNode;
use eidola_gui::theme;
use gpui::*;
use gpui_component::{button::*, *};
use gpui_markdown_editor::{EditorState, MarkdownEditor, MarkdownEditorState, MarkdownStyle};

// #[path = "../tests/visual/fixtures.rs"]
// mod fixtures;
// use fixtures::kitchen_sink_posts;

/// Height reserved at the top of the window for the (transparent) titlebar.
/// macOS extends the content view under the traffic-light buttons, so we leave
/// this much room and treat the band as a draggable title-bar surface.
const TITLE_BAR_RESERVE: gpui::Pixels = px(36.);

pub struct SpaceView {
    editor_state: Entity<MarkdownEditorState>,
    /// Set on mouse-down in the title-bar band, consumed on the first
    /// mouse-move to begin a native window drag (mirrors gpui-component's
    /// `TitleBar`: `start_window_move` wants a drag event, not the down).
    should_move_window: bool,
}

impl SpaceView {
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
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();

        div()
            .relative()
            .border_5()
            .child(self.render_title_bar(cx))
            .child(div().h(TITLE_BAR_RESERVE).w_full())
            .child(div().border_2().child("Hello!"))
            .child(Button::new("submit").ghost().label("Submit"))
            // .probe("chat/edit-editor", gpui::Role::TextInput, "Edit post")
            .child(
                MarkdownEditor::new(&self.editor_state)
                    .style(
                        MarkdownStyle::from_theme(cx)
                            // .font_size(PROSE_FONT_SIZE)
                            // .line_height(rems(PROSE_LINE_HEIGHT))
                            // .paragraph_gap(rems(1.5))
                            // .heading_base_font_size(PROSE_FONT_SIZE)
                            .heading_font_size(|level, base| match level {
                                1 => base * 1.5,
                                2 => base * 1.25,
                                3 => base * 1.125,
                                _ => base,
                            })
                            // Inline code shares its shaped line with Newsreader body text, and
                            // gpui can't size a single run independently (`TextRun` has no font
                            // size), so the ~0.9× inline-code size good typography wants is
                            // approximated by *family* instead: Courier New's x-height (0.423 em)
                            // matches Newsreader's (0.426 em) almost exactly, where the theme's
                            // Menlo (0.547 em) reads ~28% larger than the surrounding words.
                            // Fenced code blocks keep Menlo — they're whole lines of mono with no
                            // serif neighbors, so density and clarity win there.
                            .inline_code_font_family("Courier New"),
                    )
                    .disabled(false),
            )
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
                window_bounds: Some(WindowBounds::centered(size(px(800.), px(600.)), cx)),
                titlebar: Some(TitlebarOptions {
                    appears_transparent: true,
                    ..Default::default()
                }),
                ..Default::default()
            },
            |window, cx| {
                theme::observe_window_appearance(window);

                let editor_state = cx.new(|cx| {
                    MarkdownEditorState::with_state(
                        EditorState {
                            markdown: String::from("Hello, world!"),
                            ..Default::default()
                        },
                        window,
                        cx,
                    )
                });

                // Root the view in the window.
                let view = cx.new(|_| SpaceView {
                    editor_state,
                    should_move_window: false,
                });

                cx.new(|cx| Root::new(view, window, cx))
            },
        )
        .expect("Failed to open window.");
    });
}
