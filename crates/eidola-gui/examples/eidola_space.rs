use eidola_gui::theme;
use gpui::*;
use gpui_component::{button::*, *};
use gpui_markdown_editor::MarkdownEditorState;

pub struct SpaceView {
    editor: Entity<MarkdownEditorState>,
}

impl Render for SpaceView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();

        div()
            .border_5()
            .child(div().border_2().child("Hello!"))
            .child(Button::new("submit").ghost().label("Submit"))
    }
}

fn main() {
    let app = gpui_platform::application().with_assets(gpui_component_assets::Assets);

    app.run(move |cx| {
        // This must be called before using any GPUI Component features.
        gpui_component::init(cx);

        // Install the theme.
        theme::install(cx);

        // Open an initial window. `cx` here is `&mut App`, so the window opens
        // synchronously (no `cx.spawn`, which would hand back an `AsyncApp`
        // that `WindowBounds::centered` can't take).
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::centered(size(px(800.), px(600.)), cx)),
                titlebar: Some(TitlebarOptions {
                    appears_transparent: false,
                    ..Default::default()
                }),
                ..Default::default()
            },
            |window, cx| {
                theme::observe_window_appearance(window);

                let editor = cx.new(|cx| MarkdownEditorState::new(window, cx));

                // Root the view in the window.
                let view = cx.new(|_| SpaceView { editor });

                cx.new(|cx| Root::new(view, window, cx))
            },
        )
        .expect("Failed to open window.");
    });
}
