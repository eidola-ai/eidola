//! Read-only editor invariants — REGRESSION (content-dependent selection
//! failure): a `disabled` editor's selection/navigation events must never
//! rewrite the buffer.
//!
//! The editable pipeline canonicalizes the document on *every* event
//! (`enforce_invariants` — soft-break promotion, list/blockquote
//! normalization), including a bare `SetSelection` from a mouse click. Model
//! output routinely contains the non-canonical shapes (a markdown table's
//! `\n`-separated pipe rows, a heading tightly followed by its paragraph, a
//! paragraph directly followed by a blockquote), so a click on such a
//! read-only post rewrote the buffer away from the host's source of truth. In
//! the space view that divergence made `sync_bodies` re-seed the editor every
//! frame, resetting the selection to `Cursor(0)` — pointer selection on the
//! affected posts was impossible, while the autoscroll path (which re-extends
//! after each reset, later in the same render) appeared to work. The fix:
//! `dispatch` routes a disabled editor through `update::update_readonly`,
//! which applies selection events verbatim and refuses buffer mutations.

use gpui::{
    AppContext, Bounds, Entity, Modifiers, MouseButton, MouseDownEvent, MouseMoveEvent,
    MouseUpEvent, TestAppContext, VisualTestContext, WindowBounds, WindowOptions, point, px, size,
};
use gpui_component::Root;
use gpui_markdown_editor::update::{update, update_readonly};
use gpui_markdown_editor::{
    EditorEvent, EditorState, MarkdownEditor, MarkdownEditorState, Selection,
};

/// Markdown a model plausibly produces that the editable pipeline would
/// rewrite on the very first event: a table (lone `\n` between pipe rows), a
/// heading tightly followed by its paragraph, and a paragraph directly
/// followed by a blockquote.
const NON_CANONICAL: &str = "### heading\nbody text directly under the heading\n\n\
    **Note:**\n> a quoted line right after a paragraph\n\n\
    | left | right |\n| --- | --- |\n| cell one | cell two |";

/// Host view rendering the editor `disabled` (the read-only post surface).
struct ReadonlyHarness {
    state: Entity<MarkdownEditorState>,
}

impl gpui::Render for ReadonlyHarness {
    fn render(
        &mut self,
        _: &mut gpui::Window,
        _: &mut gpui::Context<Self>,
    ) -> impl gpui::IntoElement {
        MarkdownEditor::new(&self.state).disabled(true)
    }
}

fn open_readonly_editor(
    cx: &mut TestAppContext,
    markdown: &str,
) -> (VisualTestContext, Entity<MarkdownEditorState>) {
    let state = EditorState::with_markdown(markdown);
    let (handle, editor) = cx.update(|cx| {
        gpui_component::init(cx);
        let mut inner: Option<Entity<MarkdownEditorState>> = None;
        let bounds = Bounds {
            origin: point(px(0.), px(0.)),
            size: size(px(700.), px(600.)),
        };
        let window = cx
            .open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    ..Default::default()
                },
                |window, cx| {
                    let editor = cx.new(|cx| MarkdownEditorState::with_state(state, window, cx));
                    inner = Some(editor.clone());
                    let harness = cx.new(|_| ReadonlyHarness {
                        state: editor.clone(),
                    });
                    cx.new(|cx| Root::new(harness, window, cx))
                },
            )
            .expect("open window");
        (window.into(), inner.expect("editor built"))
    });
    let vcx = VisualTestContext::from_window(handle, cx);
    vcx.run_until_parked();
    (vcx, editor)
}

#[gpui::test]
fn readonly_mouse_selection_sticks_and_never_rewrites_the_buffer(cx: &mut TestAppContext) {
    let (mut vcx, editor) = open_readonly_editor(cx, NON_CANONICAL);

    // Aim the drag from the painted geometry (first laid-out line), so the
    // test doesn't hardcode font metrics.
    let (x, y, h) = editor
        .read_with(&vcx, |e, _| {
            e.debug_line_geometry()
                .first()
                .and_then(|(_, lines)| lines.first().copied())
        })
        .expect("a painted first line");
    let start = point(px(x + 30.0), px(y + h.min(24.0) / 2.0));
    let end = point(px(x + 180.0), px(y + h.min(24.0) / 2.0));

    vcx.simulate_event(MouseDownEvent {
        button: MouseButton::Left,
        position: start,
        modifiers: Modifiers::default(),
        click_count: 1,
        first_mouse: false,
    });
    vcx.simulate_event(MouseMoveEvent {
        position: end,
        pressed_button: Some(MouseButton::Left),
        modifiers: Modifiers::default(),
    });
    vcx.simulate_event(MouseUpEvent {
        button: MouseButton::Left,
        position: end,
        modifiers: Modifiers::default(),
        click_count: 1,
    });
    vcx.run_until_parked();

    editor.read_with(&vcx, |e, _| {
        assert_eq!(
            e.value(),
            NON_CANONICAL,
            "a read-only editor's click/drag must not rewrite the buffer"
        );
        let sel = e.selection();
        assert!(
            sel.upper_bound() > sel.lower_bound(),
            "the drag must leave a non-collapsed selection, got {sel:?}"
        );
    });
}

#[gpui::test]
fn readonly_editor_refuses_mutating_events(cx: &mut TestAppContext) {
    let (mut vcx, editor) = open_readonly_editor(cx, NON_CANONICAL);

    // Select something, then push mutating events straight through the
    // dispatch pipeline (belt-and-braces: the element doesn't register these
    // handlers when disabled, but the state must hold the line regardless).
    vcx.update(|_, cx| {
        editor.update(cx, |e, cx| {
            e.begin_selection_for_test(0, 1, false, cx);
            e.extend_selection_for_test(10, cx);
            e.apply_event_for_test(EditorEvent::DeleteBackward, cx);
            e.apply_event_for_test(EditorEvent::InsertText("nope".into()), cx);
        });
    });
    vcx.run_until_parked();

    editor.read_with(&vcx, |e, _| {
        assert_eq!(
            e.value(),
            NON_CANONICAL,
            "mutating events on a read-only editor must be refused"
        );
    });
}

#[test]
fn update_readonly_preserves_the_buffer_where_update_normalizes() {
    // The same SetSelection that makes the editable pipeline canonicalize a
    // soft break leaves the read-only buffer byte-identical.
    let raw = "| a | b |\n| --- | --- |\n| c | d |";
    let state = EditorState {
        markdown: raw.to_string(),
        selection: Selection::Cursor(0),
    };
    let editable = update(
        state.clone(),
        EditorEvent::SetSelection(Selection::Cursor(2)),
    );
    assert_ne!(
        editable.markdown, raw,
        "precondition: the editable pipeline normalizes this shape \
         (otherwise this regression test guards nothing)"
    );

    let readonly = update_readonly(state, EditorEvent::SetSelection(Selection::Cursor(2)));
    assert_eq!(readonly.markdown, raw);
    assert_eq!(readonly.selection, Selection::Cursor(2));
}

#[test]
fn update_readonly_refuses_document_mutations_wholesale() {
    let raw = "some plain text";
    let state = EditorState {
        markdown: raw.to_string(),
        selection: Selection::Range { anchor: 0, head: 4 },
    };
    let next = update_readonly(state, EditorEvent::DeleteBackward);
    assert_eq!(next.markdown, raw);
    assert_eq!(next.selection, Selection::Range { anchor: 0, head: 4 });
}
