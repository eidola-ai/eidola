//! Wrapped-table navigation — geometry-level regression tests for the
//! box-model table layout (`element::layout_table`): cells wrap
//! independently at their column's width, and caret/click/vertical
//! navigation must resolve into the right cell box. These run with
//! real window layout (the behavior-test harness), unlike the pure
//! keystroke tests in `tests/table.rs`.

use gpui::{
    AppContext, Bounds, Entity, Modifiers, MouseButton, MouseDownEvent, MouseUpEvent,
    TestAppContext, VisualTestContext, WindowBounds, WindowOptions, point, px, size,
};
use gpui_component::Root;
use gpui_markdown_editor::{
    Down, EditorEvent, EditorState, MarkdownEditor, MarkdownEditorState, Selection,
};

/// Two columns; the Detail column's content is far wider than the
/// narrow window, so it wraps to several sub-lines while Topic stays
/// a single line at min-content.
const WRAP_DOC: &str = "\
| Topic | Detail |\n\
| :-- | :-- |\n\
| Attestation | Every new TLS handshake re-verifies the enclave measurement against the pinned trust root |\n\
| Refunds | Unspent holds return to the wallet through the recovery endpoint |\n";

struct EditorHarness {
    state: Entity<MarkdownEditorState>,
}

impl gpui::Render for EditorHarness {
    fn render(
        &mut self,
        _: &mut gpui::Window,
        _: &mut gpui::Context<Self>,
    ) -> impl gpui::IntoElement {
        MarkdownEditor::new(&self.state)
    }
}

fn open_editor(
    cx: &mut TestAppContext,
    markdown: &str,
    cursor: usize,
) -> (VisualTestContext, Entity<MarkdownEditorState>) {
    let state = EditorState {
        markdown: markdown.to_string(),
        selection: Selection::Cursor(cursor),
        ..Default::default()
    };
    let (handle, editor) = cx.update(|cx| {
        gpui_component::init(cx);
        let mut inner: Option<Entity<MarkdownEditorState>> = None;
        let bounds = Bounds {
            origin: point(px(0.), px(0.)),
            size: size(px(560.), px(600.)),
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
                    let harness = cx.new(|_| EditorHarness {
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

/// The laid-out piece whose source range contains `offset` (first
/// match — cell pieces precede chrome fragments, mirroring the caret
/// resolution order).
fn piece_for_offset(
    vcx: &VisualTestContext,
    editor: &Entity<MarkdownEditorState>,
    offset: usize,
) -> (f32, f32, f32, f32) {
    editor
        .read_with(vcx, |e, _| {
            e.debug_line_source_geometry()
                .into_iter()
                .find(|(s, en, ..)| *s <= offset && offset <= *en)
                .map(|(_, _, x, y, w, h)| (x, y, w, h))
        })
        .expect("a piece containing the offset")
}

fn cell_range(doc: &str, needle: &str) -> (usize, usize) {
    let start = doc.find(needle).unwrap();
    (start, start + needle.len())
}

#[test]
fn detail_cells_wrap_and_topic_column_floors_at_min_content() {
    let mut cx = TestAppContext::single();
    let cx = &mut cx;
    let (vcx, editor) = open_editor(cx, WRAP_DOC, 0);
    let line_h = 17.0; // any positive floor; real assertions are relative

    // The wide Detail cell wraps: its piece is taller than one row.
    let (detail_start, _) = cell_range(WRAP_DOC, "Every new TLS");
    let (_, _, _, h_detail) = piece_for_offset(&vcx, &editor, detail_start);
    assert!(
        h_detail > line_h * 1.5,
        "the Detail cell must wrap to multiple sub-lines (height {h_detail})"
    );

    // The Topic column floors at min-content: `Attestation` stays a
    // single unwrapped line.
    let (topic_start, _) = cell_range(WRAP_DOC, "Attestation");
    let (_, _, _, h_topic) = piece_for_offset(&vcx, &editor, topic_start);
    assert!(
        h_topic < h_detail,
        "`Attestation` must not wrap (topic {h_topic} vs detail {h_detail})"
    );
}

#[test]
fn click_lands_in_a_wrapped_cells_second_sub_line() {
    let mut cx = TestAppContext::single();
    let cx = &mut cx;
    let (mut vcx, editor) = open_editor(cx, WRAP_DOC, 0);

    let (detail_start, detail_end) = cell_range(
        WRAP_DOC,
        "Every new TLS handshake re-verifies the enclave measurement against the pinned trust root",
    );
    let (cell_x, cell_y, _, cell_h) = piece_for_offset(&vcx, &editor, detail_start);
    // Aim inside the SECOND sub-line of the wrapped cell: a bit right
    // of the box's left edge, one wrap row down.
    let row_h = cell_h / 3.0; // the fixture wraps to 3 sub-lines at 560px
    let target = point(px(cell_x + 24.0), px(cell_y + row_h * 1.5));

    vcx.simulate_event(MouseDownEvent {
        button: MouseButton::Left,
        position: target,
        modifiers: Modifiers::default(),
        click_count: 1,
        first_mouse: false,
    });
    vcx.simulate_event(MouseUpEvent {
        button: MouseButton::Left,
        position: target,
        modifiers: Modifiers::default(),
        click_count: 1,
    });
    vcx.run_until_parked();

    let head = editor.read_with(&vcx, |e, _| e.selection().head());
    assert!(
        head > detail_start && head <= detail_end,
        "the click must land inside the wrapped Detail cell (offset {head}, cell {detail_start}..{detail_end})"
    );
    // …and specifically past the first sub-line's content (the first
    // wrap row holds roughly the first third of the cell).
    assert!(
        head > detail_start + 20,
        "the click aimed at the second sub-line must not resolve to the first (offset {head})"
    );
}

#[test]
fn down_moves_within_a_wrapped_cell_then_hops_to_the_same_column() {
    let mut cx = TestAppContext::single();
    let cx = &mut cx;
    let (mut vcx, editor) = open_editor(cx, WRAP_DOC, 0);

    let (detail_start, detail_end) = cell_range(
        WRAP_DOC,
        "Every new TLS handshake re-verifies the enclave measurement against the pinned trust root",
    );
    // Park the caret a few glyphs into the wrapped Detail cell.
    vcx.update(|_, cx| {
        editor.update(cx, |e, cx| {
            e.apply_event_for_test(
                EditorEvent::SetSelection(Selection::Cursor(detail_start + 3)),
                cx,
            );
        });
    });
    vcx.run_until_parked();

    // Down #1: stays inside the same wrapped cell (next sub-line).
    let focus = editor.read_with(&vcx, |e, _| e.focus_handle.clone());
    vcx.update(|window, cx| focus.dispatch_action(&Down, window, cx));
    vcx.run_until_parked();
    let head1 = editor.read_with(&vcx, |e, _| e.selection().head());
    assert!(
        head1 > detail_start + 3 && head1 <= detail_end,
        "Down must move to the wrapped cell's next sub-line, not leave the cell (offset {head1})"
    );

    // Down twice more: through the cell's last sub-line into the NEXT
    // row — landing in its Detail column (the x tie-break), never in
    // the `Refunds` topic cell.
    vcx.update(|window, cx| focus.dispatch_action(&Down, window, cx));
    vcx.run_until_parked();
    vcx.update(|window, cx| focus.dispatch_action(&Down, window, cx));
    vcx.run_until_parked();
    let (refunds_detail_start, refunds_detail_end) = cell_range(
        WRAP_DOC,
        "Unspent holds return to the wallet through the recovery endpoint",
    );
    let head3 = editor.read_with(&vcx, |e, _| e.selection().head());
    assert!(
        head3 >= refunds_detail_start && head3 <= refunds_detail_end,
        "Down out of a wrapped cell must land in the next row's SAME column \
         (offset {head3}, expected within {refunds_detail_start}..{refunds_detail_end})"
    );
}
