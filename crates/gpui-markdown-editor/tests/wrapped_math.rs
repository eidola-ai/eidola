//! Row stride on a soft-wrapped line carrying a tall inline construct.
//!
//! `gpui::WrappedLine::paint` advances each wrap row by the single line
//! height it is handed, so the vertical space a tall inline construct
//! reserves has to be part of that per-row stride. Reserving it only in
//! the line's total height left the construct overlapping the glyphs of
//! the row beneath it and piled the whole reservation up as dead air
//! after the logical line. These are geometry-level tests against real
//! window layout, using the editor's own scroll-into-view seam
//! (`content_y_for_offset`) as the measuring instrument.

use gpui::{
    AppContext, Bounds, Entity, TestAppContext, VisualTestContext, WindowBounds, WindowOptions,
    point, px, size,
};
use gpui_component::Root;
use gpui_markdown_editor::{EditorState, MarkdownEditor, MarkdownEditorState, Selection};

/// A plain lead paragraph, a paragraph whose first visual row carries a
/// deeply nested fraction (an ordinary `$\frac{a}{b}$` fits inside the
/// body font's own leading and reserves nothing) and soft-wraps several
/// times after it, then two plain paragraphs whose spacing is the
/// reference inter-block gap.
const DOC: &str = "plain lead paragraph\n\
\n\
A paragraph whose tall $\\frac{\\frac{\\frac{a}{b}}{\\frac{c}{d}}}{\\frac{e}{f}}$ \
appears early on, and which then continues with enough further words that the \
logical line must soft-wrap several times at this measure — so the tall construct \
sits on the first visual row while ordinary text keeps flowing onto the rows \
below it.\n\
\n\
plain tail paragraph\n\
\n\
another plain paragraph\n";

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
) -> (VisualTestContext, Entity<MarkdownEditorState>) {
    let state = EditorState {
        markdown: markdown.to_string(),
        selection: Selection::Cursor(0),
        ..Default::default()
    };
    let (handle, editor) = cx.update(|cx| {
        gpui_component::init(cx);
        let mut inner: Option<Entity<MarkdownEditorState>> = None;
        let bounds = Bounds {
            origin: point(px(0.), px(0.)),
            size: size(px(420.), px(900.)),
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

/// `(top, bottom)` of the visual row hosting `offset`, relative to the
/// laid-out content top.
fn span(
    vcx: &VisualTestContext,
    editor: &Entity<MarkdownEditorState>,
    offset: usize,
) -> (f32, f32) {
    editor
        .read_with(vcx, |e, _| {
            e.content_y_for_offset(offset)
                .map(|(t, b)| (t.into(), b.into()))
        })
        .expect("a laid-out row for the offset")
}

/// Distinct visual-row tops, in layout order, of every offset in
/// `range` — one entry per wrap row of the paragraph.
fn row_tops(
    vcx: &VisualTestContext,
    editor: &Entity<MarkdownEditorState>,
    range: std::ops::Range<usize>,
) -> Vec<f32> {
    let mut tops: Vec<f32> = Vec::new();
    for offset in range {
        if !DOC.is_char_boundary(offset) {
            continue;
        }
        let (top, _) = span(vcx, editor, offset);
        if tops.last().is_none_or(|last| (last - top).abs() > 0.01) {
            tops.push(top);
        }
    }
    tops.sort_by(|a, b| a.partial_cmp(b).expect("finite pixel values"));
    tops.dedup_by(|a, b| (*a - *b).abs() <= 0.01);
    tops
}

fn offset_of(needle: &str) -> usize {
    DOC.find(needle).expect("needle present in the document")
}

/// The paragraph's content, from the first word past the math to its
/// last — the span whose rows we walk.
fn math_paragraph_tail() -> std::ops::Range<usize> {
    let start = offset_of("appears early on");
    let end = offset_of("below it.") + "below it.".len();
    start..end
}

#[test]
fn tall_math_reservation_lands_in_the_per_row_stride() {
    let mut cx = TestAppContext::single();
    let cx = &mut cx;
    let (vcx, editor) = open_editor(cx, DOC);

    // The plain lead paragraph never reserves anything, so its row is
    // exactly one body row high — the reference every step is measured
    // against.
    let (lead_top, lead_bottom) = span(&vcx, &editor, offset_of("plain lead"));
    let row_h = lead_bottom - lead_top;
    assert!(row_h > 0.0, "the reference row must have a height");

    let tops = row_tops(&vcx, &editor, math_paragraph_tail());
    assert!(
        tops.len() >= 3,
        "the paragraph must soft-wrap to at least three rows at this measure (got {})",
        tops.len()
    );

    let steps: Vec<f32> = tops.windows(2).map(|w| w[1] - w[0]).collect();
    for step in &steps {
        assert!(
            (step - steps[0]).abs() <= 0.01,
            "every visual row of one logical line strides equally ({steps:?})"
        );
        // The reservation is what keeps the construct's ink off the
        // glyphs of the row below it; a bare body row height means the
        // reservation never reached the stride.
        assert!(
            *step > row_h + 0.01,
            "the tall construct's reservation must be in the row stride \
             (step {step} vs body row {row_h})"
        );
    }
}

#[test]
fn tall_math_reserves_no_dead_air_after_the_logical_line() {
    let mut cx = TestAppContext::single();
    let cx = &mut cx;
    let (vcx, editor) = open_editor(cx, DOC);

    // Reference: the gap between two ordinary paragraphs.
    let (tail_top, tail_bottom) = span(&vcx, &editor, offset_of("plain tail"));
    let (another_top, _) = span(&vcx, &editor, offset_of("another plain"));
    let reference_gap = another_top - tail_bottom;

    let (lead_top, lead_bottom) = span(&vcx, &editor, offset_of("plain lead"));
    let row_h = lead_bottom - lead_top;

    let tops = row_tops(&vcx, &editor, math_paragraph_tail());
    let step = tops[1] - tops[0];
    let last_bottom = tops.last().expect("at least one row") + row_h;

    // Space between the paragraph's last row of text and the block
    // below it, beyond the ordinary inter-block gap. One row's worth of
    // reservation is correct — that is the construct's descent
    // overshoot, reserved on every row. A *multiple* of it is the
    // reservation for the rows that never received it, piling up after
    // the whole logical line.
    let slack = tail_top - last_bottom - reference_gap;
    let one_rows_reservation = step - row_h;
    assert!(
        slack <= one_rows_reservation + 0.01,
        "reserved space must not pile up after the logical line \
         (slack {slack} vs one row's reservation {one_rows_reservation})"
    );
}
