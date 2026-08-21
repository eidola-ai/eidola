//! Highlight-plugin tests — the keystroke gate for `crate::highlight`.
//!
//! Highlights are host-supplied, **inert** decorations: they never touch the
//! buffer, never forbid a caret position, and never interfere with editing or
//! selection. A plain click on highlighted text reports the covering keys to
//! the host; a drag across a highlight selects normally and reports nothing.
//! The merge/keys math is unit-tested in `src/highlight.rs`; these tests pin
//! the editor-facing contract (including a real-layout click, mirroring the
//! embed plugin's click test).

use std::cell::Cell;
use std::rc::Rc;

use gpui::{
    AnyWindowHandle, AppContext, Bounds, Entity, Modifiers, MouseButton, MouseDownEvent,
    MouseMoveEvent, MouseUpEvent, TestAppContext, VisualTestContext, WindowBounds, WindowOptions,
    point, px, size,
};
use gpui_markdown_editor::{
    EditorEvent, EditorState, HighlightLayer, MarkdownEditor, MarkdownEditorState, Selection,
    embed_marker,
};

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
    state: EditorState,
) -> (AnyWindowHandle, Entity<MarkdownEditorState>) {
    cx.update(|cx| {
        gpui_component::init(cx);
        let mut inner: Option<Entity<MarkdownEditorState>> = None;
        let window = cx
            .open_window(WindowOptions::default(), |window, cx| {
                let editor = cx.new(|cx| MarkdownEditorState::with_state(state, window, cx));
                inner = Some(editor.clone());
                let harness = cx.new(|_| EditorHarness {
                    state: editor.clone(),
                });
                cx.new(|cx| gpui_component::Root::new(harness, window, cx))
            })
            .expect("open window");
        (window.into(), inner.expect("editor built"))
    })
}

fn apply(cx: &mut TestAppContext, editor: &Entity<MarkdownEditorState>, event: EditorEvent) {
    cx.update(|cx| editor.update(cx, |e, cx| e.apply_event_for_test(event, cx)));
}

// ---------------------------------------------------------------------------
// Inertness — highlights never touch the buffer or the caret rules
// ---------------------------------------------------------------------------

#[gpui::test]
fn set_highlights_leaves_the_buffer_and_selection_untouched(cx: &mut TestAppContext) {
    let (_, editor) = open_editor(cx, EditorState::with_markdown("alpha beta gamma"));
    apply(cx, &editor, EditorEvent::SetSelection(Selection::Cursor(7)));
    cx.update(|cx| {
        editor.update(cx, |e, cx| {
            e.set_highlights(vec![(0..5, 1), (6..10, 2)], cx)
        });
    });
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), "alpha beta gamma");
        assert_eq!(e.selection(), Selection::Cursor(7));
        assert!(!e.highlights().is_empty());
    });
}

#[gpui::test]
fn typing_and_selection_work_over_highlighted_text(cx: &mut TestAppContext) {
    let (_, editor) = open_editor(cx, EditorState::with_markdown("alpha beta gamma"));
    cx.update(|cx| {
        editor.update(cx, |e, cx| e.set_highlights(vec![(0..16, 1)], cx));
    });
    // The caret parks freely inside a highlight (no forbidden interior)…
    apply(cx, &editor, EditorEvent::SetSelection(Selection::Cursor(5)));
    editor.read_with(cx, |e, _| assert_eq!(e.selection().head(), 5));
    // …a selection can cover it…
    apply(
        cx,
        &editor,
        EditorEvent::SetSelection(Selection::range(0, 10)),
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(e.selection().selection_range(), 0..10)
    });
    // …and editing splices normally (the highlight is decoration, not text).
    apply(cx, &editor, EditorEvent::SetSelection(Selection::Cursor(5)));
    apply(cx, &editor, EditorEvent::InsertText("!".into()));
    editor.read_with(cx, |e, _| assert_eq!(e.value(), "alpha! beta gamma"));
}

// ---------------------------------------------------------------------------
// insert_embed_marker — the quote-creation host API
// ---------------------------------------------------------------------------

#[gpui::test]
fn insert_embed_marker_into_an_empty_draft_is_the_bare_marker(cx: &mut TestAppContext) {
    let (_, editor) = open_editor(cx, EditorState::with_markdown(""));
    cx.update(|cx| editor.update(cx, |e, cx| e.insert_embed_marker(1, cx)));
    editor.read_with(cx, |e, _| assert_eq!(e.value(), embed_marker(1)));
}

#[gpui::test]
fn insert_embed_marker_pads_itself_into_its_own_paragraph(cx: &mut TestAppContext) {
    // Caret at the end of a paragraph: one blank line is inserted before.
    let (_, editor) = open_editor(cx, EditorState::with_markdown("a thought"));
    apply(cx, &editor, EditorEvent::SetSelection(Selection::Cursor(9)));
    cx.update(|cx| editor.update(cx, |e, cx| e.insert_embed_marker(2, cx)));
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), format!("a thought\n\n{}", embed_marker(2)));
    });

    // Caret between two paragraphs (already blank-line-delimited on both
    // sides): no extra padding.
    let (_, editor) = open_editor(cx, EditorState::with_markdown("before\n\nafter"));
    apply(cx, &editor, EditorEvent::SetSelection(Selection::Cursor(8)));
    cx.update(|cx| editor.update(cx, |e, cx| e.insert_embed_marker(1, cx)));
    editor.read_with(cx, |e, _| {
        assert_eq!(
            e.value(),
            format!("before\n\n{}\n\nafter", embed_marker(1)),
            "mid-document insertion pads both sides into clean paragraphs"
        );
    });
}

#[gpui::test]
fn insert_embed_marker_replaces_an_active_selection(cx: &mut TestAppContext) {
    let (_, editor) = open_editor(cx, EditorState::with_markdown("keep DROP keep2"));
    apply(
        cx,
        &editor,
        EditorEvent::SetSelection(Selection::range(5, 9)),
    );
    cx.update(|cx| editor.update(cx, |e, cx| e.insert_embed_marker(1, cx)));
    editor.read_with(cx, |e, _| {
        assert_eq!(
            e.value(),
            format!("keep\n\n{}\n\nkeep2", embed_marker(1)),
            "the selection is consumed and the marker stands alone; got {:?}",
            e.value()
        );
    });
}

// ---------------------------------------------------------------------------
// remove_embed_marker — the un-place twin
// ---------------------------------------------------------------------------

/// Open an editor whose embed map already carries `ordinals`, so
/// `remove_embed_marker` can recognize the markers (removal reads the *live*
/// map, which is why the host clears the ordinal only afterwards).
fn open_with_embeds(
    cx: &mut TestAppContext,
    markdown: &str,
    ordinals: &[u64],
) -> Entity<MarkdownEditorState> {
    let (_, editor) = open_editor(cx, EditorState::with_markdown(markdown));
    let entries: Vec<(u64, String)> = ordinals
        .iter()
        .map(|o| (*o, format!("quoted {o}")))
        .collect();
    cx.update(|cx| editor.update(cx, |e, cx| e.set_embeds(entries, cx)));
    editor
}

#[gpui::test]
fn remove_embed_marker_rejoins_the_surrounding_paragraphs(cx: &mut TestAppContext) {
    let doc = format!("before\n\n{}\n\nafter", embed_marker(1));
    let editor = open_with_embeds(cx, &doc, &[1]);
    cx.update(|cx| editor.update(cx, |e, cx| e.remove_embed_marker(1, cx)));
    editor.read_with(cx, |e, _| {
        assert_eq!(
            e.value(),
            "before\n\nafter",
            "exactly one paragraph separator goes with the marker"
        );
    });
}

#[gpui::test]
fn remove_embed_marker_at_the_document_end_leaves_no_trailing_blank(cx: &mut TestAppContext) {
    let doc = format!("a thought\n\n{}", embed_marker(3));
    let editor = open_with_embeds(cx, &doc, &[3]);
    cx.update(|cx| editor.update(cx, |e, cx| e.remove_embed_marker(3, cx)));
    editor.read_with(cx, |e, _| assert_eq!(e.value(), "a thought"));
}

#[gpui::test]
fn remove_embed_marker_leaves_the_other_ordinals_alone(cx: &mut TestAppContext) {
    // Ordinals never renumber on removal — the survivors' markers already
    // address them, and the embed map is a map, so a gap is correct.
    let doc = format!(
        "{}\n\nmiddle\n\n{}\n\n{}",
        embed_marker(1),
        embed_marker(2),
        embed_marker(3)
    );
    let editor = open_with_embeds(cx, &doc, &[1, 2, 3]);
    cx.update(|cx| editor.update(cx, |e, cx| e.remove_embed_marker(2, cx)));
    editor.read_with(cx, |e, _| {
        assert_eq!(
            e.value(),
            format!("{}\n\nmiddle\n\n{}", embed_marker(1), embed_marker(3))
        );
    });
}

#[gpui::test]
fn remove_embed_marker_ignores_unmapped_and_defused_markers(cx: &mut TestAppContext) {
    // An ordinal with no mapping is not a recognized embed block — nothing to
    // remove, and the literal text stays.
    let doc = format!("{}\n\ntail", embed_marker(9));
    let editor = open_with_embeds(cx, &doc, &[1]);
    cx.update(|cx| editor.update(cx, |e, cx| e.remove_embed_marker(9, cx)));
    editor.read_with(cx, |e, _| assert_eq!(e.value(), doc, "unmapped: untouched"));

    // A marker the author defused inside a fence is literal text, not a block.
    let fenced = format!("```\n{}\n```", embed_marker(1));
    let editor = open_with_embeds(cx, &fenced, &[1]);
    cx.update(|cx| editor.update(cx, |e, cx| e.remove_embed_marker(1, cx)));
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), fenced, "fenced: still literal, still there");
    });
}

#[gpui::test]
fn insert_then_remove_round_trips_and_typing_survives(cx: &mut TestAppContext) {
    // The full compose gesture: type, quote, keep typing, drop the quote.
    let editor = open_with_embeds(cx, "my reply", &[1]);
    apply(cx, &editor, EditorEvent::SetSelection(Selection::Cursor(8)));
    cx.update(|cx| editor.update(cx, |e, cx| e.insert_embed_marker(1, cx)));
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), format!("my reply\n\n{}", embed_marker(1)));
    });

    apply(cx, &editor, EditorEvent::InsertText("\n\nand more".into()));
    editor.read_with(cx, |e, _| {
        assert_eq!(
            e.value(),
            format!("my reply\n\n{}\n\nand more", embed_marker(1))
        );
    });

    cx.update(|cx| editor.update(cx, |e, cx| e.remove_embed_marker(1, cx)));
    editor.read_with(cx, |e, _| {
        assert_eq!(
            e.value(),
            "my reply\n\nand more",
            "the quote leaves; everything typed around it stays"
        );
    });
}

// ---------------------------------------------------------------------------
// Click routing — real layout (the embed click-test pattern)
// ---------------------------------------------------------------------------

/// Harness whose render attaches `on_highlight_click`, recording the reported
/// keys, rendered read-only (the GUI's source-post configuration).
struct ClickHarness {
    state: Entity<MarkdownEditorState>,
    clicked: Rc<Cell<Option<Vec<u64>>>>,
}

impl gpui::Render for ClickHarness {
    fn render(
        &mut self,
        _: &mut gpui::Window,
        _: &mut gpui::Context<Self>,
    ) -> impl gpui::IntoElement {
        let clicked = self.clicked.clone();
        MarkdownEditor::new(&self.state)
            .disabled(true)
            .on_highlight_click(move |keys, _, _| clicked.set(Some(keys.to_vec())))
    }
}

#[allow(clippy::type_complexity)]
fn open_click_harness(
    cx: &mut TestAppContext,
    markdown: &str,
    highlights: Vec<(std::ops::Range<usize>, u64)>,
) -> (
    VisualTestContext,
    Entity<MarkdownEditorState>,
    Rc<Cell<Option<Vec<u64>>>>,
) {
    open_click_harness_in(cx, markdown, HighlightLayer::Base, highlights)
}

/// The same harness with the ranges on a chosen layer — only
/// [`HighlightLayer::Base`] is supposed to route clicks.
#[allow(clippy::type_complexity)]
fn open_click_harness_in(
    cx: &mut TestAppContext,
    markdown: &str,
    layer: HighlightLayer,
    highlights: Vec<(std::ops::Range<usize>, u64)>,
) -> (
    VisualTestContext,
    Entity<MarkdownEditorState>,
    Rc<Cell<Option<Vec<u64>>>>,
) {
    let clicked: Rc<Cell<Option<Vec<u64>>>> = Rc::new(Cell::new(None));
    let clicked_in = clicked.clone();
    let state = EditorState::with_markdown(markdown);
    let (handle, editor): (AnyWindowHandle, Entity<MarkdownEditorState>) = cx.update(|cx| {
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
                    let harness = cx.new(|_| ClickHarness {
                        state: editor.clone(),
                        clicked: clicked_in,
                    });
                    cx.new(|cx| gpui_component::Root::new(harness, window, cx))
                },
            )
            .expect("open window");
        (window.into(), inner.expect("editor built"))
    });
    cx.update(|cx| {
        editor.update(cx, |e, cx| e.set_highlights_in(layer, highlights, cx));
    });
    let vcx = VisualTestContext::from_window(handle, cx);
    vcx.run_until_parked();
    (vcx, editor, clicked)
}

/// The window position of the first laid-out line's start (offset 0), nudged
/// just inside the glyphs.
fn line_start_target(
    vcx: &VisualTestContext,
    editor: &Entity<MarkdownEditorState>,
) -> gpui::Point<gpui::Pixels> {
    let (x, y) = editor
        .read_with(vcx, |e, _| {
            e.debug_line_source_geometry()
                .into_iter()
                .next()
                .map(|(_, _, x, y, ..)| (x, y))
        })
        .expect("a laid-out line");
    point(px(x + 2.0), px(y + 4.0))
}

#[gpui::test]
fn plain_click_on_highlighted_text_reports_every_covering_key(cx: &mut TestAppContext) {
    // Two overlapping ranges cover offset 0; a third does not.
    let (mut vcx, editor, clicked) = open_click_harness(
        cx,
        "The mitochondria is the powerhouse of the cell",
        vec![(0..16, 7), (0..24, 3), (30..40, 9)],
    );
    let target = line_start_target(&vcx, &editor);

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

    assert_eq!(
        clicked.take(),
        Some(vec![7, 3]),
        "a plain click reports the keys of every range covering the offset"
    );
}

#[gpui::test]
fn drag_across_a_highlight_selects_and_never_fires_the_callback(cx: &mut TestAppContext) {
    let (mut vcx, editor, clicked) = open_click_harness(
        cx,
        "The mitochondria is the powerhouse of the cell",
        vec![(0..24, 7)],
    );
    let target = line_start_target(&vcx, &editor);

    vcx.simulate_event(MouseDownEvent {
        button: MouseButton::Left,
        position: target,
        modifiers: Modifiers::default(),
        click_count: 1,
        first_mouse: false,
    });
    // Drag well to the right before releasing — a real selection gesture.
    let dragged = point(target.x + px(160.), target.y);
    vcx.simulate_event(MouseMoveEvent {
        position: dragged,
        pressed_button: Some(MouseButton::Left),
        modifiers: Modifiers::default(),
    });
    vcx.simulate_event(MouseUpEvent {
        button: MouseButton::Left,
        position: dragged,
        modifiers: Modifiers::default(),
        click_count: 1,
    });
    vcx.run_until_parked();

    assert_eq!(clicked.take(), None, "a drag never navigates");
    editor.read_with(&vcx, |e, _| {
        assert!(
            !e.selection().is_collapsed(),
            "the drag produced a real selection over the highlighted text"
        );
    });
}

#[gpui::test]
fn click_outside_every_highlight_reports_nothing(cx: &mut TestAppContext) {
    // The highlight sits far to the right; a click at the line start (offset
    // ~0) is outside it.
    let (mut vcx, editor, clicked) = open_click_harness(
        cx,
        "The mitochondria is the powerhouse of the cell",
        vec![(30..40, 9)],
    );
    let target = line_start_target(&vcx, &editor);

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

    assert_eq!(clicked.take(), None);
}

// ---------------------------------------------------------------------------
// Layers — independent channels of decoration, only the base one clickable
// ---------------------------------------------------------------------------

#[gpui::test]
fn setting_one_layer_leaves_the_others_alone(cx: &mut TestAppContext) {
    let (_, editor) = open_editor(cx, EditorState::with_markdown("alpha beta gamma"));
    cx.update(|cx| {
        editor.update(cx, |e, cx| {
            e.set_highlights(vec![(0..5, 1)], cx);
            e.set_highlights_in(HighlightLayer::Accent, vec![(6..10, 2)], cx);
        });
    });
    editor.read_with(cx, |e, _| {
        // `set_highlights` and `highlights` are the base layer.
        assert_eq!(e.highlights().keys_at(2), vec![1]);
        assert_eq!(e.highlights_in(HighlightLayer::Base).keys_at(2), vec![1]);
        assert_eq!(e.highlights_in(HighlightLayer::Accent).keys_at(7), vec![2]);
        assert!(e.highlights_in(HighlightLayer::Overlay).is_empty());
    });

    // Clearing one layer does not clear another.
    cx.update(|cx| {
        editor.update(cx, |e, cx| e.set_highlights(Vec::new(), cx));
    });
    editor.read_with(cx, |e, _| {
        assert!(e.highlights().is_empty());
        assert_eq!(e.highlights_in(HighlightLayer::Accent).keys_at(7), vec![2]);
    });
}

#[gpui::test]
fn a_click_on_a_non_base_layer_reports_nothing(cx: &mut TestAppContext) {
    // The same click that navigates on the base layer must be inert on a layer
    // the host paints for its own reasons: a decoration is not a target.
    let (mut vcx, editor, clicked) = open_click_harness_in(
        cx,
        "The mitochondria is the powerhouse of the cell",
        HighlightLayer::Overlay,
        vec![(0..16, 7)],
    );
    let target = line_start_target(&vcx, &editor);

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

    assert_eq!(clicked.take(), None, "an upper layer never routes a click");
    // And the click still placed the caret, as an ordinary click does.
    editor.read_with(&vcx, |e, _| assert!(e.selection().is_collapsed()));
}
