//! Embed-block plugin tests — the keystroke gate for `crate::embed`.
//!
//! Exercises the plugin contract end-to-end through the production dispatch
//! path (`apply_event_for_test` drives the same `update_guarded` pipeline as
//! keystrokes): type-to-create, atomic navigation/selection, delete-as-unit,
//! re-embed by typing, unmapped degradation, literal escaping, canonicalizer
//! non-destruction, and the readonly render. Rendering geometry (the quote
//! container's pixels) is the visual tier's business, not asserted here.

use std::cell::Cell;
use std::rc::Rc;

use gpui::{
    AnyWindowHandle, AppContext, Bounds, Entity, Modifiers, MouseButton, MouseDownEvent,
    MouseUpEvent, TestAppContext, VisualTestContext, WindowBounds, WindowOptions, point, px, size,
};
use gpui_markdown_editor::{
    BlockKind, EditorEvent, EditorState, EmbedMap, MarkdownEditor, MarkdownEditorState, Selection,
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
            .open_window(gpui::WindowOptions::default(), |window, cx| {
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

/// Editor over `markdown` with ordinal 1 mapped to a small quote.
fn embed_editor(
    cx: &mut TestAppContext,
    markdown: &str,
    cursor: usize,
) -> (AnyWindowHandle, Entity<MarkdownEditorState>) {
    let state = EditorState {
        markdown: markdown.into(),
        selection: Selection::Cursor(cursor),
        embeds: EmbedMap::new([(1u64, "quoted **text**".to_string())]),
    };
    open_editor(cx, state)
}

fn apply(cx: &mut TestAppContext, editor: &Entity<MarkdownEditorState>, event: EditorEvent) {
    editor.update(cx, |e, cx| e.apply_event_for_test(event, cx));
    cx.run_until_parked();
}

fn type_str(cx: &mut TestAppContext, editor: &Entity<MarkdownEditorState>, text: &str) {
    for ch in text.chars() {
        apply(cx, editor, EditorEvent::InsertText(ch.to_string()));
    }
}

const MARKER: &str = "{{ embed 1 }}";

// ---------------------------------------------------------------------------
// Promotion + degradation
// ---------------------------------------------------------------------------

#[gpui::test]
fn mapped_marker_promotes_to_embed_block(cx: &mut TestAppContext) {
    let src = format!("before\n\n{MARKER}\n\nafter");
    let (_, editor) = embed_editor(cx, &src, 0);
    let spec = editor.read_with(cx, |e, _| e.render_spec());
    let embed = spec
        .blocks
        .iter()
        .find(|b| matches!(b.kind, BlockKind::Embed { .. }))
        .expect("mapped marker promotes to an Embed block");
    assert!(matches!(embed.kind, BlockKind::Embed { ordinal: 1 }));
    // The marker bytes are fully hidden — the mapped content paints instead.
    assert!(embed.has_hidden_range(embed.source_range.clone()));
    // (A mid-document paragraph's range may fold the trailing newline in.)
    assert_eq!(
        src[embed.source_range.clone()].trim_end_matches('\n'),
        MARKER
    );
}

#[gpui::test]
fn unmapped_marker_renders_as_plain_text(cx: &mut TestAppContext) {
    // Ordinal 9 is not in the map — honest degradation to a paragraph.
    let (_, editor) = embed_editor(cx, "{{ embed 9 }}", 0);
    let spec = editor.read_with(cx, |e, _| e.render_spec());
    assert!(
        spec.blocks
            .iter()
            .all(|b| !matches!(b.kind, BlockKind::Embed { .. })),
        "unmapped ordinal must not promote; got {spec:#?}"
    );
    // And its interior is an ordinary editable position.
    apply(cx, &editor, EditorEvent::SetSelection(Selection::Cursor(5)));
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 5));
}

#[gpui::test]
fn escaped_marker_stays_literal(cx: &mut TestAppContext) {
    // A backslash on the opener breaks the pattern in raw source bytes —
    // this is how the literal text of a *mapped* marker is typed.
    let (_, editor) = embed_editor(cx, "\\{{ embed 1 }}", 0);
    let spec = editor.read_with(cx, |e, _| e.render_spec());
    assert!(
        spec.blocks
            .iter()
            .all(|b| !matches!(b.kind, BlockKind::Embed { .. })),
        "escaped marker must stay literal; got {spec:#?}"
    );
}

#[gpui::test]
fn inline_marker_stays_literal(cx: &mut TestAppContext) {
    let (_, editor) = embed_editor(cx, "see {{ embed 1 }} here", 0);
    let spec = editor.read_with(cx, |e, _| e.render_spec());
    assert!(
        spec.blocks
            .iter()
            .all(|b| !matches!(b.kind, BlockKind::Embed { .. })),
        "inline marker must stay literal; got {spec:#?}"
    );
}

// ---------------------------------------------------------------------------
// Re-embed by typing / type-to-create
// ---------------------------------------------------------------------------

#[gpui::test]
fn typing_a_mapped_marker_materializes_the_block(cx: &mut TestAppContext) {
    let (_, editor) = embed_editor(cx, "intro", 5);
    apply(cx, &editor, EditorEvent::InsertNewline);
    type_str(cx, &editor, MARKER);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), format!("intro\n\n{MARKER}"));
    });
    let spec = editor.read_with(cx, |e, _| e.render_spec());
    assert!(
        spec.blocks
            .iter()
            .any(|b| matches!(b.kind, BlockKind::Embed { ordinal: 1 })),
        "typing the marker re-materializes the block; got {spec:#?}"
    );
}

#[gpui::test]
fn embed_marker_helper_is_canonical(cx: &mut TestAppContext) {
    // What app-core writes into a post body is what the editor recognizes.
    let src = format!("a\n\n{}", embed_marker(1));
    let (_, editor) = embed_editor(cx, &src, 0);
    let spec = editor.read_with(cx, |e, _| e.render_spec());
    assert!(
        spec.blocks
            .iter()
            .any(|b| matches!(b.kind, BlockKind::Embed { ordinal: 1 }))
    );
}

// ---------------------------------------------------------------------------
// Atomicity — caret navigation and selection
// ---------------------------------------------------------------------------

#[gpui::test]
fn caret_skips_over_the_embed_as_one_unit(cx: &mut TestAppContext) {
    let src = format!("ab\n\n{MARKER}\n\ncd");
    let start = 4; // marker start
    let end = start + MARKER.len();
    let (_, editor) = embed_editor(cx, &src, start);

    // Right from the leading edge hops the whole marker.
    apply(cx, &editor, EditorEvent::MoveRight);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), end));
    // Left from the trailing edge hops back.
    apply(cx, &editor, EditorEvent::MoveLeft);
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), start));
}

#[gpui::test]
fn click_inside_the_embed_snaps_to_an_edge(cx: &mut TestAppContext) {
    let src = format!("ab\n\n{MARKER}\n\ncd");
    let start = 4;
    let end = start + MARKER.len();
    let (_, editor) = embed_editor(cx, &src, 0);
    // A SetSelection into the interior (what a click resolves to) snaps to
    // the nearest marker edge.
    apply(
        cx,
        &editor,
        EditorEvent::SetSelection(Selection::Cursor(start + 2)),
    );
    editor.read_with(cx, |e, _| {
        let p = e.cursor_offset();
        assert!(
            p == start || p == end,
            "interior click must snap to an edge; got {p}"
        );
    });
}

#[gpui::test]
fn selection_across_the_embed_covers_it_whole(cx: &mut TestAppContext) {
    let src = format!("ab\n\n{MARKER}\n\ncd");
    let start = 4;
    let end = start + MARKER.len();
    let (_, editor) = embed_editor(cx, &src, 0);
    // A range whose head lands inside the marker snaps out to a boundary —
    // an endpoint never rests inside the atomic block.
    apply(
        cx,
        &editor,
        EditorEvent::SetSelection(Selection::range(0, start + 3)),
    );
    editor.read_with(cx, |e, _| match e.selection() {
        Selection::Range { anchor, head } => {
            assert_eq!(anchor, 0);
            assert!(
                head == start || head == end,
                "selection endpoint must sit on an embed edge; got {head}"
            );
        }
        other => panic!("expected a range selection, got {other:?}"),
    });

    // Deleting a selection that spans the whole embed removes it like any
    // other selected text (no special casing — don't regress selections).
    apply(
        cx,
        &editor,
        EditorEvent::SetSelection(Selection::range(0, src.len())),
    );
    apply(cx, &editor, EditorEvent::DeleteBackward);
    editor.read_with(cx, |e, _| assert_eq!(e.value(), ""));
}

// ---------------------------------------------------------------------------
// Delete-as-unit
// ---------------------------------------------------------------------------

#[gpui::test]
fn backspace_at_trailing_edge_deletes_the_whole_marker(cx: &mut TestAppContext) {
    let src = format!("ab\n\n{MARKER}\n\ncd");
    let end = 4 + MARKER.len();
    let (_, editor) = embed_editor(cx, &src, end);
    apply(cx, &editor, EditorEvent::DeleteBackward);
    editor.read_with(cx, |e, _| {
        assert!(
            !e.value().contains("embed"),
            "backspace at the edge removes the whole marker; got {:?}",
            e.value()
        );
        assert_eq!(e.cursor_offset(), 4);
    });
}

#[gpui::test]
fn delete_forward_at_leading_edge_deletes_the_whole_marker(cx: &mut TestAppContext) {
    let src = format!("ab\n\n{MARKER}\n\ncd");
    let (_, editor) = embed_editor(cx, &src, 4);
    apply(cx, &editor, EditorEvent::DeleteForward);
    editor.read_with(cx, |e, _| {
        assert!(
            !e.value().contains("embed"),
            "delete-forward at the edge removes the whole marker; got {:?}",
            e.value()
        );
    });
}

#[gpui::test]
fn word_delete_takes_the_embed_as_one_unit(cx: &mut TestAppContext) {
    let src = format!("ab\n\n{MARKER}\n\ncd");
    let end = 4 + MARKER.len();
    let (_, editor) = embed_editor(cx, &src, end);
    apply(cx, &editor, EditorEvent::DeleteWordBackward);
    editor.read_with(cx, |e, _| {
        assert!(
            !e.value().contains("embed"),
            "word-delete at the edge removes the whole marker; got {:?}",
            e.value()
        );
    });
}

#[gpui::test]
fn deleted_embed_can_be_retyped(cx: &mut TestAppContext) {
    // The round trip Mike's spec names: delete the block, type the marker
    // again, the block re-materializes (the map is untouched by edits).
    let src = format!("ab\n\n{MARKER}");
    let end = 4 + MARKER.len();
    let (_, editor) = embed_editor(cx, &src, end);
    apply(cx, &editor, EditorEvent::DeleteBackward);
    type_str(cx, &editor, MARKER);
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), src);
    });
    let spec = editor.read_with(cx, |e, _| e.render_spec());
    assert!(
        spec.blocks
            .iter()
            .any(|b| matches!(b.kind, BlockKind::Embed { ordinal: 1 }))
    );
}

// ---------------------------------------------------------------------------
// Round-trip + canonicalizer
// ---------------------------------------------------------------------------

#[gpui::test]
fn canonicalizer_leaves_embed_lines_alone(cx: &mut TestAppContext) {
    // Edits elsewhere in the document must not mangle the marker text —
    // the buffer's round-trip guarantee (persisted markdown stays clean).
    let src = format!("intro\n\n{MARKER}\n\ntail");
    let (_, editor) = embed_editor(cx, &src, 0);
    type_str(cx, &editor, "x");
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), format!("xintro\n\n{MARKER}\n\ntail"));
    });
}

#[gpui::test]
fn typing_at_the_trailing_edge_opens_a_paragraph_below(cx: &mut TestAppContext) {
    // The whole line is the block, so a character typed at the trailing edge
    // opens a fresh paragraph *beside* the embed instead of dissolving it.
    let src = format!("ab\n\n{MARKER}");
    let end = 4 + MARKER.len();
    let (_, editor) = embed_editor(cx, &src, end);
    type_str(cx, &editor, "xy");
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), format!("ab\n\n{MARKER}\n\nxy"));
        // The caret follows the typed text, not the injected separator.
        assert_eq!(e.selection().head(), e.value().len());
    });
    assert_embed_blocks(cx, &editor, 1);
}

#[gpui::test]
fn typing_at_the_leading_edge_opens_a_paragraph_above(cx: &mut TestAppContext) {
    let src = format!("{MARKER}\n\nab");
    let (_, editor) = embed_editor(cx, &src, 0);
    type_str(cx, &editor, "xy");
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), format!("xy\n\n{MARKER}\n\nab"));
        assert_eq!(e.selection().head(), 2);
    });
    assert_embed_blocks(cx, &editor, 1);
}

#[gpui::test]
fn pasting_at_an_edge_opens_a_paragraph_beside_the_embed(cx: &mut TestAppContext) {
    let src = format!("ab\n\n{MARKER}");
    let end = 4 + MARKER.len();
    let (_, editor) = embed_editor(cx, &src, end);
    apply(
        cx,
        &editor,
        EditorEvent::Paste {
            text: "pasted".into(),
            internal: false,
        },
    );
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), format!("ab\n\n{MARKER}\n\npasted"));
    });
    assert_embed_blocks(cx, &editor, 1);
}

#[gpui::test]
fn inserting_a_second_marker_at_an_edge_is_not_padded_twice(cx: &mut TestAppContext) {
    // `insert_embed_marker` pads its own insertion; the line protection must
    // count those newlines rather than stack another blank line on top.
    let src = MARKER.to_string();
    let (_, editor) = embed_editor(cx, &src, MARKER.len());
    editor.update(cx, |e, cx| {
        e.set_embeds(
            [
                (1u64, "quoted **text**".to_string()),
                (2u64, "second quote".to_string()),
            ],
            cx,
        );
        e.insert_embed_marker(2, cx);
    });
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), format!("{MARKER}\n\n{{{{ embed 2 }}}}"));
    });
    assert_embed_blocks(cx, &editor, 2);
}

/// Assert the document renders exactly `n` embed blocks.
fn assert_embed_blocks(cx: &mut TestAppContext, editor: &Entity<MarkdownEditorState>, n: usize) {
    let spec = editor.read_with(cx, |e, _| e.render_spec());
    let got = spec
        .blocks
        .iter()
        .filter(|b| matches!(b.kind, BlockKind::Embed { .. }))
        .count();
    assert_eq!(got, n, "expected {n} embed block(s); got {spec:#?}");
}

#[gpui::test]
fn set_value_preserves_the_embed_map(cx: &mut TestAppContext) {
    // The host seeds content after construction (the readonly space-view
    // path re-seeds on every sync) — the map must survive the swap.
    let (_, editor) = embed_editor(cx, "seed", 0);
    editor.update(cx, |e, cx| {
        e.set_value(format!("swapped\n\n{MARKER}"), cx);
    });
    cx.run_until_parked();
    let spec = editor.read_with(cx, |e, _| e.render_spec());
    assert!(
        spec.blocks
            .iter()
            .any(|b| matches!(b.kind, BlockKind::Embed { ordinal: 1 })),
        "set_value must keep the embed map; got {spec:#?}"
    );
}

#[gpui::test]
fn set_embeds_flips_promotion_live(cx: &mut TestAppContext) {
    // Mapping and unmapping ordinals re-renders without touching the buffer.
    let src = "a\n\n{{ embed 2 }}".to_string();
    let (_, editor) = open_editor(cx, EditorState::with_markdown(src.clone()));
    let spec = editor.read_with(cx, |e, _| e.render_spec());
    assert!(
        spec.blocks
            .iter()
            .all(|b| !matches!(b.kind, BlockKind::Embed { .. }))
    );
    editor.update(cx, |e, cx| e.set_embeds([(2u64, "quote".to_string())], cx));
    cx.run_until_parked();
    let spec = editor.read_with(cx, |e, _| e.render_spec());
    assert!(
        spec.blocks
            .iter()
            .any(|b| matches!(b.kind, BlockKind::Embed { ordinal: 2 }))
    );
    editor.read_with(cx, |e, _| assert_eq!(e.value(), src));
}

// ---------------------------------------------------------------------------
// Readonly editors render embeds too
// ---------------------------------------------------------------------------

#[gpui::test]
fn readonly_editor_promotes_embeds_and_refuses_mutation(_cx: &mut TestAppContext) {
    use gpui_markdown_editor::update::update_readonly;
    let src = format!("ab\n\n{MARKER}");
    let state = EditorState {
        markdown: src.clone(),
        selection: Selection::Cursor(0),
        embeds: EmbedMap::new([(1u64, "quoted".to_string())]),
    };
    // The readonly render spec promotes exactly like the editable one (the
    // GUI's post view is a disabled editor).
    let tree = gpui_markdown_editor::parse(&state.markdown);
    let spec = gpui_markdown_editor::render::render_readonly(&state, &tree);
    assert!(
        spec.blocks
            .iter()
            .any(|b| matches!(b.kind, BlockKind::Embed { ordinal: 1 }))
    );
    // Mutating events are refused wholesale; the marker text is untouched.
    let end = 4 + MARKER.len();
    let s2 = update_readonly(state, EditorEvent::SetSelection(Selection::Cursor(end)));
    let s3 = update_readonly(s2, EditorEvent::DeleteBackward);
    assert_eq!(s3.markdown, src);
    // Selection stays embed-atomic in readonly too: an interior position
    // snaps to an edge.
    let s4 = update_readonly(s3, EditorEvent::SetSelection(Selection::Cursor(4 + 3)));
    let p = s4.selection.head();
    assert!(p == 4 || p == end, "readonly interior snap; got {p}");
}

// ---------------------------------------------------------------------------
// Host click callback — real layout (the table_wrapped harness pattern)
// ---------------------------------------------------------------------------

/// Harness whose render attaches `on_embed_click`, recording the clicked
/// ordinal into a shared cell.
struct ClickHarness {
    state: Entity<MarkdownEditorState>,
    clicked: Rc<Cell<Option<u64>>>,
}

impl gpui::Render for ClickHarness {
    fn render(
        &mut self,
        _: &mut gpui::Window,
        _: &mut gpui::Context<Self>,
    ) -> impl gpui::IntoElement {
        let clicked = self.clicked.clone();
        MarkdownEditor::new(&self.state)
            .on_embed_click(move |ordinal, _, _| clicked.set(Some(ordinal)))
    }
}

/// A **non-final** embed's rendered container must fire the host callback
/// with its ordinal, under real layout. The hit-test reads the ordinal the
/// render recorded on the painted block (`LaidOutBlock::embed_ordinal`) —
/// the earlier implementation re-derived embed ranges from source per click
/// and matched them against painted block ranges by equality, which held
/// only via an incidental cross-module invariant (`inject_empty_paragraphs`
/// strips the parser's folded trailing newline exactly the way
/// `embed_blocks` trims it). This pins the behavior structurally instead.
#[gpui::test]
fn click_on_a_non_final_embed_fires_the_host_callback(cx: &mut TestAppContext) {
    let src = format!("{MARKER}\n\n{{{{ embed 2 }}}}\n\nafter");
    let state = EditorState {
        markdown: src.clone(),
        selection: Selection::Cursor(0),
        embeds: EmbedMap::new([
            (1u64, "first quote".to_string()),
            (2u64, "second quote".to_string()),
        ]),
    };
    let clicked: Rc<Cell<Option<u64>>> = Rc::new(Cell::new(None));
    let clicked_in = clicked.clone();
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
    let mut vcx = VisualTestContext::from_window(handle, cx);
    vcx.run_until_parked();

    // Locate the FIRST embed's painted line (its fully-hidden marker line —
    // the block anchor) and aim a click just inside its container.
    let (x, y) = editor
        .read_with(&vcx, |e, _| {
            e.debug_line_source_geometry()
                .into_iter()
                .find(|(s, en, ..)| *s == 0 && *en >= MARKER.len())
                .map(|(_, _, x, y, ..)| (x, y))
        })
        .expect("the first embed's laid-out line");
    let target = point(px(x + 4.0), px(y + 4.0));

    // The painted block resolves by its render-time ordinal.
    editor.read_with(&vcx, |e, _| {
        assert_eq!(
            e.embed_ordinal_at_position(target),
            Some(1),
            "the painted non-final embed must resolve by render-time ordinal"
        );
    });

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
        clicked.get(),
        Some(1),
        "clicking the first (non-final) embed fires the host callback with its ordinal"
    );
}

// ---------------------------------------------------------------------------
// set_embeds re-snaps a selection that the new map makes forbidden
// ---------------------------------------------------------------------------

/// A caret legally parked inside a *literal* (unmapped) marker must not stay
/// inside once `set_embeds` maps that ordinal — the interior turns
/// caret-forbidden, and typing from the stale position would splice into the
/// hidden marker bytes of a block now rendering as an embed.
#[gpui::test]
fn set_embeds_snaps_a_selection_inside_a_newly_mapped_marker(cx: &mut TestAppContext) {
    // Unmapped: the marker is plain text and the interior is a legal caret.
    let (_, editor) = open_editor(cx, EditorState::with_markdown(MARKER));
    apply(cx, &editor, EditorEvent::SetSelection(Selection::Cursor(5)));
    editor.read_with(cx, |e, _| assert_eq!(e.cursor_offset(), 5));

    // Mapping the ordinal must re-snap the caret to a marker edge.
    editor.update(cx, |e, cx| e.set_embeds([(1u64, "quote".to_string())], cx));
    cx.run_until_parked();
    editor.read_with(cx, |e, _| {
        let p = e.cursor_offset();
        assert!(
            p == 0 || p == MARKER.len(),
            "set_embeds must snap an interior caret to an edge; got {p}"
        );
    });

    // Typing now lands at the snapped edge — the marker bytes stay intact,
    // and the line protection opens a fresh paragraph beside the block.
    type_str(cx, &editor, "x");
    editor.read_with(cx, |e, _| {
        let v = e.value();
        assert!(
            v == format!("x\n\n{MARKER}") || v == format!("{MARKER}\n\nx"),
            "typed text must not splice into the marker; got {v:?}"
        );
    });
}
