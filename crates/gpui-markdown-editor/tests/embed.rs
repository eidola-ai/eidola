//! Embed-block plugin tests — the keystroke gate for `crate::embed`.
//!
//! Exercises the plugin contract end-to-end through the production dispatch
//! path (`apply_event_for_test` drives the same `update_guarded` pipeline as
//! keystrokes): type-to-create, atomic navigation/selection, delete-as-unit,
//! re-embed by typing, unmapped degradation, literal escaping, canonicalizer
//! non-destruction, and the readonly render. Rendering geometry (the quote
//! container's pixels) is the visual tier's business, not asserted here.

use gpui::{AnyWindowHandle, AppContext, Entity, TestAppContext};
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
fn typing_at_an_edge_degrades_the_marker_honestly(cx: &mut TestAppContext) {
    // Typing at the trailing edge extends the paragraph past the pattern —
    // the construct honestly degrades to plain text (and undoes by deleting
    // the typed character). Documented behavior, not an accident.
    let src = format!("ab\n\n{MARKER}");
    let end = 4 + MARKER.len();
    let (_, editor) = embed_editor(cx, &src, end);
    type_str(cx, &editor, "x");
    editor.read_with(cx, |e, _| {
        assert_eq!(e.value(), format!("ab\n\n{MARKER}x"));
    });
    let spec = editor.read_with(cx, |e, _| e.render_spec());
    assert!(
        spec.blocks
            .iter()
            .all(|b| !matches!(b.kind, BlockKind::Embed { .. })),
        "a marker with trailing content is plain text; got {spec:#?}"
    );
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
