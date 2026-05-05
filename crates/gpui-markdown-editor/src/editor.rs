//! `MarkdownEditor` — the gpui entity that owns editor state and routes user
//! input through the pure `update::update` pipeline.
//!
//! Wiring is intentionally minimal:
//!
//! - One `actions!` block declares every editor action (`Backspace`, `Left`,
//!   `Enter`, …). The hosting application must `cx.bind_keys` them — see
//!   `bin/demo.rs` for the standard macOS map.
//! - Text input goes through `EntityInputHandler` so dead-key composition,
//!   non-Latin layouts, and pasted text all work without a separate code
//!   path.

use std::collections::HashMap;
use std::ops::Range;

use gpui::{
    App, Bounds, ClipboardItem, Context, CursorStyle, Entity, EntityInputHandler, FocusHandle,
    Focusable, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, Pixels, Point,
    UTF16Selection, Window, actions, div, prelude::*, px,
};
use gpui_component::Theme;

use crate::element::{BlockElement, LaidOutBlock};
use crate::event::EditorEvent;
use crate::parser::parse;
use crate::render::render;
use crate::render_spec::RenderSpec;
use crate::state::{EditorState, Selection};
use crate::style::MarkdownStyle;
use crate::update;

actions!(
    markdown_editor,
    [
        Backspace,
        Delete,
        Enter,
        ShiftEnter,
        Left,
        Right,
        Up,
        Down,
        ShiftLeft,
        ShiftRight,
        ShiftUp,
        ShiftDown,
        Home,
        End,
        ShiftHome,
        ShiftEnd,
        DocumentStart,
        DocumentEnd,
        ShiftDocumentStart,
        ShiftDocumentEnd,
        SelectAll,
        Copy,
        Cut,
        Paste,
    ]
);

pub struct MarkdownEditor {
    pub state: EditorState,
    style: MarkdownStyle,
    pub focus_handle: FocusHandle,
    is_selecting: bool,
    pub(crate) last_blocks: HashMap<usize, LaidOutBlock>,
    pub(crate) last_bounds: Option<Bounds<Pixels>>,
    pub(crate) frame_input_handler_set: bool,
    marked_range: Option<Range<usize>>,
}

impl MarkdownEditor {
    pub fn new(markdown: impl Into<String>, _: &mut Window, cx: &mut Context<Self>) -> Self {
        let style = MarkdownStyle::from_theme(cx);
        Self::with_state_and_style(EditorState::with_markdown(markdown), style, cx)
    }

    pub fn with_state(state: EditorState, _: &mut Window, cx: &mut Context<Self>) -> Self {
        let style = MarkdownStyle::from_theme(cx);
        Self::with_state_and_style(state, style, cx)
    }

    fn with_state_and_style(
        state: EditorState,
        style: MarkdownStyle,
        cx: &mut Context<Self>,
    ) -> Self {
        Self {
            state,
            style,
            focus_handle: cx.focus_handle(),
            is_selecting: false,
            last_blocks: HashMap::new(),
            last_bounds: None,
            frame_input_handler_set: false,
            marked_range: None,
        }
    }

    pub fn style(mut self, style: MarkdownStyle) -> Self {
        self.style = style;
        self
    }

    pub fn render_spec(&self) -> RenderSpec {
        let tree = parse(&self.state.markdown);
        render(&self.state, &tree)
    }

    pub fn cursor_offset(&self) -> usize {
        self.state.selection.head()
    }

    fn dispatch(&mut self, event: EditorEvent, cx: &mut Context<Self>) {
        let next = std::mem::take(&mut self.state);
        self.state = update::update(next, event);
        self.marked_range = None;
        cx.notify();
    }

    fn backspace(&mut self, _: &Backspace, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::DeleteBackward, cx);
    }
    fn delete(&mut self, _: &Delete, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::DeleteForward, cx);
    }
    fn enter(&mut self, _: &Enter, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::InsertNewline, cx);
    }
    fn shift_enter(&mut self, _: &ShiftEnter, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::InsertLineBreak, cx);
    }

    fn left(&mut self, _: &Left, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::MoveLeft, cx);
    }
    fn right(&mut self, _: &Right, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::MoveRight, cx);
    }
    fn up(&mut self, _: &Up, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::MoveUp, cx);
    }
    fn down(&mut self, _: &Down, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::MoveDown, cx);
    }
    fn shift_left(&mut self, _: &ShiftLeft, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::ExtendLeft, cx);
    }
    fn shift_right(&mut self, _: &ShiftRight, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::ExtendRight, cx);
    }
    fn shift_up(&mut self, _: &ShiftUp, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::ExtendUp, cx);
    }
    fn shift_down(&mut self, _: &ShiftDown, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::ExtendDown, cx);
    }
    fn home(&mut self, _: &Home, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::MoveLineStart, cx);
    }
    fn end(&mut self, _: &End, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::MoveLineEnd, cx);
    }
    fn shift_home(&mut self, _: &ShiftHome, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::ExtendLineStart, cx);
    }
    fn shift_end(&mut self, _: &ShiftEnd, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::ExtendLineEnd, cx);
    }
    fn document_start(&mut self, _: &DocumentStart, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::MoveDocumentStart, cx);
    }
    fn document_end(&mut self, _: &DocumentEnd, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::MoveDocumentEnd, cx);
    }
    fn shift_document_start(
        &mut self,
        _: &ShiftDocumentStart,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.dispatch(EditorEvent::ExtendDocumentStart, cx);
    }
    fn shift_document_end(&mut self, _: &ShiftDocumentEnd, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::ExtendDocumentEnd, cx);
    }

    fn select_all(&mut self, _: &SelectAll, _: &mut Window, cx: &mut Context<Self>) {
        let len = self.state.markdown.len();
        self.dispatch(EditorEvent::SetSelection(Selection::range(0, len)), cx);
    }

    fn copy(&mut self, _: &Copy, _: &mut Window, cx: &mut Context<Self>) {
        let range = self.state.selection.selection_range();
        if range.is_empty() {
            return;
        }
        let text = self.state.markdown[range].to_string();
        cx.write_to_clipboard(ClipboardItem::new_string(text));
    }

    fn cut(&mut self, _: &Cut, _: &mut Window, cx: &mut Context<Self>) {
        let range = self.state.selection.selection_range();
        if range.is_empty() {
            return;
        }
        let text = self.state.markdown[range].to_string();
        cx.write_to_clipboard(ClipboardItem::new_string(text));
        self.dispatch(EditorEvent::DeleteForward, cx);
    }

    fn paste(&mut self, _: &Paste, _: &mut Window, cx: &mut Context<Self>) {
        if let Some(text) = cx.read_from_clipboard().and_then(|item| item.text()) {
            self.dispatch(EditorEvent::InsertText(text.to_string()), cx);
        }
    }

    fn on_mouse_down(&mut self, event: &MouseDownEvent, _: &mut Window, cx: &mut Context<Self>) {
        let offset = self.offset_for_position(event.position);
        self.is_selecting = true;
        let new_sel = if event.modifiers.shift {
            Selection::range(self.state.selection.anchor(), offset)
        } else {
            Selection::Cursor(offset)
        };
        self.dispatch(EditorEvent::SetSelection(new_sel), cx);
    }

    fn on_mouse_up(&mut self, _: &MouseUpEvent, _: &mut Window, _: &mut Context<Self>) {
        self.is_selecting = false;
    }

    fn on_mouse_move(&mut self, event: &MouseMoveEvent, _: &mut Window, cx: &mut Context<Self>) {
        if !self.is_selecting {
            return;
        }
        let offset = self.offset_for_position(event.position);
        let new_sel = Selection::range(self.state.selection.anchor(), offset);
        self.dispatch(EditorEvent::SetSelection(new_sel), cx);
    }

    fn offset_for_position(&self, position: Point<Pixels>) -> usize {
        if self.last_blocks.is_empty() {
            return 0;
        }
        let mut keys: Vec<&usize> = self.last_blocks.keys().collect();
        keys.sort();
        if let Some(first_key) = keys.first()
            && let Some(first_line) = self.last_blocks[*first_key].lines.first()
            && position.y < first_line.origin.y
        {
            return 0;
        }
        for key in &keys {
            let block = &self.last_blocks[*key];
            for line in &block.lines {
                let line_top = line.origin.y;
                let line_bottom = line_top + line.wrapped_height;
                if position.y >= line_top && position.y < line_bottom {
                    let local = Point::new(position.x - line.origin.x, position.y - line.origin.y);
                    return line.source_offset_for_local_point(local);
                }
            }
        }
        self.state.markdown.len()
    }

    // ---- UTF-16 conversion helpers ----

    fn offset_from_utf16(&self, offset: usize) -> usize {
        let mut utf8 = 0;
        let mut utf16 = 0;
        for ch in self.state.markdown.chars() {
            if utf16 >= offset {
                break;
            }
            utf16 += ch.len_utf16();
            utf8 += ch.len_utf8();
        }
        utf8
    }

    fn offset_to_utf16(&self, offset: usize) -> usize {
        let mut utf16 = 0;
        let mut utf8 = 0;
        for ch in self.state.markdown.chars() {
            if utf8 >= offset {
                break;
            }
            utf8 += ch.len_utf8();
            utf16 += ch.len_utf16();
        }
        utf16
    }

    fn range_to_utf16(&self, range: &Range<usize>) -> Range<usize> {
        self.offset_to_utf16(range.start)..self.offset_to_utf16(range.end)
    }

    fn range_from_utf16(&self, range_utf16: &Range<usize>) -> Range<usize> {
        self.offset_from_utf16(range_utf16.start)..self.offset_from_utf16(range_utf16.end)
    }
}

impl EntityInputHandler for MarkdownEditor {
    fn text_for_range(
        &mut self,
        range_utf16: Range<usize>,
        actual_range: &mut Option<Range<usize>>,
        _: &mut Window,
        _: &mut Context<Self>,
    ) -> Option<String> {
        let range = self.range_from_utf16(&range_utf16);
        actual_range.replace(self.range_to_utf16(&range));
        Some(self.state.markdown[range].to_string())
    }

    fn selected_text_range(
        &mut self,
        _: bool,
        _: &mut Window,
        _: &mut Context<Self>,
    ) -> Option<UTF16Selection> {
        let sel = self.state.selection;
        Some(UTF16Selection {
            range: self.range_to_utf16(&(sel.lower_bound()..sel.upper_bound())),
            reversed: sel.head() < sel.anchor(),
        })
    }

    fn marked_text_range(&self, _: &mut Window, _: &mut Context<Self>) -> Option<Range<usize>> {
        self.marked_range.as_ref().map(|r| self.range_to_utf16(r))
    }

    fn unmark_text(&mut self, _: &mut Window, _: &mut Context<Self>) {
        self.marked_range = None;
    }

    fn replace_text_in_range(
        &mut self,
        range_utf16: Option<Range<usize>>,
        new_text: &str,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let target = range_utf16
            .as_ref()
            .map(|r| self.range_from_utf16(r))
            .or_else(|| self.marked_range.clone());
        if let Some(range) = target {
            self.dispatch(
                EditorEvent::SetSelection(Selection::range(range.start, range.end)),
                cx,
            );
        }
        self.dispatch(EditorEvent::InsertText(new_text.to_string()), cx);
        self.marked_range = None;
    }

    fn replace_and_mark_text_in_range(
        &mut self,
        range_utf16: Option<Range<usize>>,
        new_text: &str,
        new_selected_range_utf16: Option<Range<usize>>,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let range = range_utf16
            .as_ref()
            .map(|r| self.range_from_utf16(r))
            .or_else(|| self.marked_range.clone())
            .unwrap_or_else(|| self.state.selection.selection_range());

        let mut new_md = String::with_capacity(
            self.state.markdown.len() - (range.end - range.start) + new_text.len(),
        );
        new_md.push_str(&self.state.markdown[..range.start]);
        new_md.push_str(new_text);
        new_md.push_str(&self.state.markdown[range.end..]);
        self.state.markdown = new_md;

        if !new_text.is_empty() {
            self.marked_range = Some(range.start..range.start + new_text.len());
        } else {
            self.marked_range = None;
        }

        let cursor = if let Some(sel_utf16) = new_selected_range_utf16 {
            let local = self.range_from_utf16(&sel_utf16);
            range.start + local.end
        } else {
            range.start + new_text.len()
        };
        self.state.selection = Selection::Cursor(cursor);
        cx.notify();
    }

    fn bounds_for_range(
        &mut self,
        range_utf16: Range<usize>,
        _bounds: Bounds<Pixels>,
        _: &mut Window,
        _: &mut Context<Self>,
    ) -> Option<Bounds<Pixels>> {
        let range = self.range_from_utf16(&range_utf16);
        for block in self.last_blocks.values() {
            for line in &block.lines {
                if line.contains_source_offset(range.start) {
                    let start = line.local_position_for_source_offset(range.start);
                    let x0 = line.origin.x + start.x;
                    let y0 = line.origin.y + start.y;
                    return Some(Bounds::from_corners(
                        Point::new(x0, y0),
                        Point::new(x0 + px(1.0), y0 + line.row_height),
                    ));
                }
            }
        }
        None
    }

    fn character_index_for_point(
        &mut self,
        point: Point<Pixels>,
        _: &mut Window,
        _: &mut Context<Self>,
    ) -> Option<usize> {
        Some(self.offset_to_utf16(self.offset_for_position(point)))
    }
}

impl Focusable for MarkdownEditor {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for MarkdownEditor {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // Refresh the style each frame so theme-mode flips re-derive colors
        // (cheap — the style struct is plain Arc / Hsla).
        self.style = self.style.clone();
        // ^ keep the caller's overrides; only refresh theme-derived fields
        let theme = Theme::global(cx);
        self.style.text_color = theme.foreground;
        self.style.delimiter_color = theme.muted_foreground;
        self.style.background = theme.background;
        self.style.caret_color = theme.caret;
        self.style.selection_color = theme.selection;

        // Reset per-frame state. Block elements re-populate `last_blocks`
        // during paint.
        self.last_blocks.clear();
        self.frame_input_handler_set = false;

        let view: Entity<Self> = cx.entity();
        let style = self.style.clone();
        let spec = self.render_spec();

        // Register the IME / EntityInputHandler at the container level so
        // typed text and dead-key composition flow through
        // `replace_text_in_range`.
        self.last_bounds = None;

        let block_count = spec.blocks.len();
        let mut container = div()
            .id("markdown-editor")
            .key_context("MarkdownEditor")
            .track_focus(&self.focus_handle(cx))
            .cursor(CursorStyle::IBeam)
            .size_full()
            .flex()
            .flex_col()
            .text_size(self.style.font_size)
            .text_color(self.style.text_color)
            .font_family(self.style.font_family.clone())
            .on_action(cx.listener(Self::backspace))
            .on_action(cx.listener(Self::delete))
            .on_action(cx.listener(Self::enter))
            .on_action(cx.listener(Self::shift_enter))
            .on_action(cx.listener(Self::left))
            .on_action(cx.listener(Self::right))
            .on_action(cx.listener(Self::up))
            .on_action(cx.listener(Self::down))
            .on_action(cx.listener(Self::shift_left))
            .on_action(cx.listener(Self::shift_right))
            .on_action(cx.listener(Self::shift_up))
            .on_action(cx.listener(Self::shift_down))
            .on_action(cx.listener(Self::home))
            .on_action(cx.listener(Self::end))
            .on_action(cx.listener(Self::shift_home))
            .on_action(cx.listener(Self::shift_end))
            .on_action(cx.listener(Self::document_start))
            .on_action(cx.listener(Self::document_end))
            .on_action(cx.listener(Self::shift_document_start))
            .on_action(cx.listener(Self::shift_document_end))
            .on_action(cx.listener(Self::select_all))
            .on_action(cx.listener(Self::copy))
            .on_action(cx.listener(Self::cut))
            .on_action(cx.listener(Self::paste))
            .on_mouse_down(MouseButton::Left, cx.listener(Self::on_mouse_down))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .on_mouse_up_out(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .on_mouse_move(cx.listener(Self::on_mouse_move));

        let block_starts: Vec<usize> = spec.blocks.iter().map(|b| b.source_range.start).collect();
        for (idx, block) in spec.blocks.into_iter().enumerate() {
            let is_last = idx + 1 == block_count;
            let next_block_start = block_starts.get(idx + 1).copied();
            container = container.child(BlockElement::new(
                block,
                idx,
                is_last,
                next_block_start,
                view.clone(),
                style.clone(),
            ));
        }

        // The first `BlockElement::paint` of the frame registers the
        // `EntityInputHandler` (so IME / typed text routes through
        // `replace_text_in_range`). The guard flag is reset above on each
        // render.
        container
    }
}
