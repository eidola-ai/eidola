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
        Tab,
        ShiftTab,
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
        /// Move the cursor to the start of the previous word
        /// (Unicode word boundary). Default macOS keybinding:
        /// `alt-left`.
        WordLeft,
        /// Move the cursor to the end of the next word. Default macOS
        /// keybinding: `alt-right`.
        WordRight,
        /// Extend the selection to the start of the previous word.
        /// Default macOS keybinding: `alt-shift-left`.
        ShiftWordLeft,
        /// Extend the selection to the end of the next word. Default
        /// macOS keybinding: `alt-shift-right`.
        ShiftWordRight,
        /// Delete back to the start of the previous word. Default
        /// macOS keybinding: `alt-backspace`.
        DeleteWordBackward,
        /// Delete forward to the end of the next word. Default macOS
        /// keybinding: `alt-delete`.
        DeleteWordForward,
        /// Delete from the cursor back to the visible start of the
        /// current line (past any hidden chain prefix). Default macOS
        /// keybinding: `cmd-backspace`.
        DeleteToLineStart,
        /// Delete from the cursor forward to the end of the current
        /// line (the byte before its trailing `\n`). Default macOS
        /// keybinding: `cmd-delete`.
        DeleteToLineEnd,
        SelectAll,
        Copy,
        Cut,
        Paste,
    ]
);

/// Sentinel string tagged onto every `copy` / `cut` clipboard write via
/// `ClipboardItem::new_string_with_metadata`. Paired with the metadata
/// check in `paste` so an editor → editor round-trip can skip the
/// markdown canonicalization pass — the bytes are already canonical
/// and re-parsing them risks rounding their structure.
///
/// The literal is intentionally crate-namespaced (`gpui-markdown-editor`)
/// rather than app-namespaced (`eidola-markdown`), matching the AGENTS
/// note that this crate carries no Eidola-specific symbols.
const CLIPBOARD_SENTINEL: &str = "gpui-markdown-editor";

/// Normalize CRLF (Windows) and bare CR (legacy macOS) line endings to
/// LF so downstream chain-prefix injection, parser passes, and
/// `enforce_invariants` only have to reason about `\n`. The clipboard
/// layer on most modern OSes already delivers LF, but Windows
/// applications and some web sources still emit CRLF.
fn normalize_line_endings(text: &str) -> String {
    if !text.contains('\r') {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\r' {
            out.push('\n');
            // Swallow the LF half of CRLF; bare CR also collapses to LF.
            if chars.peek() == Some(&'\n') {
                chars.next();
            }
        } else {
            out.push(c);
        }
    }
    out
}

pub struct MarkdownEditor {
    pub state: EditorState,
    style: MarkdownStyle,
    pub focus_handle: FocusHandle,
    is_selecting: bool,
    pub(crate) last_blocks: HashMap<usize, LaidOutBlock>,
    pub(crate) last_bounds: Option<Bounds<Pixels>>,
    pub(crate) frame_input_handler_set: bool,
    marked_range: Option<Range<usize>>,
    /// Per-block horizontal scroll offset (positive = content scrolled
    /// left under the visible band). Keyed by block index; entries
    /// persist across re-renders so a user's scroll position survives
    /// re-shape. Stale entries (block index no longer present in this
    /// frame's spec) are harmless — they're simply not read.
    code_block_scrolls: HashMap<usize, Pixels>,
    /// Persistent "intended visual column" for consecutive Up / Down
    /// arrow presses. When the user crosses a short row (or one that
    /// wraps at a different column), the cursor's source-byte column
    /// shrinks; we remember the *visual* x from the press that started
    /// the streak so the cursor returns to that x on the next long
    /// row. Reset to `None` on any non-vertical event in
    /// `dispatch_reset_intended_x_unless_vertical`.
    intended_x: Option<Pixels>,
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
            code_block_scrolls: HashMap::new(),
            intended_x: None,
        }
    }

    pub(crate) fn code_block_scroll(&self, block_index: usize) -> Pixels {
        self.code_block_scrolls
            .get(&block_index)
            .copied()
            .unwrap_or(px(0.0))
    }

    pub(crate) fn set_code_block_scroll(&mut self, block_index: usize, offset: Pixels) {
        self.code_block_scrolls.insert(block_index, offset);
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
        // Any non-vertical event invalidates the intended-x streak.
        // Vertical events (handled by `vertical_move` below) update
        // `intended_x` directly without going through this helper.
        self.intended_x = None;
        let next = std::mem::take(&mut self.state);
        self.state = update::update(next, event);
        self.marked_range = None;
        cx.notify();
    }

    /// Common path for Up / Down / Shift+Up / Shift+Down: try a
    /// *visual* move that respects the laid-out, soft-wrapped row
    /// geometry from the previous frame. Returns the new caret offset
    /// (the dispatch site decides between `Cursor` and `Range`
    /// selection shapes). Falls back to `None` when there's no layout
    /// to consult (pre-paint state, headless tests); callers can then
    /// route through the source-byte `MoveUp` / `MoveDown` event as a
    /// best-effort approximation.
    ///
    /// **Intended-x preservation.** The first vertical key press of a
    /// streak captures the cursor's *visual* x (block origin + local x
    /// returned by `local_position_for_source_offset`) and stores it
    /// on `self.intended_x`. Subsequent presses re-use that anchor
    /// instead of the (possibly column-shrunk) cursor's current x, so
    /// a long line → wrapped short row → long line round-trip lands
    /// the caret back at its original visual column. Non-vertical
    /// events clear the anchor via [`dispatch`].
    fn visual_move_caret(&mut self, direction: i32) -> Option<usize> {
        if self.last_blocks.is_empty() {
            return None;
        }
        let cursor = self.state.selection.head();
        let mut keys: Vec<usize> = self.last_blocks.keys().copied().collect();
        keys.sort();

        // Find the LaidOutLine containing the cursor. Each block has
        // multiple lines; multiple blocks claim no shared bytes (post
        // `inject_empty_paragraphs` synthesizes them with disjoint
        // ranges), so the first containing line wins.
        let mut current: Option<(&crate::element::LaidOutLine, usize)> = None;
        for k in &keys {
            let block = &self.last_blocks[k];
            for line in &block.lines {
                if line.contains_source_offset(cursor) {
                    current = Some((line, *k));
                    break;
                }
            }
            if current.is_some() {
                break;
            }
        }
        let (line, _) = current?;

        // Local point of the cursor inside the current LaidOutLine.
        // `local_position_for_source_offset` accounts for soft wraps
        // via `WrappedLine::position_for_index`, so `local.y` is the
        // wrap-row's y inside the line and `local.x` is the visual x
        // within that wrap row.
        let local = line.local_position_for_source_offset(cursor);
        let global_x = line.origin.x + local.x;
        let target_x = self.intended_x.unwrap_or(global_x);
        let row_h = line.row_height;
        if row_h <= px(0.) {
            return None;
        }
        // Step exactly one wrap-row vertically. `local.y` from
        // `position_for_index` is already row-aligned (multiples of
        // row_height); shifting by ±row_h lands at the next row.
        let target_global_y = line.origin.y + local.y + row_h * (direction as f32);

        let current_top = line.origin.y;
        let current_bot = current_top + line.wrapped_height;

        // Intra-line wrap-row navigation: target_y still falls inside
        // the current logical line's vertical extent. The line wraps,
        // we're stepping between wrap rows of the same shaped text.
        let target_line: &crate::element::LaidOutLine =
            if target_global_y >= current_top && target_global_y < current_bot {
                line
            } else {
                // Cross-line navigation: find the closest line in the
                // direction of motion. The current line is filtered out
                // (it's behind us), and lines on the *wrong* side of the
                // motion are filtered out (so a Down doesn't backtrack to
                // a line above the cursor when no line below exists, and
                // vice-versa). Within the direction-filtered set, pick
                // the line whose vertical bounds are closest to
                // `target_global_y` — this absorbs the inter-block
                // paragraph_gap by snapping the target into the nearest
                // candidate row.
                let mut best: Option<(&crate::element::LaidOutLine, Pixels)> = None;
                for k in &keys {
                    let block = &self.last_blocks[k];
                    for cand in &block.lines {
                        let top = cand.origin.y;
                        let bot = top + cand.wrapped_height;
                        if direction < 0 {
                            if bot > current_top {
                                continue;
                            }
                        } else if top < current_bot {
                            continue;
                        }
                        let dist = if target_global_y < top {
                            top - target_global_y
                        } else if target_global_y >= bot {
                            target_global_y - bot
                        } else {
                            px(0.)
                        };
                        match best {
                            Some((_, d)) if d <= dist => {}
                            _ => best = Some((cand, dist)),
                        }
                    }
                }
                best.map(|(l, _)| l)?
            };

        // Clamp y to the target line's extent so a target that fell
        // in a paragraph_gap (no row owned it directly) still picks a
        // sensible wrap-row inside the snapped-to line.
        let top = target_line.origin.y;
        let bot = top + target_line.wrapped_height;
        let clamped_y = if target_global_y < top {
            px(0.)
        } else if target_global_y >= bot {
            target_line.wrapped_height - px(1.)
        } else {
            target_global_y - top
        };
        let local_target = Point::new(target_x - target_line.origin.x, clamped_y);
        let new_offset = target_line.source_offset_for_local_point(local_target);

        // Persist the original visual x for the next press in this
        // streak.
        self.intended_x = Some(target_x);
        Some(new_offset)
    }

    /// Dispatch path for Up / Down / Shift+Up / Shift+Down. Tries
    /// `visual_move_caret` first; on success builds the appropriate
    /// `Selection` and calls `update::update(SetSelection(_))`. On
    /// failure (no layout / cursor not in any laid-out line) falls
    /// back to the source-byte event so headless tests and pre-paint
    /// state still move predictably.
    fn vertical_move(
        &mut self,
        direction: i32,
        extending: bool,
        fallback: EditorEvent,
        cx: &mut Context<Self>,
    ) {
        let new_head = match self.visual_move_caret(direction) {
            Some(offset) => offset,
            None => {
                self.intended_x = None;
                let next = std::mem::take(&mut self.state);
                self.state = update::update(next, fallback);
                self.marked_range = None;
                cx.notify();
                return;
            }
        };
        let new_sel = if extending {
            let anchor = match self.state.selection {
                Selection::Cursor(p) => p,
                Selection::Range { anchor, .. } => anchor,
            };
            if anchor == new_head {
                Selection::Cursor(new_head)
            } else {
                Selection::range(anchor, new_head)
            }
        } else {
            Selection::Cursor(new_head)
        };
        // Important: route through update so forbidden-position snap
        // and any post-pass still applies, but DON'T clear
        // `intended_x` (dispatch() does that). Hand-roll the update
        // call here to preserve the anchor.
        let next = std::mem::take(&mut self.state);
        self.state = update::update(next, EditorEvent::SetSelection(new_sel));
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
        // Context-aware insertion (code-block: `\n`; blockquote at
        // depth D: `\n[prefix]\n[prefix]`; top-level: `\n\n`) is
        // resolved inside `update::insert_newline`. The shell stays
        // a pure router so keyboard, IME, paste, and programmatic
        // dispatch all share the same rule.
        self.dispatch(EditorEvent::InsertNewline, cx);
    }
    fn shift_enter(&mut self, _: &ShiftEnter, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::InsertLineBreak, cx);
    }
    fn tab(&mut self, _: &Tab, _: &mut Window, cx: &mut Context<Self>) {
        // Tab in a list item nests it under the previous sibling.
        // Outside of a list this is a no-op (the action just falls
        // through; hosting apps that want a literal Tab character
        // can add their own keybinding).
        self.dispatch(EditorEvent::IncreaseListDepth, cx);
    }
    fn shift_tab(&mut self, _: &ShiftTab, _: &mut Window, cx: &mut Context<Self>) {
        // Symmetric: dedent the cursor's list item by one level.
        self.dispatch(EditorEvent::DecreaseListDepth, cx);
    }

    fn left(&mut self, _: &Left, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::MoveLeft, cx);
    }
    fn right(&mut self, _: &Right, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::MoveRight, cx);
    }
    fn up(&mut self, _: &Up, _: &mut Window, cx: &mut Context<Self>) {
        self.vertical_move(-1, false, EditorEvent::MoveUp, cx);
    }
    fn down(&mut self, _: &Down, _: &mut Window, cx: &mut Context<Self>) {
        self.vertical_move(1, false, EditorEvent::MoveDown, cx);
    }
    fn shift_left(&mut self, _: &ShiftLeft, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::ExtendLeft, cx);
    }
    fn shift_right(&mut self, _: &ShiftRight, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::ExtendRight, cx);
    }
    fn shift_up(&mut self, _: &ShiftUp, _: &mut Window, cx: &mut Context<Self>) {
        self.vertical_move(-1, true, EditorEvent::ExtendUp, cx);
    }
    fn shift_down(&mut self, _: &ShiftDown, _: &mut Window, cx: &mut Context<Self>) {
        self.vertical_move(1, true, EditorEvent::ExtendDown, cx);
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

    fn word_left(&mut self, _: &WordLeft, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::MoveWordLeft, cx);
    }
    fn word_right(&mut self, _: &WordRight, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::MoveWordRight, cx);
    }
    fn shift_word_left(&mut self, _: &ShiftWordLeft, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::ExtendWordLeft, cx);
    }
    fn shift_word_right(&mut self, _: &ShiftWordRight, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::ExtendWordRight, cx);
    }
    fn delete_word_backward(
        &mut self,
        _: &DeleteWordBackward,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.dispatch(EditorEvent::DeleteWordBackward, cx);
    }
    fn delete_word_forward(
        &mut self,
        _: &DeleteWordForward,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.dispatch(EditorEvent::DeleteWordForward, cx);
    }
    fn delete_to_line_start(
        &mut self,
        _: &DeleteToLineStart,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.dispatch(EditorEvent::DeleteToLineStart, cx);
    }
    fn delete_to_line_end(&mut self, _: &DeleteToLineEnd, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::DeleteToLineEnd, cx);
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
        cx.write_to_clipboard(ClipboardItem::new_string_with_metadata(
            text,
            CLIPBOARD_SENTINEL.to_string(),
        ));
    }

    fn cut(&mut self, _: &Cut, _: &mut Window, cx: &mut Context<Self>) {
        let range = self.state.selection.selection_range();
        if range.is_empty() {
            return;
        }
        let text = self.state.markdown[range].to_string();
        cx.write_to_clipboard(ClipboardItem::new_string_with_metadata(
            text,
            CLIPBOARD_SENTINEL.to_string(),
        ));
        self.dispatch(EditorEvent::DeleteForward, cx);
    }

    fn paste(&mut self, _: &Paste, _: &mut Window, cx: &mut Context<Self>) {
        let Some(item) = cx.read_from_clipboard() else {
            return;
        };
        let internal = item.metadata().is_some_and(|m| m == CLIPBOARD_SENTINEL);
        let Some(text) = item.text() else {
            return;
        };
        // Normalize CRLF / CR line endings so downstream chain-prefix
        // injection and parser passes only have to reason about `\n`.
        // Clipboards on Windows and some Unix sources deliver CRLF;
        // legacy macOS sources sometimes deliver bare CR.
        let text = normalize_line_endings(&text);
        self.dispatch(EditorEvent::Paste { text, internal }, cx);
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

        // First pass: direct hit. If `position.y` falls in any line's
        // vertical extent, hit-test inside that line.
        //
        // Second pass: nearest line. Lines don't tile vertically — there's
        // a `paragraph_gap` between blocks — so a mouse drag whose y
        // momentarily falls in the gap would otherwise hit no line at
        // all. The previous fallback returned `markdown.len()`, making
        // the selection head shoot to end-of-doc every time the mouse
        // crossed a gap. Snap to the closest line by vertical distance,
        // then clamp the local y to that line's bounds so the x
        // coordinate still picks the right column.
        let mut best: Option<&crate::element::LaidOutLine> = None;
        let mut best_distance: Pixels = px(f32::INFINITY);
        for key in &keys {
            let block = &self.last_blocks[*key];
            for line in &block.lines {
                let line_top = line.origin.y;
                let line_bottom = line_top + line.wrapped_height;
                if position.y >= line_top && position.y < line_bottom {
                    let local = Point::new(position.x - line.origin.x, position.y - line.origin.y);
                    return line.source_offset_for_local_point(local);
                }
                let distance = if position.y < line_top {
                    line_top - position.y
                } else {
                    position.y - line_bottom
                };
                if distance < best_distance {
                    best_distance = distance;
                    best = Some(line);
                }
            }
        }

        if let Some(line) = best {
            let line_top = line.origin.y;
            let line_bottom = line_top + line.wrapped_height;
            let clamped_y = if position.y < line_top {
                px(0.0)
            } else if position.y >= line_bottom {
                line.wrapped_height - px(1.0)
            } else {
                position.y - line_top
            };
            let local = Point::new(position.x - line.origin.x, clamped_y);
            return line.source_offset_for_local_point(local);
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
            .w_full()
            .flex()
            .flex_col()
            .text_size(self.style.font_size)
            .text_color(self.style.text_color)
            .font_family(self.style.font_family.clone())
            .on_action(cx.listener(Self::backspace))
            .on_action(cx.listener(Self::delete))
            .on_action(cx.listener(Self::enter))
            .on_action(cx.listener(Self::shift_enter))
            .on_action(cx.listener(Self::tab))
            .on_action(cx.listener(Self::shift_tab))
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
            .on_action(cx.listener(Self::word_left))
            .on_action(cx.listener(Self::word_right))
            .on_action(cx.listener(Self::shift_word_left))
            .on_action(cx.listener(Self::shift_word_right))
            .on_action(cx.listener(Self::delete_word_backward))
            .on_action(cx.listener(Self::delete_word_forward))
            .on_action(cx.listener(Self::delete_to_line_start))
            .on_action(cx.listener(Self::delete_to_line_end))
            .on_action(cx.listener(Self::select_all))
            .on_action(cx.listener(Self::copy))
            .on_action(cx.listener(Self::cut))
            .on_action(cx.listener(Self::paste))
            .on_mouse_down(MouseButton::Left, cx.listener(Self::on_mouse_down))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .on_mouse_up_out(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .on_mouse_move(cx.listener(Self::on_mouse_move));

        let block_starts: Vec<usize> = spec.blocks.iter().map(|b| b.source_range.start).collect();
        // Snapshot each block's container chain *before* moving the
        // blocks into elements so we can hand each `BlockElement` the
        // chains of its immediate neighbors (used to add extra
        // breathing room at container-boundary transitions).
        let containers_per_block: Vec<Vec<crate::render_spec::Container>> =
            spec.blocks.iter().map(|b| b.containers.clone()).collect();
        for (idx, block) in spec.blocks.into_iter().enumerate() {
            let is_last = idx + 1 == block_count;
            let next_block_start = block_starts.get(idx + 1).copied();
            let prev_containers = idx
                .checked_sub(1)
                .and_then(|i| containers_per_block.get(i).cloned());
            let next_containers = containers_per_block.get(idx + 1).cloned();
            container = container.child(BlockElement::new(
                block,
                idx,
                is_last,
                next_block_start,
                prev_containers,
                next_containers,
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
