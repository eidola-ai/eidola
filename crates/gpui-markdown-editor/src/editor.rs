//! The editor's state/element split (mirrors `gpui_component::Input`):
//!
//! - [`MarkdownEditorState`] is the retained **state entity** — it owns the
//!   `EditorState`, focus, the IME `EntityInputHandler`, the cross-frame
//!   layout caches, and emits [`MarkdownEditorEvent`]s. It is not `Render`.
//! - [`MarkdownEditor`] is the per-frame **element** (`RenderOnce`) built via
//!   `MarkdownEditor::new(&state)`; it registers the action/IME handlers
//!   (routing into the state via `window.listener_for`) and paints the blocks.
//! - [`init`] installs the default keymap (scoped to the `MarkdownEditor` key
//!   context) so the editor is a drop-in — the host calls it once, like
//!   `gpui_component::init`, instead of hand-rolling bindings.
//!
//! Input flows through the pure `update::update` pipeline; text input goes
//! through `EntityInputHandler` so dead-key composition, non-Latin layouts,
//! and pasted text all share one code path.

use std::collections::HashMap;
use std::ops::Range;

use gpui::{
    Action, App, Bounds, ClipboardItem, Context, CursorStyle, Entity, EntityInputHandler,
    FocusHandle, Focusable, IntoElement, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent,
    Pixels, Point, RenderOnce, Subscription, UTF16Selection, Window, actions, div, prelude::*, px,
};
use gpui_component::Theme;

use crate::element::{BlockElement, LaidOutBlock};
use crate::event::{EditorEvent, MarkdownEditorEvent};
use crate::parser::parse;
use crate::render::render;
use crate::render_spec::RenderSpec;
use crate::state::{EditorState, Selection};
use crate::style::MarkdownStyle;
use crate::update;

/// Submit-intent action carrying the modifier state at the time Enter was
/// pressed (mirrors gpui-component's `input::Enter`). The host binds the
/// modified chords — `cmd-enter`, `cmd-shift-enter` — to this with the
/// matching `secondary`/`shift` flags; the handler emits
/// [`MarkdownEditorEvent::PressEnter`] for the modified chords and inserts a
/// newline / line break for the plain ones. `no_json` because these are
/// keystroke-bound, never invoked from a serialized keymap value.
#[derive(Action, Clone, PartialEq, Eq)]
#[action(namespace = markdown_editor, no_json)]
pub struct Enter {
    /// True when the platform primary modifier (⌘ on macOS) was held.
    pub secondary: bool,
    /// True when Shift was held.
    pub shift: bool,
}

actions!(
    markdown_editor,
    [
        Backspace,
        Delete,
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
        /// Insert clipboard content with plain-text semantics. Bytes
        /// are spliced raw — no markdown parse, no soft-break-to-space
        /// collapse, no block-boundary padding. Each `\n` becomes its
        /// own paragraph break post-splice. Default macOS keybinding:
        /// `cmd-shift-v`. The dual-keybinding convention matches what
        /// most editors do for "paste as plain text."
        PastePlain,
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

/// The editor's retained state — the Entity half of the gpui-component
/// `InputState`/`Input` split. The host owns one of these (`cx.new(...)`),
/// passes it into the [`MarkdownEditor`](crate::MarkdownEditor) element each
/// frame, mutates it through the setters (`set_value`/`clear`), and
/// `cx.subscribe`s to its [`MarkdownEditorEvent`]s. It owns focus, the IME
/// `EntityInputHandler`, and every cross-frame layout cache; it is *not*
/// `Render` — the element renders it.
pub struct MarkdownEditorState {
    pub(crate) state: EditorState,
    pub focus_handle: FocusHandle,
    /// When true, the element skips registering key/IME handlers and the
    /// `EntityInputHandler` text-mutation methods early-return, so the surface
    /// is read-only. Synced from the element's `.disabled(..)` prop each frame.
    pub(crate) disabled: bool,
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
    /// Focus/blur observers that translate gpui focus transitions into
    /// outward [`MarkdownEditorEvent::Focus`]/[`Blur`](MarkdownEditorEvent::Blur).
    /// Held so they live as long as the entity.
    _focus_subscriptions: Vec<Subscription>,
}

impl MarkdownEditorState {
    /// Create an empty editor state. Chain [`default_value`](Self::default_value)
    /// to seed initial markdown — mirrors `InputState::new(window, cx)`.
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        Self::with_state(EditorState::new(), window, cx)
    }

    pub fn with_state(state: EditorState, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let focus_handle = cx.focus_handle();
        // Translate focus transitions into outward events so a host can react
        // semantically (e.g. commit an inline edit on blur) without polling.
        let _focus_subscriptions = vec![
            cx.on_focus(&focus_handle, window, |_, _, cx| {
                cx.emit(MarkdownEditorEvent::Focus);
            }),
            cx.on_blur(&focus_handle, window, |_, _, cx| {
                cx.emit(MarkdownEditorEvent::Blur);
            }),
        ];
        Self {
            state,
            focus_handle,
            disabled: false,
            is_selecting: false,
            last_blocks: HashMap::new(),
            last_bounds: None,
            frame_input_handler_set: false,
            marked_range: None,
            code_block_scrolls: HashMap::new(),
            intended_x: None,
            _focus_subscriptions,
        }
    }

    /// Seed the initial markdown (builder form, used right after `new`).
    pub fn default_value(mut self, markdown: impl Into<String>) -> Self {
        self.state = EditorState::with_markdown(markdown);
        self
    }

    /// The current markdown source. The read half of the host seam — replaces
    /// reaching into a public `state` field.
    pub fn value(&self) -> &str {
        &self.state.markdown
    }

    /// The current selection.
    pub fn selection(&self) -> Selection {
        self.state.selection
    }

    /// True when the buffer is empty (after trimming) — the common host check
    /// for "is there anything to submit?".
    pub fn is_empty(&self) -> bool {
        self.state.markdown.trim().is_empty()
    }

    /// Replace the entire buffer and collapse the cursor to the start. The
    /// write half of the host seam; emits [`MarkdownEditorEvent::Change`].
    pub fn set_value(&mut self, markdown: impl Into<String>, cx: &mut Context<Self>) {
        self.state = EditorState::with_markdown(markdown);
        self.marked_range = None;
        self.intended_x = None;
        cx.emit(MarkdownEditorEvent::Change);
        cx.notify();
    }

    /// Clear the buffer. Convenience for `set_value("")`.
    pub fn clear(&mut self, cx: &mut Context<Self>) {
        self.set_value("", cx);
    }

    /// Drive the internal update pipeline with a raw [`EditorEvent`], the way
    /// a dispatched action or IME callback would. The crate's integration
    /// tests (a separate crate, so they can't see the `pub(crate)` `state`
    /// field) use this to script keypress sessions without synthesizing
    /// platform events; not part of the host API — hosts mutate via
    /// `set_value`/`clear` and key dispatch.
    #[doc(hidden)]
    pub fn apply_event_for_test(&mut self, event: EditorEvent, cx: &mut Context<Self>) {
        self.dispatch(event, cx);
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
        // Compare the buffer across the update so selection-only events
        // (Move*/Extend*/SetSelection) don't masquerade as content changes.
        // The composer buffer is small, so the clone is negligible.
        let before = next.markdown.clone();
        self.state = update::update(next, event);
        self.marked_range = None;
        if self.state.markdown != before {
            cx.emit(MarkdownEditorEvent::Change);
        }
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
    fn enter(&mut self, action: &Enter, _: &mut Window, cx: &mut Context<Self>) {
        if action.secondary {
            // ⌘↩ / ⌘⇧↩ — a submit-intent chord. The editor reports the chord
            // and leaves the buffer untouched; the host's subscriber decides
            // what it means (post & ask vs post-only, commit-edit, reply…).
            cx.emit(MarkdownEditorEvent::PressEnter {
                secondary: true,
                shift: action.shift,
            });
            return;
        }
        if action.shift {
            // ⇧↩ — a hard line break within the current block.
            self.dispatch(EditorEvent::InsertLineBreak, cx);
            return;
        }
        // Plain ↩ — context-aware insertion (code-block: `\n`; blockquote at
        // depth D: `\n[prefix]\n[prefix]`; top-level: `\n\n`) is resolved
        // inside `update::insert_newline`. The shell stays a pure router so
        // keyboard, IME, paste, and programmatic dispatch share the rule.
        self.dispatch(EditorEvent::InsertNewline, cx);
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

    fn paste_plain(&mut self, _: &PastePlain, _: &mut Window, cx: &mut Context<Self>) {
        // PastePlain ignores the sentinel metadata: the user explicitly
        // chose plain semantics, overriding any "this came from our own
        // editor and is canonical markdown" signal.
        let Some(text) = cx.read_from_clipboard().and_then(|item| item.text()) else {
            return;
        };
        let text = normalize_line_endings(&text);
        self.dispatch(EditorEvent::PastePlain { text }, cx);
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

impl gpui::EventEmitter<MarkdownEditorEvent> for MarkdownEditorState {}

impl EntityInputHandler for MarkdownEditorState {
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
        if self.disabled {
            return;
        }
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
        if self.disabled {
            return;
        }
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
        cx.emit(MarkdownEditorEvent::Change);
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

impl Focusable for MarkdownEditorState {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

/// The editor's render half — the ephemeral element built each frame from a
/// `MarkdownEditorState`, mirroring gpui-component's `Input::new(&state)`.
/// Carries the per-render props (style overrides, disabled); holds no state
/// of its own. Construct it in the host's `render` and drop it into the tree.
#[derive(IntoElement)]
pub struct MarkdownEditor {
    state: Entity<MarkdownEditorState>,
    style: Option<MarkdownStyle>,
    disabled: bool,
}

impl MarkdownEditor {
    /// Build the element over a host-owned state entity.
    pub fn new(state: &Entity<MarkdownEditorState>) -> Self {
        Self {
            state: state.clone(),
            style: None,
            disabled: false,
        }
    }

    /// Typographic / color overrides for this frame. Theme-derived color
    /// fields are refreshed on top of these in `render`, so a caller only
    /// needs to set the knobs it cares about (font size, leading, heading
    /// scale, …) — see `MarkdownStyle::from_theme`.
    pub fn style(mut self, style: MarkdownStyle) -> Self {
        self.style = Some(style);
        self
    }

    /// Render read-only: no key/IME handlers are registered and the
    /// `EntityInputHandler` text mutations early-return, so the surface
    /// displays but rejects input. Mirrors `Input::disabled`.
    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }
}

impl RenderOnce for MarkdownEditor {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        // Per-frame reset + sync the disabled prop onto the state (so the
        // IME handler can honor it). Block elements re-populate `last_blocks`
        // during paint; `frame_input_handler_set` re-arms IME registration.
        self.state.update(cx, |st, _| {
            st.disabled = self.disabled;
            st.last_blocks.clear();
            st.frame_input_handler_set = false;
            st.last_bounds = None;
        });

        // Final style = caller overrides (or the theme default) with the
        // theme-derived color fields refreshed each frame, so a Circadian
        // light/dark flip recolors live even if the caller built its style
        // once. This is the old entity-render refresh, moved to the element.
        let mut style = self
            .style
            .clone()
            .unwrap_or_else(|| MarkdownStyle::from_theme(cx));
        let theme = Theme::global(cx);
        style.text_color = theme.foreground;
        style.delimiter_color = theme.muted_foreground;
        style.background = theme.background;
        style.caret_color = theme.caret;
        style.selection_color = theme.selection;
        style.link_color = theme.link;
        style.blockquote_border_color = theme.border;
        style.thematic_break_color = theme.border;
        style.inline_code_background = theme.accent;
        style.code_block_background = theme.muted;
        style.code_block_content_background = crate::style::shift_lightness(theme.muted, -0.04);

        let spec = self.state.read(cx).render_spec();
        let state = self.state.clone();
        let disabled = self.disabled;

        let mut container = div()
            .id("markdown-editor")
            .key_context("MarkdownEditor")
            .track_focus(&state.read(cx).focus_handle)
            .when(!disabled, |c| c.cursor(CursorStyle::IBeam))
            .w_full()
            .flex()
            .flex_col()
            .text_size(style.font_size)
            .text_color(style.text_color)
            .font_family(style.font_family.clone());

        // Key/IME handlers are registered only when editable. Each routes
        // the action into the *state* entity via `window.listener_for`
        // (the element→state bridge), the gpui-component idiom.
        if !disabled {
            container = container
                .on_action(window.listener_for(&state, MarkdownEditorState::backspace))
                .on_action(window.listener_for(&state, MarkdownEditorState::delete))
                .on_action(window.listener_for(&state, MarkdownEditorState::enter))
                .on_action(window.listener_for(&state, MarkdownEditorState::tab))
                .on_action(window.listener_for(&state, MarkdownEditorState::shift_tab))
                .on_action(window.listener_for(&state, MarkdownEditorState::left))
                .on_action(window.listener_for(&state, MarkdownEditorState::right))
                .on_action(window.listener_for(&state, MarkdownEditorState::up))
                .on_action(window.listener_for(&state, MarkdownEditorState::down))
                .on_action(window.listener_for(&state, MarkdownEditorState::shift_left))
                .on_action(window.listener_for(&state, MarkdownEditorState::shift_right))
                .on_action(window.listener_for(&state, MarkdownEditorState::shift_up))
                .on_action(window.listener_for(&state, MarkdownEditorState::shift_down))
                .on_action(window.listener_for(&state, MarkdownEditorState::home))
                .on_action(window.listener_for(&state, MarkdownEditorState::end))
                .on_action(window.listener_for(&state, MarkdownEditorState::shift_home))
                .on_action(window.listener_for(&state, MarkdownEditorState::shift_end))
                .on_action(window.listener_for(&state, MarkdownEditorState::document_start))
                .on_action(window.listener_for(&state, MarkdownEditorState::document_end))
                .on_action(window.listener_for(&state, MarkdownEditorState::shift_document_start))
                .on_action(window.listener_for(&state, MarkdownEditorState::shift_document_end))
                .on_action(window.listener_for(&state, MarkdownEditorState::word_left))
                .on_action(window.listener_for(&state, MarkdownEditorState::word_right))
                .on_action(window.listener_for(&state, MarkdownEditorState::shift_word_left))
                .on_action(window.listener_for(&state, MarkdownEditorState::shift_word_right))
                .on_action(window.listener_for(&state, MarkdownEditorState::delete_word_backward))
                .on_action(window.listener_for(&state, MarkdownEditorState::delete_word_forward))
                .on_action(window.listener_for(&state, MarkdownEditorState::delete_to_line_start))
                .on_action(window.listener_for(&state, MarkdownEditorState::delete_to_line_end))
                .on_action(window.listener_for(&state, MarkdownEditorState::select_all))
                .on_action(window.listener_for(&state, MarkdownEditorState::copy))
                .on_action(window.listener_for(&state, MarkdownEditorState::cut))
                .on_action(window.listener_for(&state, MarkdownEditorState::paste))
                .on_action(window.listener_for(&state, MarkdownEditorState::paste_plain))
                // Map the Edit-menu action types (`gpui_component::input::*`)
                // onto the editor's own implementations. The OS routes the Edit
                // menu through the responder chain via the `OsAction::*`
                // selectors; those land as `gpui_component::input::{Cut,Copy,
                // Paste,SelectAll}` dispatched to the focused element.
                .on_action(
                    window.listener_for(&state, |this, _: &gpui_component::input::Cut, w, cx| {
                        this.cut(&Cut, w, cx)
                    }),
                )
                .on_action(
                    window.listener_for(&state, |this, _: &gpui_component::input::Copy, w, cx| {
                        this.copy(&Copy, w, cx)
                    }),
                )
                .on_action(
                    window.listener_for(&state, |this, _: &gpui_component::input::Paste, w, cx| {
                        this.paste(&Paste, w, cx)
                    }),
                )
                .on_action(window.listener_for(
                    &state,
                    |this, _: &gpui_component::input::SelectAll, w, cx| {
                        this.select_all(&SelectAll, w, cx)
                    },
                ))
                .on_mouse_down(
                    MouseButton::Left,
                    window.listener_for(&state, MarkdownEditorState::on_mouse_down),
                )
                .on_mouse_up(
                    MouseButton::Left,
                    window.listener_for(&state, MarkdownEditorState::on_mouse_up),
                )
                .on_mouse_up_out(
                    MouseButton::Left,
                    window.listener_for(&state, MarkdownEditorState::on_mouse_up),
                )
                .on_mouse_move(window.listener_for(&state, MarkdownEditorState::on_mouse_move));
        }

        let spec_blocks = spec.blocks;
        let block_count = spec_blocks.len();
        let block_starts: Vec<usize> = spec_blocks.iter().map(|b| b.source_range.start).collect();
        // Snapshot each block's container chain *before* moving the blocks
        // into elements so we can hand each `BlockElement` the chains of its
        // immediate neighbors (used to add breathing room at container
        // boundaries).
        let containers_per_block: Vec<Vec<crate::render_spec::Container>> =
            spec_blocks.iter().map(|b| b.containers.clone()).collect();
        for (idx, block) in spec_blocks.into_iter().enumerate() {
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
                state.clone(),
                style.clone(),
            ));
        }

        // The first `BlockElement::paint` of the frame registers the
        // `EntityInputHandler` (IME / typed text → `replace_text_in_range`),
        // unless `disabled`.
        container
    }
}

/// Install the editor's default macOS keymap, scoped to the `MarkdownEditor`
/// key context. Self-contained so the editor is a drop-in like
/// `gpui_component::Input` (whose keymap `gpui_component::init` installs) —
/// the host calls this once at startup instead of hand-rolling the bindings.
///
/// The submit chords (`cmd-enter`, `cmd-shift-enter`) bind the `Enter` action
/// with `secondary: true`; the handler emits
/// [`MarkdownEditorEvent::PressEnter`] rather than inserting, so the host
/// subscribes for submit instead of binding the chords itself.
pub fn init(cx: &mut App) {
    let ctx = Some("MarkdownEditor");
    cx.bind_keys([
        // Enter chords — plain/shift insert; cmd-variants emit PressEnter.
        gpui::KeyBinding::new(
            "enter",
            Enter {
                secondary: false,
                shift: false,
            },
            ctx,
        ),
        gpui::KeyBinding::new(
            "shift-enter",
            Enter {
                secondary: false,
                shift: true,
            },
            ctx,
        ),
        gpui::KeyBinding::new(
            "cmd-enter",
            Enter {
                secondary: true,
                shift: false,
            },
            ctx,
        ),
        gpui::KeyBinding::new(
            "cmd-shift-enter",
            Enter {
                secondary: true,
                shift: true,
            },
            ctx,
        ),
        // Editing
        gpui::KeyBinding::new("backspace", Backspace, ctx),
        gpui::KeyBinding::new("delete", Delete, ctx),
        gpui::KeyBinding::new("tab", Tab, ctx),
        gpui::KeyBinding::new("shift-tab", ShiftTab, ctx),
        // Word / line delete (macOS standard: Option for word, Cmd for line).
        gpui::KeyBinding::new("alt-backspace", DeleteWordBackward, ctx),
        gpui::KeyBinding::new("alt-delete", DeleteWordForward, ctx),
        gpui::KeyBinding::new("cmd-backspace", DeleteToLineStart, ctx),
        gpui::KeyBinding::new("cmd-delete", DeleteToLineEnd, ctx),
        // Caret motion
        gpui::KeyBinding::new("left", Left, ctx),
        gpui::KeyBinding::new("right", Right, ctx),
        gpui::KeyBinding::new("up", Up, ctx),
        gpui::KeyBinding::new("down", Down, ctx),
        gpui::KeyBinding::new("shift-left", ShiftLeft, ctx),
        gpui::KeyBinding::new("shift-right", ShiftRight, ctx),
        gpui::KeyBinding::new("shift-up", ShiftUp, ctx),
        gpui::KeyBinding::new("shift-down", ShiftDown, ctx),
        gpui::KeyBinding::new("home", Home, ctx),
        gpui::KeyBinding::new("end", End, ctx),
        gpui::KeyBinding::new("cmd-left", Home, ctx),
        gpui::KeyBinding::new("cmd-right", End, ctx),
        gpui::KeyBinding::new("shift-home", ShiftHome, ctx),
        gpui::KeyBinding::new("shift-end", ShiftEnd, ctx),
        gpui::KeyBinding::new("cmd-shift-left", ShiftHome, ctx),
        gpui::KeyBinding::new("cmd-shift-right", ShiftEnd, ctx),
        gpui::KeyBinding::new("cmd-up", DocumentStart, ctx),
        gpui::KeyBinding::new("cmd-down", DocumentEnd, ctx),
        gpui::KeyBinding::new("cmd-shift-up", ShiftDocumentStart, ctx),
        gpui::KeyBinding::new("cmd-shift-down", ShiftDocumentEnd, ctx),
        // Word-granular motion (macOS standard: Option+arrows).
        gpui::KeyBinding::new("alt-left", WordLeft, ctx),
        gpui::KeyBinding::new("alt-right", WordRight, ctx),
        gpui::KeyBinding::new("alt-shift-left", ShiftWordLeft, ctx),
        gpui::KeyBinding::new("alt-shift-right", ShiftWordRight, ctx),
        // Clipboard — scoped to the editor context so they coexist with
        // `gpui_component::Input`'s own `Input`-context bindings.
        gpui::KeyBinding::new("cmd-a", SelectAll, ctx),
        gpui::KeyBinding::new("cmd-c", Copy, ctx),
        gpui::KeyBinding::new("cmd-x", Cut, ctx),
        gpui::KeyBinding::new("cmd-v", Paste, ctx),
        gpui::KeyBinding::new("cmd-shift-v", PastePlain, ctx),
    ]);
}
