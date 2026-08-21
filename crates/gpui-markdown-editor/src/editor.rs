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
use std::rc::Rc;

use gpui::{
    Action, App, Bounds, ClipboardItem, Context, CursorStyle, Entity, EntityInputHandler,
    FocusHandle, Focusable, IntoElement, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent,
    Pixels, Point, RenderOnce, Subscription, UTF16Selection, Window, actions, div, prelude::*, px,
};
use gpui_component::Theme;

use crate::element::{BlockElement, LaidOutBlock};
use crate::event::{EditorEvent, MarkdownEditorEvent};
use crate::formatting::InlineFormat;
use crate::parser::parse;
use crate::render::{render, render_readonly};
use crate::render_spec::RenderSpec;
use crate::state::{EditorState, Selection};
use crate::style::MarkdownStyle;
use crate::update;

/// Host callback for a click on a rendered embed block: `(ordinal, window,
/// app)`. See [`crate::embed`] for the plugin contract; the ordinal is the
/// shared key into the host's own bookkeeping.
pub type EmbedClickHandler = Rc<dyn Fn(u64, &mut Window, &mut App)>;

/// Host callback for a click on host-supplied highlighted text (see
/// [`crate::highlight`]): receives the keys of every highlight range
/// containing the clicked offset. The editor never interprets the keys.
pub type HighlightClickHandler = Rc<dyn Fn(&[u64], &mut Window, &mut App)>;

/// Host callback for a **context-menu gesture** (right mouse-down) inside the
/// editor: receives the window-coordinate position the menu should open at.
/// The editor opens no menu of its own — it has no opinion about what belongs
/// in one — it only reports the gesture, having first placed the caret (see
/// `MarkdownEditorState::on_right_mouse_down`).
pub type ContextMenuHandler = Rc<dyn Fn(&Point<Pixels>, &mut Window, &mut App)>;

/// A clipboard/selection command a host can run **programmatically** — the
/// same operations the keymap's `Cut`/`Copy`/`Paste`/`SelectAll` actions
/// perform, reachable without a keystroke and without the focus/responder
/// chain. This is what lets a host drive its own context menu over an editor
/// that isn't focused (a read-only post body).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EditorCommand {
    Cut,
    Copy,
    Paste,
    SelectAll,
}

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
        /// Toggle bold over the semantic selection or the word under the caret.
        ToggleBold,
        /// Toggle italic over the semantic selection or the word under the caret.
        ToggleItalic,
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
        /// Revert the buffer to the state before the most recent
        /// (coalesced) edit, restoring its markdown *and* selection.
        /// No-op when the undo stack is empty. Default macOS
        /// keybinding: `cmd-z`.
        Undo,
        /// Re-apply the most recently undone edit. Cleared the moment a
        /// fresh edit is made. No-op when the redo stack is empty.
        /// Default macOS keybinding: `cmd-shift-z`.
        Redo,
    ]
);

/// Cap on the number of undo entries retained. Each entry is a full
/// pre-edit [`EditorState`] snapshot (markdown + selection); the buffers
/// this editor hosts (chat composer, inline posts) are small, so a few
/// hundred whole-document snapshots is a negligible memory cost while
/// still bounding a pathological session. The oldest entry is dropped
/// (FIFO) once the stack exceeds this.
const MAX_HISTORY: usize = 256;

/// Which wrapped display row a caret sitting EXACTLY on a soft-wrap boundary
/// belongs to. The same byte offset is both the end of one wrapped row and the
/// start of the next; gpui's `position_for_index` always resolves it to the
/// upper row's end. This transient signal records the caret's intended row so
/// it renders — and vertical motion computes its current row — on the right
/// visual row. Like `intended_x`, it's set by the command that placed the caret
/// and reset on edits / plain moves. Only matters at a boundary; harmless
/// elsewhere.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WrapAffinity {
    /// Prefer the START of the lower row (Down/Home/Right onto a boundary, a
    /// fresh caret, an edit). The default — the intuitive "beginning of the
    /// next line".
    Downstream,
    /// Prefer the END of the upper row (End). Keeps the caret visually on the
    /// row the command acted on.
    Upstream,
}

/// The granularity a pointer selection extends by, set at the mouse-down that
/// began the drag from its click count (native macOS text-view idiom):
/// single-click selects by character, double-click by word, triple-click by
/// line. A drag started at a higher granularity keeps that granularity — so
/// dragging after a double-click extends the selection word-by-word, and after
/// a triple-click line-by-line, anchored on the originally-clicked unit.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SelectMode {
    /// Single click — the caret/character granularity (the default).
    Char,
    /// Double click — whole words (whitespace and punctuation delimit).
    Word,
    /// Triple click — whole `\n`-delimited lines / paragraphs.
    Line,
}

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

/// The source range of the "word" containing `offset`, for double-click
/// selection. Words are Unicode word-boundary segments (the same segmentation
/// the word-motion commands use): a maximal run of word characters selects the
/// run, while a click on whitespace or punctuation selects that delimiter
/// segment — matching native macOS text-view double-click, where whitespace and
/// punctuation delimit words. A click exactly on a boundary between two
/// segments takes the segment starting there (the one to the right), except at
/// end-of-buffer where it takes the last segment. An empty buffer yields `0..0`.
fn word_range_at(text: &str, offset: usize) -> Range<usize> {
    use unicode_segmentation::UnicodeSegmentation;
    let offset = offset.min(text.len());
    let mut last: Option<Range<usize>> = None;
    for (idx, seg) in text.split_word_bound_indices() {
        let range = idx..idx + seg.len();
        // A segment strictly containing the offset (or starting at it) is the
        // hit — start-inclusive, so a boundary click takes the right-hand word.
        if range.start <= offset && offset < range.end {
            return range;
        }
        last = Some(range);
    }
    // Offset at end-of-buffer: take the final segment so a double-click at the
    // very end still selects the trailing word.
    last.unwrap_or(offset..offset)
}

/// The source range of the `\n`-delimited line containing `offset`, for
/// triple-click selection (the whole paragraph on a hard-line-broken source).
/// The trailing newline is *not* included, so the selection covers the line's
/// text without swallowing the break into the next line.
fn line_range_at(text: &str, offset: usize) -> Range<usize> {
    let bytes = text.as_bytes();
    let offset = offset.min(text.len());
    let mut start = offset;
    while start > 0 && bytes[start - 1] != b'\n' {
        start -= 1;
    }
    let mut end = offset;
    while end < bytes.len() && bytes[end] != b'\n' {
        end += 1;
    }
    start..end
}

/// Decide whether the transition `before → after` is a *coalescible
/// typing insert*: a single non-whitespace character spliced at a
/// collapsed caret, leaving the caret immediately after it. Returns the
/// post-insert caret offset when so (the anchor a following character
/// must continue from to keep coalescing), else `None`.
///
/// Requiring a clean single-char insertion at a cursor is deliberately
/// conservative: if `update` did anything structural (promoted a soft
/// break, renumbered a list, closed a fence) the shape won't match and
/// the edit falls out of the run — landing as its own, independently
/// undoable step. Whitespace and newlines are excluded so a word break
/// or Enter ends the current run, matching how editors segment undo.
fn coalescible_type_end(before: &EditorState, after: &EditorState) -> Option<usize> {
    let Selection::Cursor(start) = before.selection else {
        return None;
    };
    let Selection::Cursor(end) = after.selection else {
        return None;
    };
    if end <= start || after.markdown.len() != before.markdown.len() + (end - start) {
        return None;
    }
    let bb = before.markdown.as_bytes();
    let ab = after.markdown.as_bytes();
    // after = before[..start] + inserted + before[start..]
    if ab[..start] != bb[..start] || ab[end..] != bb[start..] {
        return None;
    }
    let inserted = &after.markdown[start..end];
    let mut chars = inserted.chars();
    let ch = chars.next()?;
    if chars.next().is_some() || ch.is_whitespace() {
        return None;
    }
    Some(end)
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
    /// When true, the element skips registering mutating key/IME handlers, the
    /// `EntityInputHandler` text-mutation methods early-return, and the caret is
    /// not painted — so the surface is read-only. Selection, navigation, copy,
    /// and select-all handlers are still registered, so the text can be selected
    /// and copied. Synced from the element's `.disabled(..)` prop each frame.
    pub(crate) disabled: bool,
    /// Host callback invoked when the user clicks a rendered embed block
    /// (see [`crate::embed`]) — the wave-2 click-to-navigate seam. Synced
    /// from the element's `.on_embed_click(..)` prop each frame; the editor
    /// only reports the ordinal, never interprets it.
    pub(crate) on_embed_click: Option<EmbedClickHandler>,
    /// Host-supplied highlight ranges, one set per layer (see
    /// [`crate::highlight`]) — inert decorations painted as a quiet wash
    /// behind the covered text. Entity state (not [`EditorState`]): the pure
    /// update/render pipeline never consults them, only the element's paint
    /// (and the click hit-test, which reads
    /// [`crate::highlight::HighlightLayer::Base`] alone) do.
    pub(crate) highlights: crate::highlight::HighlightLayers,
    /// Host callback for a plain click on highlighted text. Synced from the
    /// element's `.on_highlight_click(..)` prop each frame.
    pub(crate) on_highlight_click: Option<HighlightClickHandler>,
    /// Host callback for a right mouse-down (the context-menu gesture).
    /// Synced from the element's `.on_context_menu(..)` prop each frame.
    pub(crate) on_context_menu: Option<ContextMenuHandler>,
    /// The source offset of an in-progress press that started on highlighted
    /// text (single click, no shift). Consumed on mouse-up: if the press
    /// resolved as a plain click (the selection is still collapsed — no drag
    /// created a range), the highlight click callback fires. A drag across a
    /// highlight therefore selects normally and never navigates.
    highlight_press: Option<usize>,
    is_selecting: bool,
    /// Granularity the active pointer drag extends by (set from the mouse-down
    /// click count — see [`SelectMode`]).
    select_mode: SelectMode,
    /// The source range of the unit (word / line) the current drag is anchored
    /// on — the double/triple-click selection the drag extends *from*, so a
    /// word/line drag grows symmetrically around the originally-clicked unit
    /// rather than from a bare offset. Ignored in [`SelectMode::Char`].
    select_anchor_range: Range<usize>,
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
    /// Transient caret wrap-boundary affinity — see [`WrapAffinity`]. Set by the
    /// command that placed the caret (Down/Home → `Downstream`, End →
    /// `Upstream`, most others reset to `Downstream`) and read by the caret
    /// renderer and vertical/line-bound motion so a caret exactly on a soft-wrap
    /// boundary resolves to the intended visual row.
    wrap_affinity: WrapAffinity,
    /// Undo history — a stack of pre-edit [`EditorState`] snapshots,
    /// oldest at the front. Every buffer-mutating edit pushes the state
    /// *before* the edit (unless it coalesces into the previous typing
    /// run — see [`coalesce_anchor`](Self::coalesce_anchor)). `undo`
    /// pops the top back into `state`. Capped at [`MAX_HISTORY`].
    undo_stack: Vec<EditorState>,
    /// Redo history — states popped off `state` by `undo`, newest on
    /// top. `redo` pops the top back into `state`. Cleared on any fresh
    /// edit (the standard branch-discard behavior).
    redo_stack: Vec<EditorState>,
    /// Typing-coalescing anchor. `Some(offset)` when the previous
    /// recorded edit was a single-character, non-whitespace insertion
    /// whose caret ended at `offset`. A new single-character insert
    /// whose *starting* caret sits exactly at this anchor extends the
    /// same undo step instead of pushing a new one — so a run of typed
    /// characters undoes as one unit. Any edit that isn't a contiguous
    /// single-char insert (deletion, whitespace/newline, paste,
    /// structural edit, a selection jump, or undo/redo itself) resets
    /// the anchor to `None`, breaking the run.
    coalesce_anchor: Option<usize>,
    /// Session-scoped table-breakage recoverability state — see
    /// [`update::TableGuard`]. Threaded into every editable dispatch
    /// so a table broken by one event keeps its line structure across
    /// the following events while the user repairs it in place.
    table_guard: update::TableGuard,
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
            on_embed_click: None,
            highlights: crate::highlight::HighlightLayers::default(),
            on_highlight_click: None,
            on_context_menu: None,
            highlight_press: None,
            is_selecting: false,
            select_mode: SelectMode::Char,
            select_anchor_range: 0..0,
            last_blocks: HashMap::new(),
            last_bounds: None,
            frame_input_handler_set: false,
            marked_range: None,
            code_block_scrolls: HashMap::new(),
            intended_x: None,
            wrap_affinity: WrapAffinity::Downstream,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            coalesce_anchor: None,
            table_guard: update::TableGuard::default(),
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

    /// Whether a pointer selection drag is currently in progress (mouse held
    /// down after a press on the editor). A host that owns the scroll — the
    /// editor has no internal vertical scroll — reads this to drive
    /// autoscroll-while-selecting: while any editor reports `true`, scroll the
    /// page when the pointer nears a viewport edge so a drag can pull off-screen
    /// content into the selection.
    pub fn is_selecting(&self) -> bool {
        self.is_selecting
    }

    /// Extend the in-progress selection so its head reaches the row under
    /// `window_pos` (window coordinates), honoring the drag's granularity. A
    /// no-op unless a drag is active. The host calls this while autoscrolling a
    /// selection drag: the pointer sits still against a viewport edge while the
    /// page scrolls under it, so no mouse-move fires — the host re-extends each
    /// frame from the (unchanged) pointer position against the freshly-scrolled
    /// geometry.
    pub fn drag_extend_to(&mut self, window_pos: Point<Pixels>, cx: &mut Context<Self>) {
        if !self.is_selecting {
            return;
        }
        let offset = self.offset_for_position(window_pos);
        self.extend_selection_to(offset, cx);
    }

    /// Test seam: every laid-out line's source range + geometry, flat
    /// across blocks — `(src_start, src_end, x, y, width, wrapped_height)`.
    /// Lets integration tests locate a table cell's box (source range →
    /// pixel rectangle) to aim real clicks and assert wrapped-cell
    /// navigation geometry.
    #[doc(hidden)]
    pub fn debug_line_source_geometry(&self) -> Vec<(usize, usize, f32, f32, f32, f32)> {
        let mut keys: Vec<usize> = self.last_blocks.keys().copied().collect();
        keys.sort();
        let mut out = Vec::new();
        for k in keys {
            for l in &self.last_blocks[&k].lines {
                out.push((
                    l.source_range.start,
                    l.source_range.end,
                    l.origin.x.as_f32(),
                    l.origin.y.as_f32(),
                    l.line.width().as_f32(),
                    l.wrapped_height.as_f32(),
                ));
            }
        }
        out
    }

    /// Test seam: what the embed for `ordinal` actually painted last frame —
    /// every text piece as `(text, x, y)` in layout order, then the counts of
    /// its non-text chrome. The marker glyphs of an embedded list (`• `,
    /// `1. `, `☑ `) are pieces here and appear in no shaped line, so this is
    /// the only way to assert them without eyeballing a snapshot.
    ///
    /// Returns `None` if that ordinal isn't a painted embed block.
    #[doc(hidden)]
    #[allow(clippy::type_complexity)]
    pub fn debug_embed_content(
        &self,
        ordinal: u64,
    ) -> Option<(Vec<(String, f32, f32)>, EmbedChromeCounts)> {
        let block = self
            .last_blocks
            .values()
            .find(|b| b.embed_ordinal == Some(ordinal))?;
        let content = block.embed_content.as_ref()?;
        let pieces = content
            .pieces
            .iter()
            .map(|(text, origin)| (text.to_string(), origin.x.as_f32(), origin.y.as_f32()))
            .collect();
        Some((
            pieces,
            EmbedChromeCounts {
                code_panels: content.code_panels.len(),
                math: content.math_origins.len(),
                rules: content.rules.len(),
                bars: content.bars.len(),
            },
        ))
    }

    /// Diagnostic seam: the recorded (window-coordinate) line geometry from
    /// the last paint — `(block_index, [(origin_x, origin_y, wrapped_height)])`
    /// per block, sorted by block index.
    #[doc(hidden)]
    #[allow(clippy::type_complexity)]
    pub fn debug_line_geometry(&self) -> Vec<(usize, Vec<(f32, f32, f32)>)> {
        let mut keys: Vec<usize> = self.last_blocks.keys().copied().collect();
        keys.sort();
        keys.into_iter()
            .map(|k| {
                let block = &self.last_blocks[&k];
                (
                    k,
                    block
                        .lines
                        .iter()
                        .map(|l| {
                            (
                                l.origin.x.as_f32(),
                                l.origin.y.as_f32(),
                                l.wrapped_height.as_f32(),
                            )
                        })
                        .collect(),
                )
            })
            .collect()
    }

    /// True when the buffer is empty (after trimming) — the common host check
    /// for "is there anything to submit?".
    pub fn is_empty(&self) -> bool {
        self.state.markdown.trim().is_empty()
    }

    /// Replace the entire buffer and collapse the cursor to the start. The
    /// write half of the host seam; emits [`MarkdownEditorEvent::Change`].
    ///
    /// **Undo boundary.** A programmatic host replacement of the whole
    /// document (loading a post to edit, clearing after submit, seeding a
    /// draft) is *not* a user-undoable keystroke, so this clears both
    /// history stacks: undo never crosses a `set_value`, and the user
    /// can't `cmd-z` their way back into a document the host swapped out
    /// from under them. Typed edits after the swap build a fresh history.
    pub fn set_value(&mut self, markdown: impl Into<String>, cx: &mut Context<Self>) {
        // The embed map is render-time state supplied by the host, not
        // document content — a buffer swap keeps it.
        let embeds = std::mem::take(&mut self.state.embeds);
        self.state = EditorState::with_markdown(markdown);
        self.state.embeds = embeds;
        self.marked_range = None;
        self.intended_x = None;
        self.wrap_affinity = WrapAffinity::Downstream;
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.coalesce_anchor = None;
        cx.emit(MarkdownEditorEvent::Change);
        cx.notify();
    }

    /// Clear the buffer. Convenience for `set_value("")`.
    pub fn clear(&mut self, cx: &mut Context<Self>) {
        self.set_value("", cx);
    }

    /// Supply the embed map — ordinals → markdown content (see
    /// [`crate::embed`] for the plugin contract and lexical rules). A buffer
    /// marker `{{ embed N }}` renders as an atomic embed block exactly when
    /// `N` is mapped; unmapped markers stay literal text. The map is
    /// **render-time state**, not document content: the buffer is untouched,
    /// so no [`MarkdownEditorEvent::Change`] is emitted — the view just
    /// re-renders.
    pub fn set_embeds(
        &mut self,
        entries: impl IntoIterator<Item = (u64, String)>,
        cx: &mut Context<Self>,
    ) {
        self.state.embeds = crate::embed::EmbedMap::new(entries);
        // Installing a map can turn a position the caret legally occupied (a
        // literal, unmapped marker's interior) into a forbidden embed
        // interior. Re-snap the selection against the new map immediately —
        // otherwise the next insertion would splice into the hidden marker
        // bytes of a block that renders as an embed.
        let md = &self.state.markdown;
        let em = &self.state.embeds;
        self.state.selection = match self.state.selection {
            Selection::Cursor(p) => {
                Selection::Cursor(crate::analysis::nearest_allowed_position_with(md, em, p))
            }
            Selection::Range { anchor, head } => {
                let a = crate::analysis::nearest_allowed_position_with(md, em, anchor);
                let h = crate::analysis::nearest_allowed_position_with(md, em, head);
                if a == h {
                    Selection::Cursor(h)
                } else {
                    Selection::Range { anchor: a, head: h }
                }
            }
        };
        cx.notify();
    }

    /// The current embed map.
    pub fn embeds(&self) -> &crate::embed::EmbedMap {
        &self.state.embeds
    }

    /// Supply the highlight ranges on [`crate::highlight::HighlightLayer::Base`]
    /// — the layer whose ranges route clicks. Shorthand for
    /// [`Self::set_highlights_in`].
    pub fn set_highlights(
        &mut self,
        entries: impl IntoIterator<Item = (std::ops::Range<usize>, u64)>,
        cx: &mut Context<Self>,
    ) {
        self.set_highlights_in(crate::highlight::HighlightLayer::Base, entries, cx);
    }

    /// Supply the highlight ranges on one layer — `(source-byte range, opaque
    /// key)` pairs (see [`crate::highlight`] for the plugin contract). Every
    /// other layer is left as it was. Render-time decoration only: the buffer
    /// is untouched (no [`MarkdownEditorEvent::Change`]), no caret position
    /// becomes forbidden, and editing/selection are unaffected — the view just
    /// re-renders with the wash. Overlapping ranges merge visually at paint
    /// time, **within the layer**.
    pub fn set_highlights_in(
        &mut self,
        layer: crate::highlight::HighlightLayer,
        entries: impl IntoIterator<Item = (std::ops::Range<usize>, u64)>,
        cx: &mut Context<Self>,
    ) {
        self.highlights
            .set(layer, crate::highlight::HighlightSet::new(entries));
        cx.notify();
    }

    /// The current highlight set on [`crate::highlight::HighlightLayer::Base`].
    pub fn highlights(&self) -> &crate::highlight::HighlightSet {
        self.highlights_in(crate::highlight::HighlightLayer::Base)
    }

    /// The current highlight set on one layer — the read half of the
    /// compare-before-set guard a host that recomputes ranges every frame
    /// needs (setting notifies unconditionally).
    pub fn highlights_in(
        &self,
        layer: crate::highlight::HighlightLayer,
    ) -> &crate::highlight::HighlightSet {
        self.highlights.get(layer)
    }

    /// Every layer's highlight ranges — what the paint pass walks.
    pub(crate) fn highlight_layers(&self) -> &crate::highlight::HighlightLayers {
        &self.highlights
    }

    /// Place the caret at the end of the buffer and insert `text` there (host
    /// API). The seam for "the user started typing at something that isn't
    /// this editor, and the host decided the keystrokes belong here" — the
    /// space view's type-to-compose jump. Routed through the normal update
    /// pipeline ([`EditorEvent::InsertText`]), so it records one undo step and
    /// emits [`MarkdownEditorEvent::Change`] like any typed insertion; the
    /// preceding `SetSelection` collapses any selection rather than replacing
    /// it, because the host's intent is *append*, not *replace*. A no-op on a
    /// disabled (read-only) editor, whose buffer the host owns.
    pub fn append_at_end(&mut self, text: impl Into<String>, cx: &mut Context<Self>) {
        if self.disabled {
            return;
        }
        let end = self.state.markdown.len();
        self.dispatch(EditorEvent::SetSelection(Selection::Cursor(end)), cx);
        self.dispatch(EditorEvent::InsertText(text.into()), cx);
    }

    /// Insert the canonical `{{ embed N }}` marker as its **own top-level
    /// paragraph** at the caret (host API — the quote-creation seam). The
    /// marker only renders as an embed block when it stands as a
    /// blank-line-delimited paragraph, so the insertion pads with the blank
    /// lines the surrounding bytes are missing. Routed through the normal
    /// update pipeline ([`EditorEvent::InsertText`]), so it replaces any
    /// active selection, records one undo step, and emits
    /// [`MarkdownEditorEvent::Change`]. (Inside a verbatim region the marker
    /// lands as literal text — the documented honest degradation.)
    pub fn insert_embed_marker(&mut self, ordinal: u64, cx: &mut Context<Self>) {
        let md = &self.state.markdown;
        let mut sel = self.state.selection.selection_range();
        sel.start = sel.start.min(md.len());
        sel.end = sel.end.min(md.len());
        // Swallow spaces/tabs adjacent to the splice on any side that gets
        // structural padding, so the neighboring paragraphs come out clean
        // (no trailing-space paragraph before the marker, no leading-space
        // one after it).
        let pad_before = {
            let before = &md[..sel.start];
            let trimmed = before.trim_end_matches([' ', '\t']);
            if trimmed.is_empty() || trimmed.ends_with("\n\n") {
                None
            } else if trimmed.ends_with('\n') {
                Some("\n")
            } else {
                Some("\n\n")
            }
        };
        let pad_after = {
            let after = &md[sel.end..];
            let trimmed = after.trim_start_matches([' ', '\t']);
            if trimmed.is_empty() || trimmed.starts_with("\n\n") {
                None
            } else if trimmed.starts_with('\n') {
                Some("\n")
            } else {
                Some("\n\n")
            }
        };
        let ws = |b: u8| b == b' ' || b == b'\t';
        while sel.start > 0 && ws(md.as_bytes()[sel.start - 1]) {
            sel.start -= 1;
        }
        while sel.end < md.len() && ws(md.as_bytes()[sel.end]) {
            sel.end += 1;
        }
        self.state.selection = if sel.start == sel.end {
            Selection::Cursor(sel.start)
        } else {
            Selection::range(sel.start, sel.end)
        };

        let mut text = String::new();
        if let Some(pad) = pad_before {
            text.push_str(pad);
        }
        text.push_str(&crate::embed::embed_marker(ordinal));
        if let Some(pad) = pad_after {
            text.push_str(pad);
        }
        self.dispatch(EditorEvent::InsertText(text), cx);
    }

    /// Remove the `{{ embed N }}` marker for `ordinal` — the symmetric twin of
    /// [`Self::insert_embed_marker`], and the host's only way to un-place an
    /// embed without rewriting the whole buffer.
    ///
    /// Only a **recognized** embed block is removed (the marker must be mapped
    /// in the current embed set and stand as its own top-level paragraph), so
    /// the ordinal must still be in the map when this is called — clear it
    /// from the map afterwards, not before. The marker's paragraph goes, along
    /// with the blank line that separated it from a neighbour, leaving the
    /// surrounding prose joined as if the embed had never been placed. Routed
    /// through the normal update pipeline, so it is one undo step and emits
    /// [`MarkdownEditorEvent::Change`]. A no-op when no such block exists.
    pub fn remove_embed_marker(&mut self, ordinal: u64, cx: &mut Context<Self>) {
        let md = &self.state.markdown;
        let Some(block) = crate::embed::embed_blocks(md, &self.state.embeds)
            .into_iter()
            .find(|b| b.ordinal == ordinal)
        else {
            return;
        };
        let (mut start, mut end) = (block.range.start, block.range.end);
        // Swallow exactly one paragraph separator so the neighbours rejoin as
        // they were: the **trailing** blank-line run when there is one (the
        // leading run then separates the surviving neighbours), otherwise the
        // leading run (a trailing embed must not leave the body ending in
        // blank lines).
        let bytes = md.as_bytes();
        let mut after = end;
        while after < md.len() && (bytes[after] == b'\n' || bytes[after] == b'\r') {
            after += 1;
        }
        if after > end {
            end = after;
        } else {
            while start > 0 && (bytes[start - 1] == b'\n' || bytes[start - 1] == b'\r') {
                start -= 1;
            }
        }
        self.state.selection = Selection::range(start, end);
        self.dispatch(EditorEvent::InsertText(String::new()), cx);
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

    /// Test seam: begin a pointer selection at a source `offset` with a given
    /// `click_count` (1 = char, 2 = word, 3 = line), bypassing hit-testing so
    /// double/triple-click and drag-granularity behavior can be exercised
    /// without laid-out geometry. Mirrors what [`Self::on_mouse_down`] does
    /// after resolving the click position.
    #[doc(hidden)]
    pub fn begin_selection_for_test(
        &mut self,
        offset: usize,
        click_count: usize,
        shift: bool,
        cx: &mut Context<Self>,
    ) {
        self.begin_selection(offset, click_count, shift, cx);
    }

    /// Test seam: extend the in-progress selection to a source `offset`
    /// (mirrors a drag step), honoring the [`SelectMode`] set by the last
    /// [`Self::begin_selection_for_test`].
    #[doc(hidden)]
    pub fn extend_selection_for_test(&mut self, offset: usize, cx: &mut Context<Self>) {
        self.extend_selection_to(offset, cx);
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
        // A disabled (read-only) editor renders as *published* markdown: there
        // is no live cursor, so every WYSIWYG delimiter is hidden and the
        // content reads as finished prose. This is what makes a read-only post
        // pixel-identical to a clean rendered reply (used by the space view to
        // render every transcript post through the same editor as the composer).
        if self.disabled {
            render_readonly(&self.state, &tree)
        } else {
            render(&self.state, &tree)
        }
    }

    pub fn cursor_offset(&self) -> usize {
        self.state.selection.head()
    }

    /// True when the caret's wrap-boundary affinity is [`WrapAffinity::Downstream`]
    /// — the caret prefers the START of the lower wrapped row. The caret renderer
    /// (`build_caret_and_selection`) reads this to place a boundary caret on the
    /// intended visual row.
    pub(crate) fn caret_downstream(&self) -> bool {
        matches!(self.wrap_affinity, WrapAffinity::Downstream)
    }

    /// The editor's natural content height — the vertical extent of the laid-out
    /// text, independent of any `min_height` the element reserves below it (this
    /// reads the union of painted block bounds, which sit at the top regardless
    /// of the container floor). Zero until the first paint. Hosts that grow the
    /// editor into a taller slot (the space composer's docked runway) read this
    /// to size themselves without a feedback loop against their own floor.
    pub fn content_height(&self) -> Pixels {
        self.last_bounds.map(|b| b.size.height).unwrap_or(px(0.))
    }

    /// The caret's vertical span **relative to the editor's laid-out content
    /// top** (content-top = y 0), returned as `(top, bottom)` in pixels;
    /// `bottom - top` is the caret's row height. This is the read half of the
    /// host-driven scroll-into-view contract: the editor has no internal
    /// vertical scroll (it lays out to its full content height and the host
    /// scrolls it), so a host that wraps the editor in its own scroll container
    /// (the space composer) reads this on every [`MarkdownEditorEvent::Change`]
    /// and adjusts *its* scroll offset to keep the caret visible.
    ///
    /// [`Self::content_y_for_offset`] at the caret, with the caret's
    /// wrap-boundary affinity (see [`WrapAffinity`]) rather than the default.
    pub fn caret_content_y(&self) -> Option<(Pixels, Pixels)> {
        self.content_y_at(self.state.selection.head(), self.caret_downstream())
    }

    /// The vertical span of an arbitrary source `offset`, relative to the
    /// editor's laid-out content top — [`Self::caret_content_y`] generalized
    /// to any offset, for a host that scrolls something *other* than the caret
    /// into view (a search match, a cited passage).
    ///
    /// An offset on a soft-wrap boundary resolves **downstream** (the start of
    /// the lower row): unlike the caret, an arbitrary offset carries no
    /// affinity, and a range's start is what a host reveals.
    ///
    /// **Coordinate frame.** See [`Self::caret_content_y`]. Paint-derived, so
    /// `None` before the first paint — a host revealing a position inside
    /// content that has not rendered yet (a virtualized transcript) must
    /// estimate first and correct once this answers.
    pub fn content_y_for_offset(&self, offset: usize) -> Option<(Pixels, Pixels)> {
        self.content_y_at(offset, true)
    }

    /// **Coordinate frame.** Derived from the previous frame's `last_blocks`
    /// exactly like [`Self::bounds_for_range`], but re-based to content-local
    /// coordinates: `last_blocks`/`last_bounds` are window-absolute, so
    /// subtracting `last_bounds.origin.y` (the top of the painted content)
    /// yields the y a scroll container measures from its own content top. The
    /// value is independent of any `min_height` runway (the laid-out text sits
    /// at the content top regardless). Returns `None` before the first paint
    /// (no layout to consult) or when the offset isn't covered by any laid-out
    /// line.
    fn content_y_at(&self, offset: usize, downstream: bool) -> Option<(Pixels, Pixels)> {
        let content_top = self.last_bounds?.origin.y;
        // Sort keys so an offset sitting on a block boundary (claimed by two
        // adjacent lines) resolves deterministically to the earlier block,
        // rather than by `HashMap` iteration order.
        let mut keys: Vec<usize> = self.last_blocks.keys().copied().collect();
        keys.sort_unstable();
        // Fallback for an offset past every laid-out range — the document end
        // after a trailing newline synthesizes an empty paragraph that isn't
        // always laid out as a line, the same edge `bounds_for_range` hits. Keep
        // the latest line whose range ends at or before the offset and clamp it
        // onto it (its last row) rather than returning `None`.
        let mut fallback: Option<&crate::element::LaidOutLine> = None;
        for k in keys {
            for line in &self.last_blocks[&k].lines {
                if line.contains_source_offset(offset) {
                    let local = line.local_position_for_source_offset_biased(offset, downstream);
                    let top = line.origin.y + local.y - content_top;
                    return Some((top, top + line.row_height));
                }
                if line.source_range.end <= offset
                    && fallback.is_none_or(|f| line.source_range.end >= f.source_range.end)
                {
                    fallback = Some(line);
                }
            }
        }
        let line = fallback?;
        let local = line.local_position_for_source_offset_biased(offset, downstream);
        let top = line.origin.y + local.y - content_top;
        Some((top, top + line.row_height))
    }

    /// Whether an IME composition (preedit) is in progress — the buffer holds
    /// marked text the user has not committed.
    ///
    /// Unlike some input primitives, this editor emits
    /// [`MarkdownEditorEvent::Change`] on **every** preedit keystroke, because
    /// the marked text really is in the buffer and a host sizing itself to the
    /// content must follow it. A host that acts on the text's *meaning*
    /// instead — searching it, sending it, parsing it — should ask this first
    /// and wait: the reader has not chosen those characters yet. (The
    /// `EntityInputHandler::marked_text_range` half of the IME contract needs
    /// a `Window` and is not a host-facing query.)
    pub fn is_composing(&self) -> bool {
        self.marked_range.is_some()
    }

    fn dispatch(&mut self, event: EditorEvent, cx: &mut Context<Self>) {
        // Any non-vertical event invalidates the intended-x streak.
        // Vertical events (handled by `vertical_move` below) update
        // `intended_x` directly without going through this helper.
        self.intended_x = None;
        // Left/Right/word moves/edits/etc. reset the caret to the default
        // downstream affinity — a Right onto a soft-wrap boundary reads as the
        // start of the next row, and an edit's caret prefers the lower row.
        self.wrap_affinity = WrapAffinity::Downstream;
        let before = std::mem::take(&mut self.state);
        // A read-only editor's events (selection, navigation, select-all) must
        // never rewrite the buffer: the host owns the document, and the
        // canonicalization passes belong to editing. Routing through the
        // readonly update keeps the buffer byte-identical to what the host
        // seeded (see `update::update_readonly`).
        self.state = if self.disabled {
            update::update_readonly(before.clone(), event)
        } else {
            update::update_guarded(before.clone(), event, &mut self.table_guard)
        };
        self.marked_range = None;
        // Compare the buffer across the update so selection-only events
        // (Move*/Extend*/SetSelection) don't push an undo step or count
        // as a content change. The composer buffer is small, so the
        // pre-edit clone is negligible.
        if self.state.markdown != before.markdown {
            self.record_history(before);
            cx.emit(MarkdownEditorEvent::Change);
        } else if self.state.selection != before.selection {
            // A pure caret/selection move (Left/Right/word/Move*/Extend*):
            // no buffer change, so no `Change` — but a host that scrolls the
            // caret into view still needs to know it moved.
            cx.emit(MarkdownEditorEvent::SelectionChanged);
        }
        cx.notify();
    }

    /// Push the pre-edit `before` state onto the undo stack — the single
    /// history-recording choke point shared by `dispatch` (action- and
    /// paste-driven edits) and `replace_and_mark_text_in_range` (the IME
    /// / typed-input path that mutates `self.state` directly). Callers
    /// invoke it *after* `self.state` has been advanced to the post-edit
    /// value, having confirmed the buffer actually changed.
    ///
    /// **Coalescing.** Consecutive single-character, non-whitespace
    /// insertions that continue at the caret collapse into one undo
    /// step, so a typed word undoes as a unit rather than
    /// character-by-character. The run breaks (a new undo entry is
    /// pushed) whenever the edit is anything else — a deletion, a
    /// whitespace or newline insertion, a paste, a structural edit, or a
    /// caret jump away from where the last insert ended. This is the
    /// simpler clock-free heuristic: no time window, just contiguity +
    /// "is it a plain typed character". Every fresh edit clears the redo
    /// stack.
    fn record_history(&mut self, before: EditorState) {
        // Is this edit a coalescible single-char type, and where did its
        // caret land? `None` for anything that should break the run.
        let type_end = coalescible_type_end(&before, &self.state);
        let coalesce = match (type_end, self.coalesce_anchor) {
            // Continue the run only when the previous edit was also a
            // coalescible type and *this* edit began exactly where it
            // left off (contiguous typing, no caret jump between).
            (Some(_), Some(anchor)) => before.selection == Selection::Cursor(anchor),
            _ => false,
        };
        if !coalesce {
            self.undo_stack.push(before);
            if self.undo_stack.len() > MAX_HISTORY {
                self.undo_stack.remove(0);
            }
        }
        // A coalescing type extends the existing top entry (the snapshot
        // from the start of the run) — we simply don't push a new one.
        self.redo_stack.clear();
        self.coalesce_anchor = type_end;
    }

    fn undo(&mut self, _: &Undo, _: &mut Window, cx: &mut Context<Self>) {
        let Some(prev) = self.undo_stack.pop() else {
            return;
        };
        let current = std::mem::replace(&mut self.state, prev);
        self.redo_stack.push(current);
        self.marked_range = None;
        self.intended_x = None;
        // An undo is a hard boundary: the next typed character starts a
        // fresh coalescing run rather than folding into the step we just
        // reverted.
        self.coalesce_anchor = None;
        cx.emit(MarkdownEditorEvent::Change);
        cx.notify();
    }

    fn redo(&mut self, _: &Redo, _: &mut Window, cx: &mut Context<Self>) {
        let Some(next) = self.redo_stack.pop() else {
            return;
        };
        let current = std::mem::replace(&mut self.state, next);
        self.undo_stack.push(current);
        self.marked_range = None;
        self.intended_x = None;
        self.coalesce_anchor = None;
        cx.emit(MarkdownEditorEvent::Change);
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
        let local = line.local_position_for_source_offset_biased(
            cursor,
            matches!(self.wrap_affinity, WrapAffinity::Downstream),
        );
        let global_x = line.origin.x + local.x;
        let target_x = self.intended_x.unwrap_or(global_x);
        let row_h = line.row_height;
        if row_h <= px(0.) {
            return None;
        }
        // The caret's current wrap-row within its line. `local.y` is a whole
        // multiple of `row_h` (position_for_index is passed `row_h`), so this
        // `round` is exact — unlike a `floor`, which float-drift would spoil.
        let cur_row = (local.y / row_h).round() as i32;
        let line_rows = line.row_count() as i32;
        let target_row = cur_row + direction;

        // A coarse target y, only for picking the nearest line when the move
        // leaves this logical line (the paragraph_gap-absorbing search below).
        let target_global_y = line.origin.y + (target_row as f32) * row_h;
        let current_top = line.origin.y;
        let current_bot = current_top + line.wrapped_height;

        // Intra-line iff the target row stays within this line's rows;
        // otherwise cross to the nearest line in the direction of motion and
        // enter it at its edge row (first row going down, last going up).
        let (target_line, target_row_in_line): (&crate::element::LaidOutLine, i32) =
            if target_row >= 0 && target_row < line_rows {
                (line, target_row)
            } else {
                // The current line is filtered out (behind us) and lines on the
                // wrong side of the motion are filtered out; among the rest pick
                // the one whose vertical bounds are closest to `target_global_y`
                // — and at equal vertical distance, the one horizontally
                // closest to `target_x`. The x tie-break matters for table
                // rows, whose side-by-side cell boxes all sit at the same y
                // (see `element::layout_table`): Down from column 2 must land
                // in the next row's column-2 box, not its first cell.
                let mut best: Option<(&crate::element::LaidOutLine, Pixels, Pixels)> = None;
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
                        let left = cand.origin.x;
                        let right = left + cand.line.width();
                        let x_dist = if target_x < left {
                            left - target_x
                        } else if target_x > right {
                            target_x - right
                        } else {
                            px(0.)
                        };
                        let better = match best {
                            None => true,
                            Some((_, bd, bx)) => dist < bd || (dist == bd && x_dist < bx),
                        };
                        if better {
                            best = Some((cand, dist, x_dist));
                        }
                    }
                }
                let best = best.map(|(l, _, _)| l)?;
                let edge = if direction > 0 {
                    0
                } else {
                    best.row_count() as i32 - 1
                };
                (best, edge)
            };

        // Sample the VERTICAL CENTER of the target row — never its top edge.
        // gpui's `closest_index_for_position` floors `y / row_h`, and a top-edge
        // sample `row * row_h` divided back float-rounds to `row - 1` (the
        // "Down stalls after a few rows / Home lands a row up" bug — it takes a
        // few rows for the drift between our `row_h` and gpui's internal line
        // spacing to cross a boundary, which is why it always struck a specific
        // mid-paragraph row). The half-row offset keeps the floor exact.
        let sample_y = px((target_row_in_line as f32 + 0.5) * row_h.as_f32());
        let local_target = Point::new(target_x - target_line.origin.x, sample_y);
        let new_offset = target_line.source_offset_for_local_point(local_target);

        // Record where the caret landed relative to the boundary ambiguity: at
        // the target row's START (a soft-wrap boundary whose upper-affinity row
        // is the row above) it's `Downstream`, else `Upstream`. Lets the *next*
        // vertical press compute the caret's current row correctly.
        let up_row =
            (target_line.local_position_for_source_offset(new_offset).y / row_h).round() as i32;
        self.wrap_affinity = if up_row < target_row_in_line {
            WrapAffinity::Downstream
        } else {
            WrapAffinity::Upstream
        };

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
                self.wrap_affinity = WrapAffinity::Downstream;
                let next = std::mem::take(&mut self.state);
                self.state = update::update_guarded(next, fallback, &mut self.table_guard);
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
        let before_sel = self.state.selection;
        let next = std::mem::take(&mut self.state);
        self.state = update::update_guarded(
            next,
            EditorEvent::SetSelection(new_sel),
            &mut self.table_guard,
        );
        self.marked_range = None;
        // A vertical move changes the caret without touching the buffer — tell
        // a host that scrolls the caret into view (no `Change` is emitted).
        if self.state.selection != before_sel {
            cx.emit(MarkdownEditorEvent::SelectionChanged);
        }
        cx.notify();
    }

    /// Source offset at the start (`to_end == false`) or end
    /// (`to_end == true`) of the caret's current *display line* — the
    /// soft-wrapped visual row it sits on, not the whole `\n`-delimited
    /// source line. Consults the laid-out row geometry from the previous
    /// frame in `last_blocks`. Returns `None` when there's no layout to
    /// consult (pre-paint state, headless tests), so the caller can fall
    /// back to the source-line `MoveLineStart` / `MoveLineEnd` event.
    ///
    /// Mirrors [`Self::visual_move_caret`]: it locates the `LaidOutLine`
    /// containing the caret, reads the caret's wrap-row `y` via
    /// `local_position_for_source_offset`, then maps a point at that
    /// row's left edge (`x = 0`, Home) or far-right edge (`x = huge`,
    /// End) back to a source offset with `source_offset_for_local_point`.
    /// On the first wrap row of a blockquote / list line the shaped
    /// display text already begins past the hidden chain prefix, so
    /// display-line Home lands on the visible content edge for free; the
    /// caller additionally routes the result through `SetSelection`,
    /// whose `nearest_allowed_position` snap keeps the caret off any
    /// forbidden position (the source path's `next_allowed_position`
    /// chain-prefix skip, preserved).
    fn visual_line_bound(&self, to_end: bool) -> Option<usize> {
        if self.last_blocks.is_empty() {
            return None;
        }
        let cursor = self.state.selection.head();
        let mut keys: Vec<usize> = self.last_blocks.keys().copied().collect();
        keys.sort();

        // Find the LaidOutLine containing the cursor (same disjoint-range
        // search as `visual_move_caret`). A display line is always within
        // a single logical line, so this is the only line we consult.
        let mut current: Option<&crate::element::LaidOutLine> = None;
        for k in &keys {
            let block = &self.last_blocks[k];
            for line in &block.lines {
                if line.contains_source_offset(cursor) {
                    current = Some(line);
                    break;
                }
            }
            if current.is_some() {
                break;
            }
        }
        let line = current?;

        // The caret's wrap-row inside this line. `x` picks the edge: 0 for the
        // row's start, a large finite value for its end (the mapping clamps into
        // the row, resolving to the last display index of that wrap row).
        let local = line.local_position_for_source_offset_biased(
            cursor,
            matches!(self.wrap_affinity, WrapAffinity::Downstream),
        );
        let row_h = line.row_height;
        if row_h <= px(0.) {
            return None;
        }
        // Sample the row's VERTICAL CENTER, never its top edge: `local.y` is a
        // whole multiple of `row_h`, and `closest_index_for_position` floors
        // `y / row_h`, so a top-edge sample float-rounds down into the row above
        // (Home/End would land a row too high). `round` is exact for the row
        // index; the half-row offset keeps the floor exact.
        let cur_row = (local.y / row_h).round();
        let sample_y = px((cur_row + 0.5) * row_h.as_f32());
        let x = if to_end { px(1.0e6) } else { px(0.0) };
        let target = Point::new(x, sample_y);
        Some(line.source_offset_for_local_point(target))
    }

    /// Dispatch path for Home / End / Shift+Home / Shift+End (and their
    /// Cmd+Left / Cmd+Right / Cmd+Shift+Left / Cmd+Shift+Right aliases).
    /// Tries the display-line-aware [`Self::visual_line_bound`] first; on
    /// success builds the appropriate `Selection` and routes it through
    /// `SetSelection` (which applies the forbidden-position snap). On
    /// failure (no layout to consult) it falls back to the source-line
    /// `MoveLineStart` / `MoveLineEnd` (or the `Extend*` variant) event
    /// so headless tests and pre-paint state still move predictably.
    fn line_bound_move(
        &mut self,
        to_end: bool,
        extending: bool,
        fallback: EditorEvent,
        cx: &mut Context<Self>,
    ) {
        // Home / End are horizontal motions: they end any vertical
        // intended-x streak, so the next Up / Down re-anchors from the
        // caret's new column. (`dispatch` clears this for other events;
        // this path is hand-rolled, so clear it explicitly.)
        self.intended_x = None;
        // Read the current caret's row (via `visual_line_bound`, which uses the
        // *existing* affinity to disambiguate a boundary caret) BEFORE updating
        // the affinity below — otherwise the read would be self-referential.
        let bound = self.visual_line_bound(to_end);
        // End keeps the caret on the row it acted on (`Upstream`); Home lands it
        // at the row's start (`Downstream`). Set on both the layout-aware and the
        // no-layout fallback paths.
        self.wrap_affinity = if to_end {
            WrapAffinity::Upstream
        } else {
            WrapAffinity::Downstream
        };
        let new_head = match bound {
            Some(offset) => offset,
            None => {
                let next = std::mem::take(&mut self.state);
                self.state = update::update_guarded(next, fallback, &mut self.table_guard);
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
        let before_sel = self.state.selection;
        let next = std::mem::take(&mut self.state);
        self.state = update::update_guarded(
            next,
            EditorEvent::SetSelection(new_sel),
            &mut self.table_guard,
        );
        self.marked_range = None;
        // Home/End move the caret without a buffer change — notify a
        // caret-into-view host (no `Change` is emitted).
        if self.state.selection != before_sel {
            cx.emit(MarkdownEditorEvent::SelectionChanged);
        }
        cx.notify();
    }

    fn backspace(&mut self, _: &Backspace, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::DeleteBackward, cx);
    }
    fn delete(&mut self, _: &Delete, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::DeleteForward, cx);
    }
    fn toggle_bold(&mut self, _: &ToggleBold, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::ToggleInlineFormat(InlineFormat::Strong), cx);
    }
    fn toggle_italic(&mut self, _: &ToggleItalic, _: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(EditorEvent::ToggleInlineFormat(InlineFormat::Emphasis), cx);
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
        self.line_bound_move(false, false, EditorEvent::MoveLineStart, cx);
    }
    fn end(&mut self, _: &End, _: &mut Window, cx: &mut Context<Self>) {
        self.line_bound_move(true, false, EditorEvent::MoveLineEnd, cx);
    }
    fn shift_home(&mut self, _: &ShiftHome, _: &mut Window, cx: &mut Context<Self>) {
        self.line_bound_move(false, true, EditorEvent::ExtendLineStart, cx);
    }
    fn shift_end(&mut self, _: &ShiftEnd, _: &mut Window, cx: &mut Context<Self>) {
        self.line_bound_move(true, true, EditorEvent::ExtendLineEnd, cx);
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

    /// Run a clipboard/selection command programmatically — the host-driven
    /// twin of the keymap actions, for a UI that issues the command itself (a
    /// context menu) rather than routing a keystroke through the focused
    /// element. `Cut` and `Paste` are refused on a read-only editor, exactly
    /// as their action handlers are (they're never registered there).
    pub fn perform(&mut self, command: EditorCommand, window: &mut Window, cx: &mut Context<Self>) {
        match command {
            EditorCommand::Cut if !self.disabled => self.cut(&Cut, window, cx),
            EditorCommand::Paste if !self.disabled => self.paste(&Paste, window, cx),
            EditorCommand::Cut | EditorCommand::Paste => {}
            EditorCommand::Copy => self.copy(&Copy, window, cx),
            EditorCommand::SelectAll => self.select_all(&SelectAll, window, cx),
        }
    }

    /// Whether `command`'s preconditions are met on this editor right now —
    /// the **enablement twin** of [`Self::perform`], and it lives beside it
    /// deliberately: a host that decides which verbs to offer by re-deriving
    /// the conditions itself will eventually advertise one `perform` then
    /// declines. `Cut`/`Paste` need an editable editor, `Cut`/`Copy` a
    /// non-empty selection, and `Paste` **text on the clipboard** — without
    /// which `perform` returns having touched nothing at all. `SelectAll` has
    /// no precondition; selecting all of an empty document is the degenerate
    /// case of a command that works, not a verb that does nothing.
    pub fn can_perform(&self, command: EditorCommand, cx: &App) -> bool {
        match command {
            EditorCommand::Cut => {
                !self.disabled && !self.state.selection.selection_range().is_empty()
            }
            EditorCommand::Copy => !self.state.selection.selection_range().is_empty(),
            EditorCommand::Paste => {
                !self.disabled
                    && cx
                        .read_from_clipboard()
                        .and_then(|item| item.text())
                        .is_some_and(|text| !text.is_empty())
            }
            EditorCommand::SelectAll => true,
        }
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

    fn on_mouse_down(
        &mut self,
        event: &MouseDownEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // A click on a rendered embed block invokes the host's callback (the
        // wave-2 click-to-navigate seam). The click still places the caret
        // through the normal path below — the snap machinery resolves it to
        // the marker's edge, since the interior is forbidden.
        if let Some(ordinal) = self.embed_ordinal_at_position(event.position)
            && let Some(cb) = self.on_embed_click.clone()
        {
            cb(ordinal, window, cx);
        }
        let offset = self.offset_for_position(event.position);
        // A single unmodified press on highlighted text arms the highlight
        // click; the callback fires on mouse-up only if no drag created a
        // selection range in between (see `highlight_press`).
        self.highlight_press = (event.click_count == 1
            && !event.modifiers.shift
            && self.on_highlight_click.is_some()
            && !self
                .highlights_in(crate::highlight::HighlightLayer::Base)
                .keys_at(offset)
                .is_empty())
        .then_some(offset);
        self.begin_selection(offset, event.click_count, event.modifiers.shift, cx);
    }

    /// The context-menu gesture (right mouse-down). The editor opens no menu:
    /// it places the caret and hands the position to the host's
    /// `on_context_menu` callback.
    ///
    /// **Caret placement follows the platform convention**, and it is what
    /// makes a host's "Paste" land where the user pointed: a press *inside* the
    /// current selection leaves it alone (that selection is what Cut/Copy will
    /// act on), a press outside collapses the caret to the clicked offset.
    fn on_right_mouse_down(
        &mut self,
        event: &MouseDownEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(cb) = self.on_context_menu.clone() else {
            return;
        };
        let offset = self.offset_for_position(event.position);
        if !self.state.selection.selection_range().contains(&offset) {
            self.dispatch(EditorEvent::SetSelection(Selection::cursor(offset)), cx);
        }
        cb(&event.position, window, cx);
    }

    /// The rendered embed block under `position` (window coordinates), if
    /// any. Reads the ordinal the render recorded on the painted block
    /// (`LaidOutBlock::embed_ordinal`) — never re-derived from source, so
    /// range conventions (the parser's folded trailing newline vs the
    /// scan's trimmed ranges) can't desynchronize the hit-test from what
    /// actually painted, and a mousedown never re-parses the buffer.
    #[doc(hidden)]
    pub fn embed_ordinal_at_position(&self, position: Point<Pixels>) -> Option<u64> {
        self.last_blocks
            .values()
            .find(|b| b.block_bounds.contains(&position))
            .and_then(|b| b.embed_ordinal)
    }

    /// Start (or, with `shift`, extend) a pointer selection at `offset`, keyed
    /// by `click_count`: 1 = character (caret / shift-range), 2 = the word at
    /// `offset`, ≥3 = the whole line. The chosen granularity is recorded as the
    /// drag's [`SelectMode`] so a subsequent drag extends by the same unit
    /// (native macOS behavior). Shared by the mouse handler and the test seam.
    fn begin_selection(
        &mut self,
        offset: usize,
        click_count: usize,
        shift: bool,
        cx: &mut Context<Self>,
    ) {
        self.is_selecting = true;
        let mode = match click_count {
            0 | 1 => SelectMode::Char,
            2 => SelectMode::Word,
            _ => SelectMode::Line,
        };
        self.select_mode = mode;

        if shift {
            // A shift-press *extends* the existing selection instead of starting
            // a new one — and it must keep the existing anchor as the drag
            // anchor, so a subsequent drag continues to grow the original range
            // rather than discarding it and re-anchoring at the click point.
            // For character mode we capture the live selection anchor (it may
            // have moved by keyboard since the last press); for word/line mode
            // we preserve the previously-anchored unit (`select_anchor_range`),
            // so shift-double/triple-click keeps extending from the first word /
            // line. Then extend to the click at the current granularity — the
            // same path a drag takes.
            if mode == SelectMode::Char {
                let anchor = self.state.selection.anchor();
                self.select_anchor_range = anchor..anchor;
            }
            self.extend_selection_to(offset, cx);
            return;
        }

        let new_sel = match mode {
            SelectMode::Char => {
                self.select_anchor_range = offset..offset;
                Selection::Cursor(offset)
            }
            SelectMode::Word => {
                let range = word_range_at(&self.state.markdown, offset);
                self.select_anchor_range = range.clone();
                Selection::range(range.start, range.end)
            }
            SelectMode::Line => {
                let range = line_range_at(&self.state.markdown, offset);
                self.select_anchor_range = range.clone();
                Selection::range(range.start, range.end)
            }
        };
        self.dispatch(EditorEvent::SetSelection(new_sel), cx);
    }

    fn on_mouse_up(&mut self, _: &MouseUpEvent, window: &mut Window, cx: &mut Context<Self>) {
        self.is_selecting = false;
        // Resolve an armed highlight press: a plain click (no drag range)
        // reports the keys of every range containing the pressed offset.
        if let Some(offset) = self.highlight_press.take()
            && self.state.selection.is_collapsed()
            && let Some(cb) = self.on_highlight_click.clone()
        {
            let keys = self
                .highlights_in(crate::highlight::HighlightLayer::Base)
                .keys_at(offset);
            if !keys.is_empty() {
                cb(&keys, window, cx);
            }
        }
    }

    fn on_mouse_move(&mut self, event: &MouseMoveEvent, _: &mut Window, cx: &mut Context<Self>) {
        if !self.is_selecting {
            return;
        }
        let offset = self.offset_for_position(event.position);
        self.extend_selection_to(offset, cx);
    }

    /// Extend the in-progress selection so its head reaches `offset`, honoring
    /// the drag's [`SelectMode`]: a character drag moves the head to `offset`; a
    /// word/line drag grows to cover the union of the anchored unit and the
    /// unit at `offset`, with the head on whichever side the pointer is past —
    /// so dragging back past the anchor flips the selected end the way a native
    /// word/line drag does. Shared by the in-bounds handler, the window-global
    /// drag listener, and the test seam.
    fn extend_selection_to(&mut self, offset: usize, cx: &mut Context<Self>) {
        let anchor = self.select_anchor_range.clone();
        let new_sel = match self.select_mode {
            SelectMode::Char => Selection::range(anchor.start, offset),
            SelectMode::Word | SelectMode::Line => {
                let unit = if self.select_mode == SelectMode::Word {
                    word_range_at(&self.state.markdown, offset)
                } else {
                    line_range_at(&self.state.markdown, offset)
                };
                if offset >= anchor.end {
                    // Dragging forward: anchor's start is fixed, head grows to
                    // the far edge of the unit under the pointer.
                    Selection::range(anchor.start, unit.end.max(anchor.end))
                } else {
                    // Dragging back past the anchored unit: anchor's end is
                    // fixed, head shrinks to the near edge of the unit.
                    Selection::range(anchor.end, unit.start.min(anchor.start))
                }
            }
        };
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

        // Symmetric bottom case: a click below the last line's bottom is in the
        // editor's blank tail (the excess a `min_height` reserves below the
        // text) — collapse to document end so clicking the runway lands the
        // caret after the last character, notes-editor style. This is distinct
        // from a click in an inter-block *gap* (handled by the nearest-line pass
        // below), which is always above the last line's bottom.
        if let Some(last_key) = keys.last()
            && let Some(last_line) = self.last_blocks[*last_key].lines.last()
            && position.y >= last_line.origin.y + last_line.wrapped_height
        {
            return self.state.markdown.len();
        }

        // First pass: direct hit. If `position.y` falls in a line's
        // vertical extent, hit-test inside that line. Multiple lines
        // can share a y-band — a table row is several side-by-side
        // cell boxes (see `element::layout_table`) — so among y-hits
        // the pick is by **horizontal** proximity: a line whose x
        // span contains the point wins outright; otherwise the
        // nearest by x. Ordinary blocks have one line per band, so
        // this degenerates to the old first-hit behavior.
        //
        // Second pass: nearest line. Lines don't tile vertically — there's
        // a `paragraph_gap` between blocks — so a mouse drag whose y
        // momentarily falls in the gap would otherwise hit no line at
        // all. The previous fallback returned `markdown.len()`, making
        // the selection head shoot to end-of-doc every time the mouse
        // crossed a gap. Snap to the closest line by vertical distance,
        // then clamp the local y to that line's bounds so the x
        // coordinate still picks the right column.
        let mut y_hit: Option<(&crate::element::LaidOutLine, Pixels)> = None;
        let mut best: Option<&crate::element::LaidOutLine> = None;
        let mut best_distance: Pixels = px(f32::INFINITY);
        for key in &keys {
            let block = &self.last_blocks[*key];
            for line in &block.lines {
                let line_top = line.origin.y;
                let line_bottom = line_top + line.wrapped_height;
                if position.y >= line_top && position.y < line_bottom {
                    let left = line.origin.x;
                    let right = left + line.line.width();
                    let x_dist = if position.x < left {
                        left - position.x
                    } else if position.x > right {
                        position.x - right
                    } else {
                        px(0.0)
                    };
                    match y_hit {
                        Some((_, d)) if d <= x_dist => {}
                        _ => y_hit = Some((line, x_dist)),
                    }
                    continue;
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
        if let Some((line, _)) = y_hit {
            let local = Point::new(position.x - line.origin.x, position.y - line.origin.y);
            return line.source_offset_for_local_point(local);
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

        // Snapshot the pre-edit state so this direct-mutation path (IME /
        // typed input, which bypasses `dispatch`/`update`) records undo
        // history through the same choke point as everything else.
        let before = self.state.clone();

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
        // A typed / IME insertion places the caret fresh — default downstream
        // affinity (a caret pushed onto a soft-wrap boundary reads as the start
        // of the next row).
        self.wrap_affinity = WrapAffinity::Downstream;
        if self.state.markdown != before.markdown {
            self.record_history(before);
        }
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
    min_height: Option<Pixels>,
    on_embed_click: Option<EmbedClickHandler>,
    on_highlight_click: Option<HighlightClickHandler>,
    on_context_menu: Option<ContextMenuHandler>,
}

impl MarkdownEditor {
    /// Build the element over a host-owned state entity.
    pub fn new(state: &Entity<MarkdownEditorState>) -> Self {
        Self {
            state: state.clone(),
            style: None,
            disabled: false,
            min_height: None,
            on_embed_click: None,
            on_highlight_click: None,
            on_context_menu: None,
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

    /// Render read-only: no mutating key/IME handlers are registered, the
    /// `EntityInputHandler` text mutations early-return, and the caret is
    /// hidden, so the surface rejects edits. Text can still be selected (mouse
    /// or shift-navigation) and copied. Mirrors `Input::disabled`.
    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }

    /// Reserve at least `height` for the editor container, so it fills a slot
    /// taller than its text. The extra space below the last line is part of the
    /// editor's own click surface: a click there resolves (via
    /// `offset_for_position`) to document end, giving a "notes editor" feel
    /// without any host-side overlay listener. The natural text height is still
    /// reported by [`MarkdownEditorState::content_height`], so a host can size a
    /// container from the text without feeding its own floor back in.
    pub fn min_height(mut self, height: Pixels) -> Self {
        self.min_height = Some(height);
        self
    }

    /// Host callback for a click on a rendered embed block: receives the
    /// embed's ordinal (the shared key into the host's own bookkeeping — see
    /// [`crate::embed`]). Registered on both editable and read-only editors;
    /// the editor never interprets the ordinal.
    pub fn on_embed_click(
        mut self,
        callback: impl Fn(u64, &mut Window, &mut App) + 'static,
    ) -> Self {
        self.on_embed_click = Some(Rc::new(callback));
        self
    }

    /// Host callback for a plain click on host-supplied highlighted text (see
    /// [`crate::highlight`]): receives the keys of every highlight range
    /// containing the clicked offset. A drag over a highlight selects text
    /// normally and never fires this. Registered on both editable and
    /// read-only editors; the editor never interprets the keys.
    pub fn on_highlight_click(
        mut self,
        callback: impl Fn(&[u64], &mut Window, &mut App) + 'static,
    ) -> Self {
        self.on_highlight_click = Some(Rc::new(callback));
        self
    }

    /// Host callback for the context-menu gesture (right mouse-down):
    /// receives the window-coordinate position a menu should open at. The
    /// editor places the caret first (a press outside the selection collapses
    /// to it, a press inside keeps the selection) and opens nothing itself —
    /// what belongs in the menu is the host's business. Registered on
    /// read-only editors too. Without this prop a right-click does nothing.
    pub fn on_context_menu(
        mut self,
        callback: impl Fn(&Point<Pixels>, &mut Window, &mut App) + 'static,
    ) -> Self {
        self.on_context_menu = Some(Rc::new(callback));
        self
    }
}

impl RenderOnce for MarkdownEditor {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        // Per-frame reset + sync the disabled prop onto the state (so the
        // IME handler can honor it). Block elements re-populate `last_blocks`
        // during paint; `frame_input_handler_set` re-arms IME registration.
        let embed_cb = self.on_embed_click.clone();
        let highlight_cb = self.on_highlight_click.clone();
        let context_menu_cb = self.on_context_menu.clone();
        self.state.update(cx, |st, _| {
            st.disabled = self.disabled;
            st.on_embed_click = embed_cb;
            st.on_highlight_click = highlight_cb;
            st.on_context_menu = context_menu_cb;
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
            // The IBeam cursor shows even when disabled: a read-only editor
            // still supports selecting text (for copy), so the affordance
            // should advertise that.
            .cursor(CursorStyle::IBeam)
            .w_full()
            .flex()
            .flex_col()
            .text_size(style.font_size)
            .text_color(style.text_color)
            .font_family(style.font_family.clone());

        // Fill a taller slot when asked; the excess below the text is part of
        // the editor's click surface (see `min_height`).
        if let Some(mh) = self.min_height {
            container = container.min_h(mh);
        }

        // Selection / navigation / copy handlers are registered even when
        // disabled: a read-only editor still supports selecting text (mouse
        // drag, shift-navigation, select-all) and copying it. None of these
        // mutate the document. Each routes the action into the *state* entity
        // via `window.listener_for` (the element→state bridge), the
        // gpui-component idiom.
        container = container
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
            .on_action(window.listener_for(&state, MarkdownEditorState::select_all))
            .on_action(window.listener_for(&state, MarkdownEditorState::copy))
            // Map the read-only Edit-menu action types
            // (`gpui_component::input::{Copy,SelectAll}`) onto the editor's own
            // implementations. The OS routes the Edit menu through the
            // responder chain via the `OsAction::*` selectors; those land as
            // `gpui_component::input::*` dispatched to the focused element.
            .on_action(
                window.listener_for(&state, |this, _: &gpui_component::input::Copy, w, cx| {
                    this.copy(&Copy, w, cx)
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
            .on_mouse_move(window.listener_for(&state, MarkdownEditorState::on_mouse_move))
            // The context-menu gesture, registered on read-only editors too:
            // a post body offers Select All / Copy / Quote even though it
            // rejects edits.
            .on_mouse_down(
                MouseButton::Right,
                window.listener_for(&state, MarkdownEditorState::on_right_mouse_down),
            );

        // Mutating handlers are registered only when editable.
        if !disabled {
            container = container
                .on_action(window.listener_for(&state, MarkdownEditorState::backspace))
                .on_action(window.listener_for(&state, MarkdownEditorState::delete))
                .on_action(window.listener_for(&state, MarkdownEditorState::enter))
                .on_action(window.listener_for(&state, MarkdownEditorState::tab))
                .on_action(window.listener_for(&state, MarkdownEditorState::shift_tab))
                .on_action(window.listener_for(&state, MarkdownEditorState::delete_word_backward))
                .on_action(window.listener_for(&state, MarkdownEditorState::delete_word_forward))
                .on_action(window.listener_for(&state, MarkdownEditorState::delete_to_line_start))
                .on_action(window.listener_for(&state, MarkdownEditorState::delete_to_line_end))
                .on_action(window.listener_for(&state, MarkdownEditorState::toggle_bold))
                .on_action(window.listener_for(&state, MarkdownEditorState::toggle_italic))
                .on_action(window.listener_for(&state, MarkdownEditorState::cut))
                .on_action(window.listener_for(&state, MarkdownEditorState::paste))
                .on_action(window.listener_for(&state, MarkdownEditorState::paste_plain))
                .on_action(window.listener_for(&state, MarkdownEditorState::undo))
                .on_action(window.listener_for(&state, MarkdownEditorState::redo))
                // Map the Edit-menu Undo/Redo action types
                // (`gpui_component::input::{Undo,Redo}`) onto the editor's
                // own implementations, so the macOS Edit menu's Undo/Redo
                // items reach the composer when it has focus — the same
                // bridge used above for Cut/Copy/Paste/Select All.
                .on_action(
                    window.listener_for(&state, |this, _: &gpui_component::input::Undo, w, cx| {
                        this.undo(&Undo, w, cx)
                    }),
                )
                .on_action(
                    window.listener_for(&state, |this, _: &gpui_component::input::Redo, w, cx| {
                        this.redo(&Redo, w, cx)
                    }),
                )
                // Map the mutating Edit-menu action types
                // (`gpui_component::input::{Cut,Paste}`) onto the editor's own
                // implementations.
                .on_action(
                    window.listener_for(&state, |this, _: &gpui_component::input::Cut, w, cx| {
                        this.cut(&Cut, w, cx)
                    }),
                )
                .on_action(
                    window.listener_for(&state, |this, _: &gpui_component::input::Paste, w, cx| {
                        this.paste(&Paste, w, cx)
                    }),
                );
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

        // Window-global drag tracking. `on_mouse_down` starts a selection, but
        // gpui only fires the div's hitbox-gated `on_mouse_move` while the
        // pointer stays inside the editor's bounds — and those bounds are
        // clipped to the visible viewport (the content mask). For a post taller
        // than the window that froze the selection at the fold: dragging toward
        // off-screen text left the hitbox, `on_mouse_move` stopped firing, and
        // the selection couldn't grow past the visible screenful — which read
        // as "long responses can't be selected". So we register **window-global**
        // move/up listeners (the same pattern the space view's minimap-scrollbar
        // drag uses), so the selection keeps extending to the offset under the
        // pointer anywhere in the window — including off-screen rows, whose
        // geometry is laid out because a visible post renders its whole editor.
        //
        // These are registered **unconditionally every frame**, not gated on
        // `is_selecting` — matching the minimap's rationale ("registering
        // unconditionally is cheap and avoids a first-move gap"). gpui dispatches
        // events against the *last painted* frame's listeners and only repaints
        // on the `cx.notify()` scheduled by the mouse-down; so had we gated on
        // `is_selecting`, the frame that was on screen when the press landed
        // carried no global listeners, and a fast first move that jumped outside
        // the clipped hitbox before that repaint would be missed by both the
        // local (hitbox-gated) and global paths — a drag starting near the
        // viewport edge could still freeze. Registering always means the
        // listeners are already present on the pre-press frame; they no-op unless
        // a drag is actually in flight (the `is_selecting` guards live inside the
        // handlers). Listeners are cleared each frame, so a hitbox-free `canvas`
        // re-registers them every frame.
        let drag_state = state.clone();
        container = container.child(
            gpui::canvas(
                |_, _, _| {},
                move |_bounds, _, window, _cx| {
                    let move_state = drag_state.clone();
                    window.on_mouse_event(move |ev: &MouseMoveEvent, phase, _window, cx| {
                        if phase != gpui::DispatchPhase::Bubble {
                            return;
                        }
                        move_state.update(cx, |st, cx| {
                            if !st.is_selecting {
                                return;
                            }
                            // The button was released without a
                            // delivered up event (e.g. off-window) —
                            // end the drag rather than track a phantom.
                            if !ev.dragging() {
                                st.is_selecting = false;
                                cx.notify();
                                return;
                            }
                            let offset = st.offset_for_position(ev.position);
                            st.extend_selection_to(offset, cx);
                        });
                    });
                    let up_state = drag_state.clone();
                    window.on_mouse_event(move |ev: &MouseUpEvent, phase, _window, cx| {
                        if phase != gpui::DispatchPhase::Bubble || ev.button != MouseButton::Left {
                            return;
                        }
                        up_state.update(cx, |st, cx| {
                            if st.is_selecting {
                                st.is_selecting = false;
                                cx.notify();
                            }
                        });
                    });
                },
            )
            .absolute()
            .size_full(),
        );

        // The first `BlockElement::paint` of the frame registers the
        // `EntityInputHandler` (IME / typed text → `replace_text_in_range`),
        // unless `disabled`.
        container
    }
}

/// Install the editor's default keymap, scoped to the `MarkdownEditor`
/// key context. Self-contained so the editor is a drop-in like
/// `gpui_component::Input` (whose keymap `gpui_component::init` installs) —
/// the host calls this once at startup instead of hand-rolling the bindings.
///
/// Chord-style commands (clipboard, select-all, the submit chords) bind with
/// gpui's `secondary-` modifier alias — ⌘ on macOS, Ctrl elsewhere — so one
/// binding serves both platforms. Motion and word/line deletion differ
/// *structurally* between the platforms (macOS: ⌥ = word, ⌘ = line/document;
/// Linux/Windows: Ctrl = word, Home/End = line, Ctrl+Home/End = document), so
/// those get per-platform tables rather than the alias — a blanket alias
/// would make Ctrl+← "line start" on Linux, which no Linux user expects.
///
/// The submit chords (`secondary-enter`, `secondary-shift-enter`) bind the
/// `Enter` action with `secondary: true`; the handler emits
/// [`MarkdownEditorEvent::PressEnter`] rather than inserting, so the host
/// subscribes for submit instead of binding the chords itself.
pub fn init(cx: &mut App) {
    let ctx = Some("MarkdownEditor");
    let mut bindings = vec![
        // Enter chords — plain/shift insert; secondary variants emit
        // PressEnter (⌘↩/⌘⇧↩ on macOS, Ctrl+↩/Ctrl+⇧↩ elsewhere).
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
            "secondary-enter",
            Enter {
                secondary: true,
                shift: false,
            },
            ctx,
        ),
        gpui::KeyBinding::new(
            "secondary-shift-enter",
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
        // Caret motion (platform-neutral)
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
        gpui::KeyBinding::new("shift-home", ShiftHome, ctx),
        gpui::KeyBinding::new("shift-end", ShiftEnd, ctx),
        // Clipboard — the `secondary-` alias, scoped to the editor context so
        // they coexist with `gpui_component::Input`'s own `Input`-context
        // bindings (which are per-platform upstream).
        gpui::KeyBinding::new("secondary-a", SelectAll, ctx),
        gpui::KeyBinding::new("secondary-c", Copy, ctx),
        gpui::KeyBinding::new("secondary-x", Cut, ctx),
        gpui::KeyBinding::new("secondary-v", Paste, ctx),
        gpui::KeyBinding::new("secondary-shift-v", PastePlain, ctx),
        // Semantic inline formatting (⌘B/⌘I on macOS, Ctrl+B/Ctrl+I
        // elsewhere). The update pipeline owns all delimiter/context logic.
        gpui::KeyBinding::new("secondary-b", ToggleBold, ctx),
        gpui::KeyBinding::new("secondary-i", ToggleItalic, ctx),
        // Undo / redo — the `secondary-` alias, scoped to the editor context
        // so they coexist with `gpui_component::Input`'s own bindings.
        gpui::KeyBinding::new("secondary-z", Undo, ctx),
        gpui::KeyBinding::new("secondary-shift-z", Redo, ctx),
    ];

    // macOS motion/deletion idiom: ⌥ = word, ⌘ = line (arrows) / document
    // (up/down), ⌥⌫/⌘⌫ delete by the same granularity.
    #[cfg(target_os = "macos")]
    bindings.extend([
        gpui::KeyBinding::new("alt-backspace", DeleteWordBackward, ctx),
        gpui::KeyBinding::new("alt-delete", DeleteWordForward, ctx),
        gpui::KeyBinding::new("cmd-backspace", DeleteToLineStart, ctx),
        gpui::KeyBinding::new("cmd-delete", DeleteToLineEnd, ctx),
        gpui::KeyBinding::new("cmd-left", Home, ctx),
        gpui::KeyBinding::new("cmd-right", End, ctx),
        gpui::KeyBinding::new("cmd-shift-left", ShiftHome, ctx),
        gpui::KeyBinding::new("cmd-shift-right", ShiftEnd, ctx),
        gpui::KeyBinding::new("cmd-up", DocumentStart, ctx),
        gpui::KeyBinding::new("cmd-down", DocumentEnd, ctx),
        gpui::KeyBinding::new("cmd-shift-up", ShiftDocumentStart, ctx),
        gpui::KeyBinding::new("cmd-shift-down", ShiftDocumentEnd, ctx),
        gpui::KeyBinding::new("alt-left", WordLeft, ctx),
        gpui::KeyBinding::new("alt-right", WordRight, ctx),
        gpui::KeyBinding::new("alt-shift-left", ShiftWordLeft, ctx),
        gpui::KeyBinding::new("alt-shift-right", ShiftWordRight, ctx),
    ]);

    // Linux/Windows motion/deletion idiom: Ctrl = word (arrows + ⌫/Del),
    // Home/End = line (bound platform-neutrally above), Ctrl+Home/End =
    // document.
    #[cfg(not(target_os = "macos"))]
    bindings.extend([
        gpui::KeyBinding::new("ctrl-backspace", DeleteWordBackward, ctx),
        gpui::KeyBinding::new("ctrl-delete", DeleteWordForward, ctx),
        gpui::KeyBinding::new("ctrl-left", WordLeft, ctx),
        gpui::KeyBinding::new("ctrl-right", WordRight, ctx),
        gpui::KeyBinding::new("ctrl-shift-left", ShiftWordLeft, ctx),
        gpui::KeyBinding::new("ctrl-shift-right", ShiftWordRight, ctx),
        gpui::KeyBinding::new("ctrl-home", DocumentStart, ctx),
        gpui::KeyBinding::new("ctrl-end", DocumentEnd, ctx),
        gpui::KeyBinding::new("ctrl-shift-home", ShiftDocumentStart, ctx),
        gpui::KeyBinding::new("ctrl-shift-end", ShiftDocumentEnd, ctx),
    ]);

    cx.bind_keys(bindings);
}

#[cfg(test)]
mod tests {
    use super::{line_range_at, word_range_at};

    #[test]
    fn word_range_selects_the_alphanumeric_run() {
        let text = "the quick brown fox";
        // Click inside "quick" → the whole word, not the spaces around it.
        assert_eq!(word_range_at(text, 6), 4..9);
        assert_eq!(&text[word_range_at(text, 6)], "quick");
        // Click at the word's first byte still selects it.
        assert_eq!(&text[word_range_at(text, 4)], "quick");
        // First and last words.
        assert_eq!(&text[word_range_at(text, 0)], "the");
        assert_eq!(&text[word_range_at(text, 17)], "fox");
    }

    #[test]
    fn word_range_treats_punctuation_and_whitespace_as_delimiters() {
        let text = "scatter short (blue) wavelengths";
        let paren_open = text.find('(').unwrap();
        // Double-clicking the word inside the parens selects just "blue".
        assert_eq!(&text[word_range_at(text, paren_open + 1)], "blue");
        // Clicking the "(" selects the punctuation segment, not the word.
        assert_eq!(&text[word_range_at(text, paren_open)], "(");
        // Clicking whitespace selects the whitespace run (native behavior).
        let space = text.find(' ').unwrap();
        assert!(text[word_range_at(text, space)].chars().all(|c| c == ' '));
    }

    #[test]
    fn word_range_edges() {
        // End-of-buffer takes the trailing word; empty buffer is a caret.
        let text = "alpha";
        assert_eq!(&text[word_range_at(text, 5)], "alpha");
        assert_eq!(word_range_at("", 0), 0..0);
    }

    #[test]
    fn line_range_covers_the_paragraph_without_the_newline() {
        let text = "first line\nsecond line\nthird";
        // Anywhere on the middle line → the whole middle line, no `\n`.
        assert_eq!(&text[line_range_at(text, 15)], "second line");
        assert_eq!(&text[line_range_at(text, 11)], "second line"); // at line start
        assert_eq!(&text[line_range_at(text, 22)], "second line"); // at line end
        // First and last lines.
        assert_eq!(&text[line_range_at(text, 0)], "first line");
        assert_eq!(&text[line_range_at(text, 26)], "third");
    }

    #[test]
    fn line_range_on_a_blank_line_is_empty() {
        let text = "a\n\nb";
        assert_eq!(line_range_at(text, 2), 2..2);
    }
}

/// Counts of the non-text chrome an embed painted — the companion of
/// [`MarkdownEditorState::debug_embed_content`]'s text pieces.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EmbedChromeCounts {
    pub code_panels: usize,
    pub math: usize,
    pub rules: usize,
    pub bars: usize,
}
