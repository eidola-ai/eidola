//! User actions that can mutate `EditorState`. Routed by `editor.rs` from
//! gpui actions and IME callbacks; consumed by `update::update`.

use crate::state::Selection;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditorEvent {
    InsertText(String),
    InsertNewline,
    InsertLineBreak,
    DeleteBackward,
    DeleteForward,
    SetSelection(Selection),

    /// Increase the nesting level of the list item containing the
    /// cursor. No-op if the cursor isn't inside a list item, or if
    /// the item has no previous sibling at the same depth (since
    /// there'd be nothing to nest under).
    IncreaseListDepth,
    /// Decrease the nesting level of the list item containing the
    /// cursor. For a top-level item, drops the marker bytes
    /// (item becomes a paragraph). For a nested item, removes the
    /// parent item's marker-width worth of leading spaces from
    /// every line of the item, so the item becomes a sibling of
    /// its former parent. No-op outside of a list.
    DecreaseListDepth,

    MoveLeft,
    MoveRight,
    MoveUp,
    MoveDown,
    MoveLineStart,
    MoveLineEnd,
    MoveDocumentStart,
    MoveDocumentEnd,

    /// Move the cursor to the start of the previous word per Unicode
    /// word-boundary rules. Skips whitespace and punctuation, lands at
    /// the leading byte of the previous alphanumeric-containing
    /// segment. Standard macOS Option+Left.
    MoveWordLeft,
    /// Symmetric: move to the end of the next word. Standard macOS
    /// Option+Right.
    MoveWordRight,

    ExtendLeft,
    ExtendRight,
    ExtendUp,
    ExtendDown,
    ExtendLineStart,
    ExtendLineEnd,
    ExtendDocumentStart,
    ExtendDocumentEnd,

    /// Extend the selection to the start of the previous word.
    /// Selection-extension variant of [`MoveWordLeft`].
    ExtendWordLeft,
    /// Extend the selection to the end of the next word.
    ExtendWordRight,

    /// Delete the byte range from the start of the previous word
    /// through the cursor. Standard macOS Option+Backspace. With a
    /// non-collapsed selection, deletes the selection instead.
    ///
    /// Inside a structural chain (BQ paragraph, list item, nested
    /// combination) the word-target is clamped to the line's
    /// chain-prefix end so the `> ` / `- ` / continuation-indent bytes
    /// survive — the user's deletion only affects content, not
    /// structure. At top level no clamp applies and the word walk can
    /// cross `\n` naturally.
    DeleteWordBackward,
    /// Delete the byte range from the cursor through the end of the
    /// next word. Standard macOS Option+Delete (fn+Backspace).
    ///
    /// When the word-target would spill onto a line whose chain
    /// prefix it would destroy, the target is clamped to the cursor's
    /// line end (no `\n` crossing). Top-level next lines have no
    /// chain prefix, so word-delete crosses `\n` and consumes the
    /// next word naturally.
    DeleteWordForward,
    /// Delete from the cursor backward to the *visible content edge*
    /// of the current line — past every byte the renderer paints as
    /// chain chrome (BQ markers, list-item markers on the marker
    /// line, list-continuation indent on continuation lines) but
    /// before any user content. Standard macOS Cmd+Backspace, adapted
    /// so structural markers in nested scopes survive. At top level
    /// the floor degenerates to the raw line start (no prefix to
    /// preserve). No-op when the cursor already sits at the content
    /// edge.
    DeleteToLineStart,
    /// Delete from the cursor forward to the end of the current line
    /// (the byte right before the trailing `\n`, if any). Standard
    /// macOS Cmd+fn+Backspace / Ctrl+K. No-op at end of line.
    DeleteToLineEnd,
}
