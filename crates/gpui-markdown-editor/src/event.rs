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

    ExtendLeft,
    ExtendRight,
    ExtendUp,
    ExtendDown,
    ExtendLineStart,
    ExtendLineEnd,
    ExtendDocumentStart,
    ExtendDocumentEnd,
}
