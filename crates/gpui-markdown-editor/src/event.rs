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
