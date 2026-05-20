//! WYSIWYG markdown editor as a `gpui-component`-style widget.
//!
//! See `AGENTS.md` for the design rationale and target behavior. The
//! pipeline is:
//!
//! ```text
//! EditorState + EditorEvent  →  update()  →  new EditorState
//!                                                  ↓
//!                                              parse()  →  SyntaxTree
//!                                                  ↓
//!                                              render(state, tree) → RenderSpec
//!                                                  ↓
//!                                              BlockElement (gpui Element, one per block)
//! ```

pub mod analysis;
pub mod editor;
pub mod element;
pub mod escapes;
pub mod event;
pub mod image;
pub mod math;
pub mod parser;
pub mod render;
pub mod render_spec;
pub mod state;
pub mod style;
pub mod syntax;
pub mod update;

pub use editor::{
    Backspace, Copy, Cut, Delete, DeleteToLineEnd, DeleteToLineStart, DeleteWordBackward,
    DeleteWordForward, DocumentEnd, DocumentStart, Down, End, Enter, Home, Left, MarkdownEditor,
    Paste, Right, SelectAll, ShiftDocumentEnd, ShiftDocumentStart, ShiftDown, ShiftEnd, ShiftEnter,
    ShiftHome, ShiftLeft, ShiftRight, ShiftTab, ShiftUp, ShiftWordLeft, ShiftWordRight, Tab, Up,
    WordLeft, WordRight,
};
pub use event::EditorEvent;
pub use parser::parse;
pub use render::render;
pub use render_spec::{
    BlockKind, Container, InlineRun, InlineStyle, ListItemKind, RenderBlock, RenderSpec,
    Substitution,
};
pub use state::{EditorState, Selection};
pub use style::MarkdownStyle;
pub use syntax::{NodeKind, SyntaxNode};
pub use update::update;
