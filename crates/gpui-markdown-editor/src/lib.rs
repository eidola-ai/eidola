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

pub mod editor;
pub mod element;
pub mod event;
pub mod parser;
pub mod render;
pub mod render_spec;
pub mod state;
pub mod style;
pub mod syntax;
pub mod update;

pub use editor::{
    Backspace, Copy, Cut, Delete, DocumentEnd, DocumentStart, Down, End, Enter, Home, Left,
    MarkdownEditor, Paste, Right, SelectAll, ShiftDocumentEnd, ShiftDocumentStart, ShiftDown,
    ShiftEnd, ShiftEnter, ShiftHome, ShiftLeft, ShiftRight, ShiftUp, Up,
};
pub use event::EditorEvent;
pub use parser::parse;
pub use render::render;
pub use render_spec::{BlockKind, Container, InlineRun, InlineStyle, RenderBlock, RenderSpec};
pub use state::{EditorState, Selection};
pub use style::MarkdownStyle;
pub use syntax::{NodeKind, SyntaxNode};
pub use update::update;
