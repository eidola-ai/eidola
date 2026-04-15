//! Editor state — markdown source plus selection. UTF-8 byte offsets.

use std::ops::Range;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditorState {
    pub markdown: String,
    pub selection: Selection,
}

impl EditorState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_markdown(markdown: impl Into<String>) -> Self {
        Self {
            markdown: markdown.into(),
            selection: Selection::Cursor(0),
        }
    }
}

impl Default for EditorState {
    fn default() -> Self {
        Self {
            markdown: String::new(),
            selection: Selection::Cursor(0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Selection {
    Cursor(usize),
    Range { anchor: usize, head: usize },
}

impl Selection {
    pub fn cursor(offset: usize) -> Self {
        Selection::Cursor(offset)
    }

    pub fn range(anchor: usize, head: usize) -> Self {
        Selection::Range { anchor, head }
    }

    pub fn head(&self) -> usize {
        match *self {
            Selection::Cursor(p) => p,
            Selection::Range { head, .. } => head,
        }
    }

    pub fn anchor(&self) -> usize {
        match *self {
            Selection::Cursor(p) => p,
            Selection::Range { anchor, .. } => anchor,
        }
    }

    pub fn is_collapsed(&self) -> bool {
        match *self {
            Selection::Cursor(_) => true,
            Selection::Range { anchor, head } => anchor == head,
        }
    }

    pub fn lower_bound(&self) -> usize {
        match *self {
            Selection::Cursor(p) => p,
            Selection::Range { anchor, head } => anchor.min(head),
        }
    }

    pub fn upper_bound(&self) -> usize {
        match *self {
            Selection::Cursor(p) => p,
            Selection::Range { anchor, head } => anchor.max(head),
        }
    }

    pub fn selection_range(&self) -> Range<usize> {
        self.lower_bound()..self.upper_bound()
    }
}

impl Default for Selection {
    fn default() -> Self {
        Selection::Cursor(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_helpers() {
        let sel = Selection::cursor(5);
        assert_eq!(sel.head(), 5);
        assert_eq!(sel.anchor(), 5);
        assert!(sel.is_collapsed());
        assert_eq!(sel.selection_range(), 5..5);
    }

    #[test]
    fn range_left_to_right() {
        let sel = Selection::range(2, 7);
        assert_eq!(sel.head(), 7);
        assert_eq!(sel.anchor(), 2);
        assert_eq!(sel.selection_range(), 2..7);
        assert!(!sel.is_collapsed());
    }

    #[test]
    fn range_right_to_left_normalizes_bounds() {
        let sel = Selection::range(10, 4);
        assert_eq!(sel.lower_bound(), 4);
        assert_eq!(sel.upper_bound(), 10);
    }
}
