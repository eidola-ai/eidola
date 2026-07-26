//! Host-supplied **highlight ranges** — the second opaque plugin surface
//! (sibling of [`crate::embed`]).
//!
//! The host hands the editor a set of source-byte ranges, each paired with an
//! opaque `u64` key, via `MarkdownEditorState::set_highlights`. The editor
//! paints a quiet wash behind the covered text (overlapping ranges merge
//! visually — the wash never double-darkens) and reports a click on
//! highlighted text through `MarkdownEditor::on_highlight_click` with the
//! keys of every range containing the clicked offset. The editor never
//! learns what a highlight *means* — in Eidola the key indexes the host's
//! incoming-reference list ("this passage was quoted by …"), but this crate
//! carries no such symbols.
//!
//! Highlights are **inert decorations**: they are not document content
//! (`value()` and copy never see them), they create no forbidden caret
//! positions, and they never affect editing, selection, or the canonicalizer.
//! A selection drag across highlighted text selects normally; only a plain
//! single click (press and release with no range created) fires the click
//! callback.

use std::ops::Range;
use std::sync::Arc;

/// One host-supplied highlight: the source-byte range it covers and the
/// opaque key reported back on a click.
pub type HighlightEntry = (Range<usize>, u64);

/// The host-supplied set of [`HighlightEntry`] pairs. Cheap to clone (shared
/// `Arc`); an empty set (the default) disables the plugin.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HighlightSet(Option<Arc<Vec<HighlightEntry>>>);

impl HighlightSet {
    /// Build a set from `(range, key)` pairs. Empty and inverted ranges are
    /// dropped (they cover nothing and can never be clicked).
    pub fn new(entries: impl IntoIterator<Item = HighlightEntry>) -> Self {
        let entries: Vec<HighlightEntry> = entries
            .into_iter()
            .filter(|(r, _)| r.start < r.end)
            .collect();
        if entries.is_empty() {
            Self(None)
        } else {
            Self(Some(Arc::new(entries)))
        }
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    /// The keys of every range containing `offset` (`start <= offset < end`),
    /// in insertion order — what the click callback reports. One key means
    /// the host can act directly; several mean the clicked text is covered by
    /// several overlapping ranges (the host disambiguates).
    pub fn keys_at(&self, offset: usize) -> Vec<u64> {
        let Some(entries) = self.0.as_ref() else {
            return Vec::new();
        };
        entries
            .iter()
            .filter(|(r, _)| r.start <= offset && offset < r.end)
            .map(|(_, k)| *k)
            .collect()
    }

    /// The visual union of all ranges: sorted, non-overlapping (adjacent
    /// ranges coalesce). The paint pass draws one wash quad set per merged
    /// range, so overlapping highlights never stack alpha into a darker band.
    pub fn merged_ranges(&self) -> Vec<Range<usize>> {
        let Some(entries) = self.0.as_ref() else {
            return Vec::new();
        };
        let mut ranges: Vec<Range<usize>> = entries.iter().map(|(r, _)| r.clone()).collect();
        ranges.sort_by_key(|r| (r.start, r.end));
        let mut merged: Vec<Range<usize>> = Vec::with_capacity(ranges.len());
        for r in ranges {
            match merged.last_mut() {
                Some(last) if r.start <= last.end => last.end = last.end.max(r.end),
                _ => merged.push(r),
            }
        }
        merged
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_inverted_ranges_are_dropped() {
        // `9..3` written structurally: a reversed literal is a clippy deny,
        // but an inverted range is exactly what this test feeds the filter.
        let inverted = Range { start: 9, end: 3 };
        let set = HighlightSet::new(vec![(5..5, 1), (inverted, 2)]);
        assert!(set.is_empty());
        assert!(set.keys_at(5).is_empty());
        assert!(set.merged_ranges().is_empty());
    }

    #[test]
    fn keys_at_reports_every_containing_range_in_order() {
        let set = HighlightSet::new(vec![(0..10, 7), (5..15, 3), (20..25, 9)]);
        assert_eq!(set.keys_at(0), vec![7]);
        assert_eq!(set.keys_at(7), vec![7, 3]);
        assert_eq!(set.keys_at(12), vec![3]);
        // End-exclusive: the byte at `end` is outside.
        assert_eq!(set.keys_at(15), Vec::<u64>::new());
        assert_eq!(set.keys_at(22), vec![9]);
    }

    #[test]
    fn merged_ranges_union_overlaps_and_adjacency() {
        let set = HighlightSet::new(vec![(20..25, 1), (0..10, 2), (5..15, 3), (15..18, 4)]);
        // 0..10 ∪ 5..15 ∪ 15..18 coalesce (overlap + adjacency); 20..25 stands.
        assert_eq!(set.merged_ranges(), vec![0..18, 20..25]);
    }

    #[test]
    fn identical_ranges_merge_to_one() {
        let set = HighlightSet::new(vec![(3..9, 1), (3..9, 2)]);
        assert_eq!(set.merged_ranges(), vec![3..9]);
        assert_eq!(set.keys_at(4), vec![1, 2]);
    }
}
