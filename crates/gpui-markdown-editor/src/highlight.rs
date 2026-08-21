//! Host-supplied **highlight ranges** — the second opaque plugin surface
//! (sibling of [`crate::embed`]).
//!
//! The host hands the editor a set of source-byte ranges, each paired with an
//! opaque `u64` key, via `MarkdownEditorState::set_highlights_in`. The editor
//! paints a quiet wash behind the covered text (overlapping ranges merge
//! visually — the wash never double-darkens) and reports a click on
//! highlighted text through `MarkdownEditor::on_highlight_click` with the
//! keys of every range containing the clicked offset. The editor never
//! learns what a highlight *means* — in Eidola the key indexes the host's
//! incoming-reference list ("this passage was quoted by …"), but this crate
//! carries no such symbols.
//!
//! ## Layers
//!
//! A host with two unrelated kinds of decoration (a passage someone quoted;
//! a phrase the reader is searching for) cannot put both in one set: the
//! ranges would merge into a single wash, take a single color, and a click on
//! either would report both. So the plugin holds **one [`HighlightSet`] per
//! [`HighlightLayer`]** ([`HighlightLayers`]), and each layer is independent:
//!
//! - layers paint bottom to top in [`HighlightLayer::ALL`] order, each in its
//!   own color ([`crate::style::MarkdownStyle::highlight_layer_color`]), so an
//!   upper layer's wash sits *on top of* a lower one rather than merging with
//!   it; merging stays within a layer;
//! - **only [`HighlightLayer::Base`] routes clicks.** Every other layer is
//!   inert paint: `keys_at` is never consulted for it, so a decoration a host
//!   paints for its own reasons can never fire the host's click callback.
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

/// One of the editor's highlight layers — an ordered, independent channel of
/// host-supplied decoration (see the module docs).
///
/// Named by *stacking position and weight* rather than by meaning: the crate
/// never learns what a host paints on a layer. An enum rather than an index
/// so every layer has a color by construction and no out-of-range layer can
/// be named.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum HighlightLayer {
    /// The bottom layer, and **the only one whose ranges route clicks**
    /// through `MarkdownEditor::on_highlight_click`.
    #[default]
    Base,
    /// A quiet layer above the base.
    Overlay,
    /// The top layer, painted most prominently — for the one range a host
    /// wants to single out among many.
    Accent,
}

impl HighlightLayer {
    /// Every layer, bottom to top: the paint order.
    pub const ALL: [Self; 3] = [Self::Base, Self::Overlay, Self::Accent];

    const fn index(self) -> usize {
        self as usize
    }
}

/// The editor's highlight layers: one [`HighlightSet`] per
/// [`HighlightLayer`]. Cheap to clone; the default is every layer empty,
/// which disables the plugin.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HighlightLayers([HighlightSet; HighlightLayer::ALL.len()]);

impl HighlightLayers {
    /// The set on `layer`.
    pub fn get(&self, layer: HighlightLayer) -> &HighlightSet {
        &self.0[layer.index()]
    }

    /// Replace the set on `layer`, leaving every other layer alone.
    pub fn set(&mut self, layer: HighlightLayer, entries: HighlightSet) {
        self.0[layer.index()] = entries;
    }

    /// True when no layer carries a range.
    pub fn is_empty(&self) -> bool {
        self.0.iter().all(HighlightSet::is_empty)
    }

    /// Each non-empty layer's merged ranges, bottom to top — what the paint
    /// pass walks. Merging is per layer: an upper layer's wash overlays a
    /// lower one instead of coalescing with it.
    pub fn merged_by_layer(&self) -> Vec<(HighlightLayer, Vec<Range<usize>>)> {
        HighlightLayer::ALL
            .iter()
            .copied()
            .filter(|layer| !self.get(*layer).is_empty())
            .map(|layer| (layer, self.get(layer).merged_ranges()))
            .collect()
    }
}

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
    fn layers_are_independent_and_merge_only_within_themselves() {
        let mut layers = HighlightLayers::default();
        assert!(layers.is_empty());
        assert!(layers.merged_by_layer().is_empty());

        layers.set(HighlightLayer::Base, HighlightSet::new(vec![(0..10, 1)]));
        layers.set(HighlightLayer::Accent, HighlightSet::new(vec![(5..15, 2)]));

        assert!(!layers.is_empty());
        // Overlapping ranges on *different* layers stay separate ...
        assert_eq!(
            layers.merged_by_layer(),
            vec![
                (HighlightLayer::Base, vec![0..10]),
                (HighlightLayer::Accent, vec![5..15]),
            ]
        );
        // ... and only the base layer answers a click.
        assert_eq!(layers.get(HighlightLayer::Base).keys_at(7), vec![1]);
        assert_eq!(
            layers.get(HighlightLayer::Overlay).keys_at(7),
            Vec::<u64>::new()
        );

        // Replacing one layer leaves the others alone.
        layers.set(HighlightLayer::Accent, HighlightSet::default());
        assert_eq!(
            layers.merged_by_layer(),
            vec![(HighlightLayer::Base, vec![0..10])]
        );
    }

    #[test]
    fn layers_paint_bottom_to_top() {
        let mut layers = HighlightLayers::default();
        for layer in HighlightLayer::ALL {
            layers.set(layer, HighlightSet::new(vec![(0..4, 0)]));
        }
        let order: Vec<HighlightLayer> = layers
            .merged_by_layer()
            .into_iter()
            .map(|(layer, _)| layer)
            .collect();
        assert_eq!(order, HighlightLayer::ALL.to_vec());
    }

    #[test]
    fn identical_ranges_merge_to_one() {
        let set = HighlightSet::new(vec![(3..9, 1), (3..9, 2)]);
        assert_eq!(set.merged_ranges(), vec![3..9]);
        assert_eq!(set.keys_at(4), vec![1, 2]);
    }
}
