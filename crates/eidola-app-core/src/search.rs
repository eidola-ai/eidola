//! **Text search** — the pure matcher: no database, no I/O, no view types.
//! Free functions over `&str`, so every surface that searches user text (a
//! window, the CLI, a future cross-space index) gets identical semantics.
//!
//! ## What it matches
//!
//! Literal substring matching with **smart case**: a query in all lower case
//! matches case-insensitively, a query containing an upper-case letter matches
//! exactly. Matches are leftmost and non-overlapping.
//!
//! Case-insensitive matching folds both sides with [`fold_case`] — Rust's
//! Unicode-aware per-character lower-casing, plus Greek final sigma (`ς`) and
//! medial sigma (`σ`) folded together so `ΟΔΟΣ`, `οδος` and `οδοσ` are one
//! word to a reader searching for any of them.
//!
//! ## The one structural commitment: offsets travel through a map
//!
//! **A caller never treats an offset into transformed text as an offset into
//! the text it came from, and this module never returns one.** Every
//! transformation — the case fold here, and a caller's own rendering
//! projection — is expressed as a [`Projection`]: the transformed text plus a
//! run table that records each run's source range and its projected range
//! *separately*. Mapping back is [`Projection::source_range`].
//!
//! This is load-bearing rather than decorative, because transformations of
//! text are routinely **length-changing**:
//!
//! - lower-casing changes byte length (`İ` → `i̇`: two bytes become three),
//!   and changes it *within* a character (`ß` and `ss` are both two bytes but
//!   share no boundary), so even an equal-length run is not interpolatable;
//! - a caller projecting rendered markdown hides source spans wholesale (a
//!   link's URL, an embed marker, math source) and substitutes text for others
//!   (an entity such as `&amp;` renders as one character);
//! - the folds this module will grow — diacritic stripping, NFKC — are
//!   length-changing by construction.
//!
//! Projections compose by applying their maps in sequence: match in the
//! folded text, map back to the caller's projected text, map that back to the
//! caller's source. Each step is a [`Projection::source_range`] call.

use std::ops::Range;

/// One run of a [`Projection`]: a source span and the projected span it
/// produced. The two lengths are recorded separately — that is the whole
/// point of the structure.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Run {
    source: Range<usize>,
    projected: Range<usize>,
    kind: RunKind,
}

/// How offsets inside a run relate to offsets in its source span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunKind {
    /// The projected bytes *are* the source bytes: an offset inside the run
    /// maps by adding the same delta. Only [`ProjectionBuilder::copy`] can
    /// build one, so the equal lengths hold by construction.
    Copied,
    /// The projected bytes replace the source span as a unit: any overlap
    /// maps to the *whole* source span. A substituted entity is one atom — a
    /// match on the `&` of a rendered `&amp;` covers the five source bytes.
    Substituted,
}

/// Transformed text plus the map back to the text it was built from.
///
/// Build one with [`ProjectionBuilder`]; read the transformed text with
/// [`Projection::text`] and map any range of it back with
/// [`Projection::source_range`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Projection {
    text: String,
    runs: Vec<Run>,
}

impl Projection {
    /// The transformed text — what a matcher scans.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The source range covering `projected`: the union of the source spans of
    /// every run the projected range touches. `None` when the range is empty,
    /// inverted, or lies outside the projected text.
    ///
    /// The result **covers** the projected range rather than corresponding to
    /// it byte for byte: a range that reaches into a substituted run takes
    /// that run's whole source span, and a range spanning two runs takes the
    /// gap between them (a hidden span the projection dropped) along with
    /// them. Covering is the honest direction — the source range always
    /// contains everything the projected range showed the reader.
    pub fn source_range(&self, projected: Range<usize>) -> Option<Range<usize>> {
        if projected.start >= projected.end {
            return None;
        }
        // Runs are appended in projected order, so a binary search finds the
        // first run that can overlap.
        let first = self
            .runs
            .partition_point(|run| run.projected.end <= projected.start);
        let mut source: Option<Range<usize>> = None;
        for run in &self.runs[first..] {
            if run.projected.start >= projected.end {
                break;
            }
            let covered = match run.kind {
                RunKind::Copied => {
                    let start =
                        run.source.start + projected.start.saturating_sub(run.projected.start);
                    let end = run.source.start + projected.end.min(run.projected.end)
                        - run.projected.start;
                    start..end
                }
                RunKind::Substituted => run.source.clone(),
            };
            source = Some(match source {
                Some(acc) => acc.start.min(covered.start)..acc.end.max(covered.end),
                None => covered,
            });
        }
        source
    }
}

/// Builds a [`Projection`] of `source` by appending runs in source order.
///
/// The two appenders are the whole vocabulary, and the choice between them is
/// the choice of how offsets map:
///
/// - [`copy`](Self::copy) — the source span appears verbatim; offsets inside
///   it map one to one.
/// - [`substitute`](Self::substitute) — different text stands for the source
///   span; the span maps as one atom.
///
/// A source span that is neither copied nor substituted is simply **not
/// appended**: it contributes no projected bytes, so it can never be matched
/// (a hidden link URL, an embed marker, math source).
pub struct ProjectionBuilder<'a> {
    source: &'a str,
    projection: Projection,
}

impl<'a> ProjectionBuilder<'a> {
    pub fn new(source: &'a str) -> Self {
        Self {
            source,
            projection: Projection::default(),
        }
    }

    /// Append `source[range]` verbatim. Panics if `range` is not a character
    /// boundary range of the source — the same contract as slicing it.
    pub fn copy(&mut self, range: Range<usize>) {
        if range.start >= range.end {
            return;
        }
        let slice = &self.source[range.clone()];
        let start = self.projection.text.len();
        self.projection.text.push_str(slice);
        // Coalesce with the previous run when it is a contiguous copy, so a
        // per-character fold produces a run table proportional to the number
        // of *changed* characters rather than to the text length.
        if let Some(last) = self.projection.runs.last_mut()
            && last.kind == RunKind::Copied
            && last.source.end == range.start
            && last.projected.end == start
        {
            last.source.end = range.end;
            last.projected.end = self.projection.text.len();
            return;
        }
        self.projection.runs.push(Run {
            source: range,
            projected: start..self.projection.text.len(),
            kind: RunKind::Copied,
        });
    }

    /// Append `text` as the projection of `source[range]`. The span maps as
    /// one atom: any match touching it covers all of `range`.
    pub fn substitute(&mut self, range: Range<usize>, text: &str) {
        if text.is_empty() || range.start >= range.end {
            return;
        }
        let start = self.projection.text.len();
        self.projection.text.push_str(text);
        self.projection.runs.push(Run {
            source: range,
            projected: start..self.projection.text.len(),
            kind: RunKind::Substituted,
        });
    }

    pub fn finish(self) -> Projection {
        self.projection
    }
}

/// Fold `text` for case-insensitive matching, as a [`Projection`] back onto
/// `text`.
///
/// Per-character lower-casing (`char::to_lowercase`, so the multi-character
/// mappings of SpecialCasing apply), with the two Greek sigmas folded
/// together. Per-character rather than `str::to_lowercase` because the map is
/// built from the same walk; the sigma fold is what recovers — symmetrically,
/// in both directions — the contextual final-sigma rule that whole-string
/// lower-casing applies and a per-character walk cannot.
///
/// Unchanged characters extend a copied run; changed ones become substituted
/// runs, so a fold that changes byte length maps back correctly.
pub fn fold_case(text: &str) -> Projection {
    let mut builder = ProjectionBuilder::new(text);
    // One reused stack buffer for the folded form of the character in hand.
    // `char::to_lowercase` yields at most three scalars, so twelve bytes is the
    // widest UTF-8 a fold can produce and `encode_utf8` can never run out of
    // room. The buffer is what keeps the walk allocation-free: the matcher
    // folds a whole haystack on every case-insensitive search, so a `String`
    // per character would put one heap round trip per character in the hot
    // path.
    let mut folded = [0u8; 4 * 3];
    for (offset, ch) in text.char_indices() {
        let range = offset..offset + ch.len_utf8();
        let mut len = 0;
        for lowered in ch.to_lowercase() {
            let lowered = if lowered == 'ς' { 'σ' } else { lowered };
            len += lowered.encode_utf8(&mut folded[len..]).len();
        }
        // SAFETY-free by construction: `encode_utf8` wrote whole scalars, so
        // the prefix is valid UTF-8 — but say it with the checked conversion,
        // which optimizes to the same thing and cannot be wrong.
        let folded = std::str::from_utf8(&folded[..len]).expect("encoded scalars");
        if folded == &text[range.clone()] {
            builder.copy(range);
        } else {
            builder.substitute(range, folded);
        }
    }
    builder.finish()
}

/// A prepared search query: the needle, already folded when the query is
/// case-insensitive.
///
/// Constructing one is the only way to search, and an empty query cannot
/// produce one — so "the empty query matches everything / nothing" is not a
/// state any caller has to decide about.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Query {
    /// The needle to scan for: the raw query when case-sensitive, its
    /// [`fold_case`] projection's text when not.
    needle: String,
    case_sensitive: bool,
}

impl Query {
    /// Prepare `raw` for matching, or `None` when it is empty.
    ///
    /// **Smart case**: the query matches case-sensitively if and only if it
    /// contains an upper-case character.
    pub fn new(raw: &str) -> Option<Self> {
        if raw.is_empty() {
            return None;
        }
        let case_sensitive = raw.chars().any(char::is_uppercase);
        let needle = if case_sensitive {
            raw.to_string()
        } else {
            fold_case(raw).text
        };
        // A query of only characters that fold away cannot be scanned for.
        if needle.is_empty() {
            return None;
        }
        Some(Self {
            needle,
            case_sensitive,
        })
    }

    /// Whether smart case resolved this query to case-sensitive matching.
    pub fn is_case_sensitive(&self) -> bool {
        self.case_sensitive
    }

    /// Every match in `haystack`, leftmost and non-overlapping, as **ranges of
    /// `haystack`**.
    ///
    /// When the query is case-insensitive the scan runs over the folded text
    /// and each hit is mapped back through the fold's [`Projection`] — never
    /// reported as a folded offset. A caller that passes projected text of its
    /// own maps these ranges back through *its* projection the same way.
    pub fn find_in(&self, haystack: &str) -> Vec<Range<usize>> {
        if self.case_sensitive {
            return scan(haystack, &self.needle);
        }
        let folded = fold_case(haystack);
        scan(folded.text(), &self.needle)
            .into_iter()
            .filter_map(|hit| folded.source_range(hit))
            .collect()
    }
}

/// Leftmost, non-overlapping occurrences of `needle` (never empty) in `hay`.
fn scan(hay: &str, needle: &str) -> Vec<Range<usize>> {
    debug_assert!(!needle.is_empty());
    let mut hits = Vec::new();
    let mut from = 0;
    while let Some(index) = hay[from..].find(needle) {
        let start = from + index;
        let end = start + needle.len();
        hits.push(start..end);
        from = end;
    }
    hits
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deliberately length-changing projection, in the shape a rendered
    /// markdown projection has: a span hidden wholesale (the link URL), a
    /// span substituted by shorter text (the entity), and copied text either
    /// side. Nothing here preserves length, and the projected offsets of the
    /// tail differ from their source offsets by a wide margin.
    fn rendered(source: &str) -> Projection {
        let mut builder = ProjectionBuilder::new(source);
        let mut at = 0;
        while at < source.len() {
            if source[at..].starts_with("[") {
                // `[text](url)` → `text`, with the URL contributing nothing.
                let close = at + source[at..].find(']').expect("link text closes");
                let end = at + source[at..].find(')').expect("link closes") + 1;
                builder.copy(at + 1..close);
                at = end;
            } else if source[at..].starts_with("&amp;") {
                builder.substitute(at..at + "&amp;".len(), "&");
                at += "&amp;".len();
            } else {
                let next = source[at + 1..]
                    .find(['[', '&'])
                    .map(|i| at + 1 + i)
                    .unwrap_or(source.len());
                builder.copy(at..next);
                at = next;
            }
        }
        builder.finish()
    }

    #[test]
    fn a_length_changing_projection_survives_the_round_trip() {
        let source = "See [the report](https://reports.example/perf) &amp; the notes.";
        let projection = rendered(source);
        assert_eq!(projection.text(), "See the report & the notes.");
        // The premise of the whole structure: the projection is shorter than
        // its source, so projected offsets are not source offsets.
        assert_ne!(projection.text().len(), source.len());

        let query = Query::new("the notes").expect("non-empty");
        let hits = query.find_in(projection.text());
        assert_eq!(hits.len(), 1);

        let mapped = projection
            .source_range(hits[0].clone())
            .expect("a hit maps back");
        assert_eq!(&source[mapped], "the notes");
    }

    #[test]
    fn a_projected_offset_used_as_a_source_offset_points_at_the_wrong_text() {
        // The failure the map exists to prevent, pinned so the round-trip
        // test above is visibly doing work: the naive reading of a hit —
        // "projected offsets are source offsets" — lands mid-word.
        let source = "See [the report](https://reports.example/perf) &amp; the notes.";
        let projection = rendered(source);
        let query = Query::new("the notes").expect("non-empty");
        let hit = query.find_in(projection.text()).remove(0);

        let naive = &source[hit.clone()];
        assert_ne!(naive, "the notes");
        assert_eq!(naive, "https://r");

        let mapped = projection.source_range(hit).expect("a hit maps back");
        assert_eq!(&source[mapped], "the notes");
    }

    #[test]
    fn a_match_inside_a_substitution_covers_the_whole_source_atom() {
        let source = "Tom &amp; Jerry";
        let projection = rendered(source);
        assert_eq!(projection.text(), "Tom & Jerry");

        let query = Query::new("m & j").expect("non-empty");
        let hit = query.find_in(projection.text()).remove(0);
        let mapped = projection.source_range(hit).expect("a hit maps back");
        // The `&` is one atom: its five source bytes come along whole.
        assert_eq!(&source[mapped], "m &amp; J");
    }

    #[test]
    fn a_match_spanning_a_hidden_span_covers_it() {
        let source = "the [alpha](https://example/x)beta tail";
        let projection = rendered(source);
        assert_eq!(projection.text(), "the alphabeta tail");

        let query = Query::new("alphabeta").expect("non-empty");
        let hit = query.find_in(projection.text()).remove(0);
        let mapped = projection.source_range(hit).expect("a hit maps back");
        // Covering, not corresponding: the hidden URL sits between the two
        // matched runs, so the source range contains it.
        assert_eq!(&source[mapped], "alpha](https://example/x)beta");
    }

    #[test]
    fn case_folding_is_itself_length_changing_and_maps_back() {
        // U+1E9E LATIN CAPITAL LETTER SHARP S lower-cases to `ß` — three
        // bytes become two, so the folded text is shorter than its source and
        // every offset after the change is displaced.
        let source = "Die GROẞE Halle";
        let folded = fold_case(source);
        assert_eq!(folded.text(), "die große halle");
        assert!(folded.text().len() < source.len());

        let hits = Query::new("große halle")
            .expect("non-empty")
            .find_in(source);
        assert_eq!(hits.len(), 1);
        assert_eq!(&source[hits[0].clone()], "GROẞE Halle");
    }

    #[test]
    fn folding_is_case_only_today_so_a_combining_mark_still_separates() {
        // U+0130 lower-cases to `i` + U+0307 COMBINING DOT ABOVE, so a reader
        // searching `istanbul` does not find `İstanbul`. That is a
        // *normalization* gap, not a mapping one: closing it is the deferred
        // diacritic/NFKC fold, which composes into `fold_case` as more
        // substituted runs and needs no change to the map.
        let folded = fold_case("İstanbul");
        assert_eq!(folded.text(), "i\u{307}stanbul");
        assert!(
            Query::new("istanbul")
                .expect("non-empty")
                .find_in("İstanbul")
                .is_empty()
        );
        // What the map does guarantee: the hit that *is* found maps back onto
        // the original capital, length change and all.
        let hits = Query::new("i\u{307}stanbul")
            .expect("non-empty")
            .find_in("İstanbul");
        assert_eq!(hits.len(), 1);
        assert_eq!(&"İstanbul"[hits[0].clone()], "İstanbul");
    }

    #[test]
    fn smart_case_is_insensitive_until_the_query_shifts() {
        let hay = "Turbulence and turbulence and TURBULENCE";

        let any = Query::new("turbulence").expect("non-empty");
        assert!(!any.is_case_sensitive());
        assert_eq!(any.find_in(hay).len(), 3);

        let exact = Query::new("Turbulence").expect("non-empty");
        assert!(exact.is_case_sensitive());
        let hits = exact.find_in(hay);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0], 0..10);
    }

    #[test]
    fn both_greek_sigmas_fold_together() {
        // Whole-string lower-casing turns a final Σ into ς; a per-character
        // walk turns it into σ. Folding the two together makes every spelling
        // findable from every other.
        for hay in ["ΟΔΟΣ", "οδος", "οδοσ"] {
            for needle in ["οδος", "οδοσ"] {
                let hits = Query::new(needle).expect("non-empty").find_in(hay);
                assert_eq!(hits.len(), 1, "{needle} in {hay}");
                assert_eq!(hits[0], 0..hay.len());
            }
        }
    }

    #[test]
    fn matches_are_leftmost_and_non_overlapping() {
        let hits = Query::new("aa").expect("non-empty").find_in("aaaaa");
        assert_eq!(hits, vec![0..2, 2..4]);
    }

    #[test]
    fn multibyte_text_reports_character_boundary_ranges() {
        let hay = "καλημέρα κόσμε";
        let hits = Query::new("κόσμε").expect("non-empty").find_in(hay);
        assert_eq!(hits.len(), 1);
        assert_eq!(&hay[hits[0].clone()], "κόσμε");
    }

    #[test]
    fn an_empty_query_cannot_be_built() {
        assert!(Query::new("").is_none());
    }

    #[test]
    fn a_query_longer_than_the_text_matches_nothing() {
        let query = Query::new("a longer phrase").expect("non-empty");
        assert!(query.find_in("short").is_empty());
    }

    #[test]
    fn mapping_rejects_empty_and_out_of_range_projected_ranges() {
        let projection = rendered("plain text");
        assert_eq!(projection.source_range(0..0), None);
        assert_eq!(projection.source_range(Range { start: 5, end: 2 }), None);
        assert_eq!(projection.source_range(50..60), None);
    }

    #[test]
    fn a_projection_of_only_hidden_text_matches_nothing() {
        let source = "[label](https://example/only)";
        let projection = rendered(source);
        assert_eq!(projection.text(), "label");
        assert!(
            Query::new("https")
                .expect("non-empty")
                .find_in(projection.text())
                .is_empty()
        );
    }
}
