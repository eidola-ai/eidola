//! **Find in the conversation** — ⌘F over the visible branch.
//!
//! Three pieces live here: the *searchable projection* of a post (what
//! read-only rendering actually shows), the [`FindSession`] a window holds
//! while the bar is open, and the bar itself.
//!
//! ## Why a projection at all
//!
//! A post's body is markdown **source**, and the editor's highlight plugin
//! takes **source** byte offsets — so the obvious implementation, scanning the
//! source, produces a count that lies. A read-only editor hides every
//! delimiter, and bytes inside a hidden range contribute no display bytes at
//! all: searching `performance` inside `[perf](https://performance.example)`
//! would count a match the reader can never see, and paint it as a zero-width
//! quad. In the other direction, a phrase crossing an emphasis delimiter
//! (`very **important** thing` searched for `important thing`) plainly matches
//! what the reader sees and does not match the source.
//!
//! So [`searchable_projection`] renders the post exactly as the transcript
//! does (`render_readonly`) and appends what that render leaves visible,
//! through [`eidola_app_core::search::ProjectionBuilder`] — which records each
//! run's source and projected lengths separately, so every match maps back to
//! a **source** range. The result: the count equals what is highlighted,
//! matches cross inline markup, and matches inside hidden syntax neither count
//! nor paint.
//!
//! What is deliberately not searchable, because it has no display bytes: a
//! link's URL, the source of math that typesets, an image's markup and alt
//! text, and an embed marker. Math whose LaTeX *fails* to typeset is the
//! exception that proves the rule — the reader is shown its raw `$…$` bytes,
//! so those bytes are searchable, and matching them needs nothing special
//! because they are already the source. Embedded quoted text is not
//! searchable either — it is re-parsed
//! standalone by the element layer and is not in the parent document's offset
//! space, so no source range in this post could name it.
//!
//! ## Paying only when find is used
//!
//! Building a projection costs a parse and a render pass, so the cache lives
//! **inside** [`FindSession`]: no session, no cache, and closing the bar drops
//! every projection with it. That is an invariant rather than an observation,
//! and it is structural — there is nowhere else for a projection to be kept.
//! [`SpaceView::projections_built_for_test`] is what lets a test see that none
//! was *built* either, which no amount of looking at the cache could show.

use std::collections::{HashMap, HashSet};
use std::ops::Range;

use eidola_app_core::search::{Projection, ProjectionBuilder, Query};
use gpui::{
    AnyElement, AppContext, Context, InteractiveElement, IntoElement, ParentElement, Pixels,
    SharedString, StatefulInteractiveElement, Styled, Window, div, px,
};
use gpui_component::{ActiveTheme, h_flex};
use gpui_markdown_editor::{EmbedMap, Selection};

use super::layout::{GutterPlacement, compact_gutter_occupancy, page_layout};
use super::model::{NodeSrc, TreeNode};
use super::{POST_PAD_Y, SpaceView, TITLE_BAR_RESERVE};
use crate::focus::TabRegion;
use crate::overlay::{Contain, Overlay};
use crate::probe::Probe;

/// The searchable text of one post, with the map back to its markdown source.
///
/// Built from the same `render_readonly` pass the transcript paints with, so
/// what is scanned is what the reader can see. Every span the render hides
/// contributes nothing, and every span it substitutes (a backslash escape, an
/// entity reference) is appended as its *displayed* text mapped to the whole
/// source atom — a match on the `&` of a rendered `&amp;` reports all five
/// source bytes.
///
/// The `embeds` argument matters: a `{{ embed N }}` marker is hidden wholesale
/// only when its ordinal is **mapped**, and is ordinary literal text when it
/// is not. Passing the post's own map is what keeps the projection agreeing
/// with the post's own editor.
/// A node's searchable text **and the blocks the render laid it out in**.
///
/// The two come out of one pass because they are two readings of the same
/// render: the projection is what a query is scanned against, and the block
/// ranges are the units the Find-all overlay cuts a *fragment* out of (see
/// [`super::find_overlay`]). Deriving the blocks a second time would be a
/// second parse of every post, and — worse — a second answer to "what does this
/// post render as", which is exactly the drift the projection exists to remove.
pub(crate) struct NodeProjection {
    pub(crate) projection: Projection,
    /// Each rendered block's source range, in document order.
    pub(crate) blocks: Vec<Range<usize>>,
}

impl NodeProjection {
    /// The source ranges `query` matches — the projection's own answer.
    pub(crate) fn find(&self, query: &Query) -> Vec<Range<usize>> {
        self.projection.find(query)
    }
}

pub(crate) fn searchable_projection(
    content: &str,
    embeds: &EmbedMap,
    cursor: Option<Selection>,
) -> NodeProjection {
    let mut state = gpui_markdown_editor::EditorState::with_markdown(content);
    state.embeds = embeds.clone();
    let tree = gpui_markdown_editor::parse(&state.markdown);
    // **The render mode is the node's, not a constant.** A node the reader is
    // *editing* — an inline edit, any draft — keeps an enabled editor, and an
    // enabled editor renders cursor-aware: the delimiters and the link URL its
    // cursor sits on are revealed, on the page, in front of them. Projecting
    // that node read-only searched a different document than the one being
    // shown, in both directions — the exposed `https://…` could not match, and
    // a phrase that only closes up once the delimiters hide matched text that
    // no longer reads that way.
    let spec = match cursor {
        Some(selection) => {
            state.selection = selection;
            gpui_markdown_editor::render::render(&state, &tree)
        }
        None => gpui_markdown_editor::render::render_readonly(&state, &tree),
    };

    let mut builder = ProjectionBuilder::new(content);
    let mut blocks: Vec<Range<usize>> = Vec::new();
    let mut prev_end: Option<usize> = None;
    for block in &spec.blocks {
        let block_range = clamp(&block.source_range, content.len());
        if block_range.start >= block_range.end {
            continue;
        }
        blocks.push(block_range.clone());
        // **A barrier between blocks.** Two adjacent paragraphs are two
        // separate things on the page, so a query must not match across the
        // gap between them — but the gap's bytes are not a run of their own,
        // and a fabricated separator would be projected text mapping to no
        // source. So one real newline out of the gap is copied instead: it is
        // a byte the source has, at a place the reader really does see a line
        // break, and a find query typed into a one-line field can never
        // contain one.
        if let Some(prev) = prev_end
            && prev <= block_range.start
            && let Some(offset) = content[prev..block_range.start].find('\n')
        {
            builder.copy(prev + offset..prev + offset + 1);
        }
        append_block(&mut builder, content, block, block_range.clone());
        prev_end = Some(block_range.end.max(prev_end.unwrap_or(0)));
    }
    NodeProjection {
        projection: builder.finish(),
        blocks,
    }
}

/// What the walk over one block's source finds at a given byte.
enum Mark<'a> {
    /// Bytes the render replaces with display text of its own.
    Substitute(&'a str),
    /// Bytes the render shows nothing for.
    Skip,
}

/// What source bytes project as when they stand between two things the reader
/// sees apart — the block gap's newline, reached by every other door.
///
/// **Not every excluded span is zero-width formatting**, and there are two
/// ways to reach the mistake of treating one as if it were:
///
/// - **A table's grid chrome.** A read-only table hides the pipes and padding
///   between cells, so deleting them wholesale concatenated the visible cell
///   texts and let `| left | right |` match `leftright` — a phrase occupying
///   two cells the reader plainly sees apart.
/// - **An inline overlay the element layer really replaces.** Typeset math and
///   an inline image are *atoms on the page*, each occupying width between the
///   text either side of it, so deleting their source zero-width let
///   `left$x$right` match `leftright` — again a phrase the reader sees in two
///   pieces, with a rendered thing standing between them.
///
/// Both spans are *structural*, the way a paragraph gap is, so all three
/// project as the same thing: one newline, which a query typed into a one-line
/// field can never contain.
///
/// It is a **substitution**, not a fabricated run: the newline stands for the
/// span's own source bytes, so the projected text still maps back to a range
/// of the post — the one rule [`ProjectionBuilder`] exists to keep. (Nothing
/// can ever match *into* it, so the atom-coverage semantics never come up.)
const BARRIER: &str = "\n";

fn append_block(
    builder: &mut ProjectionBuilder<'_>,
    content: &str,
    block: &gpui_markdown_editor::RenderBlock,
    block_range: Range<usize>,
) {
    // **A sole-image paragraph is the promoted block form, and it hides
    // itself one layer further down than every other hide.** The render
    // layer emits `BlockKind::Image` with no `hidden_ranges` and no overlay
    // at all; the element layer pushes the hide over the whole block once the
    // image is loading or loaded — every case that paints a picture. Reading
    // the render spec alone therefore found nothing to exclude and copied the
    // entire `![alt](url)`, reporting matches on an alt text and a URL that
    // stand behind the image rather than on the page.
    //
    // **Load state is deliberately not consulted**, which is the one place
    // this differs from the math rule beside it. A failed load is the single
    // case that leaves the markup shaped, but whether an image loads is not a
    // function of the post's source: it is asynchronous, external, and can
    // change with no edit behind it. Consulting it would put an input in the
    // projection cache's validity key that no `ProjectionSeed` could honestly
    // carry, and would need a `Window` the projection does not have. So both
    // image forms take the same verdict the inline overlay already took, and
    // the admitted cost is that a broken image's visible markup is not
    // searchable.
    if matches!(
        block.kind,
        gpui_markdown_editor::BlockKind::Image {
            edit_mode: false,
            ..
        }
    ) {
        return;
    }

    let mut marks: Vec<(Range<usize>, Mark<'_>)> = Vec::new();
    // Every substitution's start, in order — the boundaries a hidden range
    // has to be interrupted at. See [`push_hide`].
    let mut sub_starts: Vec<usize> = Vec::new();
    for sub in &block.substitutions {
        let r = clamp(&sub.source_range, content.len());
        if r.start < r.end {
            sub_starts.push(r.start);
            marks.push((r, Mark::Substitute(sub.display.as_str())));
        }
    }
    sub_starts.sort_unstable();
    // **Which hidden bytes are a barrier rather than a deletion**, decided by
    // where they fall rather than by re-deriving the render's own chrome
    // arithmetic: a table's cell content ranges come out of the same
    // `geometry` the render laid the grid from, so a hidden byte *inside* a
    // cell is inline markup (a link's brackets, an emphasis run) and one
    // outside every cell is the grid itself — the leading `| `, each ` | `,
    // the trailing ` |`, the whole delimiter row. Non-table blocks have no
    // cells and take the ordinary path unchanged.
    //
    // **Per byte, not per range**, because a hidden range can straddle a cell
    // edge: `merge_hidden_ranges` joins an entity's own hidden bytes to the
    // chrome beside them, and one verdict over the whole of that either drops
    // the `&` the reader sees or drops the boundary between two cells.
    // Splitting at the cell edges is what makes the two questions independent
    // again, and it is the same rule stated at the granularity it is true at.
    let cells = table_cells(block);
    for hidden in &block.hidden_ranges {
        let r = clamp(hidden, content.len());
        if r.start >= r.end {
            continue;
        }
        if cells.is_empty() {
            push_hide(r, &sub_starts, &mut marks);
            continue;
        }
        split_at_cell_edges(r, &cells, &sub_starts, &mut marks);
    }
    // **Inline math and inline images carry no hidden range of their own.**
    // The render layer deliberately leaves suppressing their source bytes to
    // the element layer, which does it differently per typeset outcome — so a
    // projection reading `hidden_ranges` alone would make a URL, an alt text
    // and a `\frac` matchable. Only the *promoted block* forms (a sole-image
    // paragraph, a `$$…$$` block) hide themselves.
    //
    // **Math is skipped only where math is what the reader gets.** The
    // element layer substitutes a width-matched pad run and paints typeset
    // math over it *when the LaTeX typesets*; when it does not, the raw
    // `$…$` shapes as itself — dim delimiters, mono content — and the reader
    // is looking at the source bytes. Skipping those would report no match
    // and paint no highlight on text plainly on screen, so the overlay is
    // excluded only when [`gpui_markdown_editor::math_overlay_typesets`]
    // agrees the math exists. Leaving a failed one unmarked is all the
    // projection has to do: its bytes are the visible glyphs, so the walk
    // copies them like any other run and the source-range rule holds without
    // a substitution.
    //
    // **And what the element layer puts in an overlay's place is an atom, not
    // a deletion.** An overlay it really replaces occupies width on the page,
    // so the text either side of it is two things the reader sees apart:
    // `left$x$right` reads as `left`, a formula, `right`. Excluding the span
    // zero-width projected `leftright`, and counted — and painted — a match on
    // a phrase that is nowhere on the page. That is the grid chrome's mistake
    // reached by a third door, so these spans project as the same [`BARRIER`],
    // substituted for their own source bytes.
    for math in &block.math_overlays {
        let r = clamp(&math.source_range, content.len());
        if r.start < r.end && gpui_markdown_editor::math_overlay_typesets(block, content, math) {
            marks.push((r, Mark::Substitute(BARRIER)));
        }
    }
    for image in &block.image_overlays {
        let r = clamp(&image.source_range, content.len());
        if r.start < r.end {
            marks.push((r, Mark::Substitute(BARRIER)));
        }
    }
    // Substitutions before skips at the same start: an escape or an entity is
    // recorded as both (hidden bytes, displayed replacement), and the
    // replacement is what the reader sees.
    marks.sort_by_key(|(r, mark)| (r.start, matches!(mark, Mark::Skip), r.end));

    let mut pos = block_range.start;
    let mut next = 0usize;
    while pos < block_range.end {
        // Drop marks the walk has already passed.
        while next < marks.len() && marks[next].0.end <= pos {
            next += 1;
        }
        let Some((range, mark)) = marks.get(next) else {
            builder.copy(pos..block_range.end);
            break;
        };
        if range.start >= block_range.end {
            builder.copy(pos..block_range.end);
            break;
        }
        if range.start > pos {
            builder.copy(pos..range.start);
            pos = range.start;
            continue;
        }
        // The mark covers `pos`. A substitution is appended whole (its source
        // span is one atom); a hidden span contributes nothing at all.
        if let Mark::Substitute(display) = mark
            && range.start == pos
        {
            builder.substitute(range.clone(), display);
        }
        pos = range.end.min(block_range.end).max(pos);
        next += 1;
    }
}

fn clamp(range: &Range<usize>, len: usize) -> Range<usize> {
    range.start.min(len)..range.end.min(len)
}

/// Push one hidden range as [`Mark::Skip`], split at every substitution that
/// begins strictly inside it so that substitution still gets its turn.
///
/// **This is the display walker's rule, not a new one.** `build_display_line`
/// clamps a covering hide's jump to the earliest in-span substitution start,
/// so the reader of `**&amp;**` sees `&` even though `merge_hidden_ranges`
/// coalesced the emphasis delimiters and the entity into a single hide
/// starting at byte 0. The walk below applies a substitution only where it
/// lands on the start exactly, so an un-split covering hide consumed the whole
/// span first and that `&` projected as nothing — invisible to a search over
/// text plainly on the page. Splitting at the start is enough: the piece that
/// begins there sorts after the substitution, which takes the byte and hands
/// the rest of the hide back.
///
/// Splitting rather than teaching the walk a special case keeps the per-byte
/// philosophy the cell-edge rule already states — each piece of a merged hide
/// carries the verdict its own bytes earn — and it composes with that rule
/// instead of racing it, since a cell-edge piece is pushed through here too.
///
/// This says nothing about the *overlay* skips, and must not: a math or image
/// overlay is replaced wholesale by the element layer, so a substitution
/// inside one (an entity in an image's alt text) never reaches the page and
/// must stay unsearchable.
fn push_hide<'a>(
    range: Range<usize>,
    sub_starts: &[usize],
    out: &mut Vec<(Range<usize>, Mark<'a>)>,
) {
    let mut pos = range.start;
    for &start in sub_starts {
        if start > pos && start < range.end {
            out.push((pos..start, Mark::Skip));
            pos = start;
        }
    }
    out.push((pos..range.end, Mark::Skip));
}

/// Split one hidden range of a table block at the cell edges it crosses,
/// pushing each piece with the verdict its own bytes earn: inside a cell it is
/// inline markup and contributes nothing, outside every cell it is grid chrome
/// and contributes a barrier.
///
/// `cells` is in source order and non-overlapping (the render's own geometry —
/// [`table_cells`] asserts it), and every edge is a character boundary of the
/// source, so each piece is a range [`ProjectionBuilder`] will accept.
fn split_at_cell_edges<'a>(
    range: Range<usize>,
    cells: &[Range<usize>],
    sub_starts: &[usize],
    out: &mut Vec<(Range<usize>, Mark<'a>)>,
) {
    let mut pos = range.start;
    while pos < range.end {
        match cell_at(cells, pos) {
            // Inside a cell — inline markup, up to that cell's end.
            CellAt::Inside(end) => {
                let end = end.min(range.end);
                push_hide(pos..end, sub_starts, out);
                pos = end;
            }
            // Between cells — the grid, up to wherever the next cell begins.
            CellAt::Between(next) => {
                let end = next.unwrap_or(range.end).min(range.end);
                out.push((pos..end, Mark::Substitute(BARRIER)));
                pos = end;
            }
        }
    }
}

/// Where one byte of a table block falls, relative to the grid's cells.
#[derive(Debug, PartialEq, Eq)]
enum CellAt {
    /// Inside a cell, whose content ends here.
    Inside(usize),
    /// Outside every cell — grid chrome — running until the next cell begins,
    /// or to the end of the block where none does.
    Between(Option<usize>),
}

/// Which cell (if any) a byte falls in, by **binary search** rather than a scan.
///
/// `append_block` calls `split_at_cell_edges` once per hidden range, and a table
/// has one hidden chrome range per cell edge — so a linear `find` over the cell
/// list, plus the second linear scan the between-cells arm took for the next
/// cell start, made opening the bar on a large generated table quadratic in its
/// cells and froze the window synchronously. The cells are already in source
/// order and non-overlapping (the render laid the grid), so the position is a
/// question a `partition_point` answers, and one call answers both arms: the
/// cell before the partition is the only one that can contain `pos`, and the
/// cell at it is the next one to begin.
///
/// **A lookup, not a rule.** The verdict this feeds — inside a cell is inline
/// markup, outside is a barrier — is unchanged, which is what keeps the
/// cell-edge and merged-hide semantics exactly as they were.
fn cell_at(cells: &[Range<usize>], pos: usize) -> CellAt {
    let next = cells.partition_point(|c| c.start <= pos);
    if let Some(cell) = next.checked_sub(1).map(|i| &cells[i])
        && pos < cell.end
    {
        return CellAt::Inside(cell.end);
    }
    CellAt::Between(cells.get(next).map(|c| c.start))
}

/// The content ranges of every cell a table block carries, or an empty vec for
/// any other block. The delimiter row has no cells to name — the render hides
/// its whole line, which is then outside every cell and so a barrier like the
/// rest of the grid.
///
/// Rows come out of the geometry in source order and a row's cells in column
/// order, so the result is sorted and non-overlapping — the premise
/// [`cell_at`]'s binary search rests on, asserted here where it is established
/// rather than assumed where it is used.
fn table_cells(block: &gpui_markdown_editor::RenderBlock) -> Vec<Range<usize>> {
    let gpui_markdown_editor::BlockKind::Table { geometry, .. } = &block.kind else {
        return Vec::new();
    };
    let cells: Vec<Range<usize>> = geometry
        .rows
        .iter()
        .filter(|row| row.kind != gpui_markdown_editor::table::RowKind::Delimiter)
        .flat_map(|row| row.cells.iter().cloned())
        .collect();
    debug_assert!(
        cells.windows(2).all(|w| w[0].end <= w[1].start),
        "the render's grid is in source order and non-overlapping: {cells:?}"
    );
    cells
}

/// One match: where it is, and the identity that lets the current-match anchor
/// survive the transcript being replaced under it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Match {
    /// The tree node the match is in — a post's action id, or a draft's
    /// sentinel id. What keys the body editor whose highlight layer paints it.
    pub(crate) node: SharedString,
    /// The post's **item** id, which survives an edit or a regeneration where
    /// the action id does not. `None` for a draft and for an optimistic row.
    pub(crate) item_id: Option<SharedString>,
    /// This match's index within its own node, in projection order — the half
    /// of the anchor that says *which* match, once the item says which post.
    pub(crate) ordinal: usize,
    /// The byte range in the node's markdown source. What
    /// `set_highlights_in` takes, and what `content_y_for_offset` resolves.
    pub(crate) source: Range<usize>,
    /// Where the match sits in the node's source, as a fraction of its length.
    ///
    /// The honest approximation two surfaces need before anything has been
    /// laid out: the minimap tick's position within its cell, and the reveal's
    /// first phase for a post that has not rendered for real. Both correct
    /// themselves once the post is measured — the cell from the height cache,
    /// the reveal from `content_y_for_offset`.
    pub(crate) fraction: f32,
}

/// Every match on the visible branch, grouped by the node it belongs to.
///
/// **The grouping is the point.** Two surfaces ask a per-node question on every
/// frame a session is open — the highlight layers, once per body editor
/// (`sync_references` walks every post that has one), and the minimap, once per
/// cell of the selected column — and each used to answer it by scanning the
/// whole match vector and cloning what matched. That is `O(posts × matches)`
/// per frame, on a list a one-character query in a long conversation makes very
/// long, and it is paid while the reader scrolls and while a reveal animates,
/// with none of the virtualization that bounds post *shaping* helping at all.
///
/// So the group is built where the matches are: `sync_find` walks the scope in
/// document order and pushes one node's matches at a time, which already lays
/// them out as contiguous runs — [`Self::push`] only has to record where each
/// run starts and ends. A per-node read is then a **slice of that node's own
/// run**, which is what makes "an editor never looks at another node's matches"
/// structural rather than careful.
///
/// The two halves cannot drift, because `push` and `clear` are the only ways to
/// change either: the index is a function of the vector, maintained in the same
/// statement that grows it.
#[derive(Default)]
pub(crate) struct MatchSet {
    all: Vec<Match>,
    /// Node id → that node's contiguous run of `all`.
    runs: HashMap<SharedString, Range<usize>>,
}

impl MatchSet {
    /// Append one match. Matches arrive grouped by node (the scope walk), so a
    /// match whose node is the one the last run names extends that run; any
    /// other opens a new one.
    fn push(&mut self, m: Match) {
        let at = self.all.len();
        match self.runs.get_mut(&m.node) {
            Some(run) if run.end == at => run.end = at + 1,
            Some(_) => debug_assert!(
                false,
                "a node's matches arrive in one run: {} came back later",
                m.node
            ),
            None => {
                self.runs.insert(m.node.clone(), at..at + 1);
            }
        }
        self.all.push(m);
    }

    fn clear(&mut self) {
        self.all.clear();
        self.runs.clear();
    }

    /// One node's own matches, in projection order. The read every per-node
    /// surface takes, and the reason neither of them is O(all).
    fn of(&self, node: &SharedString) -> &[Match] {
        match self.runs.get(node) {
            Some(run) => &self.all[run.clone()],
            None => &[],
        }
    }

    /// Which of `node`'s own matches the anchor names, as an offset into
    /// [`Self::of`].
    ///
    /// O(1) and local: a node's ordinals are `0..run.len()` by construction
    /// (`sync_find` enumerates a projection's hits), so the anchor's ordinal
    /// *is* the offset once its key names this run — which is the whole of what
    /// a per-node surface needs to know about "which one is current", without
    /// resolving the anchor against every match in the space.
    fn current_of(&self, node: &SharedString, anchor: &Option<MatchAnchor>) -> Option<usize> {
        let anchor = anchor.as_ref()?;
        let run = self.of(node);
        let m = run.get(anchor.ordinal)?;
        (MatchAnchor::of(m) == *anchor).then_some(anchor.ordinal)
    }
}

impl std::ops::Deref for MatchSet {
    type Target = [Match];

    fn deref(&self) -> &[Match] {
        &self.all
    }
}

impl FromIterator<Match> for MatchSet {
    fn from_iter<T: IntoIterator<Item = Match>>(iter: T) -> Self {
        let mut set = Self::default();
        for m in iter {
            set.push(m);
        }
        set
    }
}

/// The anchor for "the current match": an identity, never an index into the
/// match list.
///
/// A `Change::Space` fires on every post, turn, memory write and background
/// summary pass; each one reloads the transcript and replaces `posts`, and an
/// edit or regeneration mints a **new action id** for the same post. An index
/// would then name a different match, and an action id would name nothing —
/// the defect `retarget_tree_focus` and `rethread_drafts` already forward
/// through item identity to avoid. So the anchor is `(item, ordinal)` where
/// the node has an item, and falls back to the node id for a draft, which has
/// no durable identity but also cannot be superseded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MatchAnchor {
    pub(crate) key: SharedString,
    pub(crate) ordinal: usize,
}

impl MatchAnchor {
    fn of(m: &Match) -> Self {
        Self {
            key: m.item_id.clone().unwrap_or_else(|| m.node.clone()),
            ordinal: m.ordinal,
        }
    }
}

/// A reveal waiting for the post it names to render for real.
///
/// Revealing a match is two-phase because only posts intersecting the viewport
/// render a real `MarkdownEditor`: everything else is a sized placeholder with
/// no shaped lines and therefore no per-offset geometry. Phase 1 glides to an
/// estimate (the match's byte fraction of the post's height); phase 2 corrects
/// it once `content_y_for_offset` answers. Like `PendingSelect` and the tail
/// pin, the reader takes it back the moment they scroll, navigate or type.
#[derive(Clone, Debug)]
pub(crate) struct PendingReveal {
    pub(crate) node: SharedString,
    /// The match's own byte range. The vertical correction needs only its
    /// start; the horizontal one needs the whole span, because what has to end
    /// up inside a block's clip is the *last* matched glyph.
    pub(crate) source: Range<usize>,
}

/// What the whole-space count is an answer **about**.
///
/// A total that is merely old is a settled number that is wrong, so the pass
/// starts over whenever any of this moves rather than letting the readout
/// stand on an answer about a conversation that has changed underneath it.
/// Three inputs, and each is a real one: the committed query; the transcript
/// ([`SpaceView::posts_generation`], bumped by every `rebuild`); and the posts
/// whose answer is being replaced, which is the one exclusion that moves with
/// **no** rebuild behind it — a regeneration begins by pushing a turn, not by
/// writing a row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CountKey {
    query: Option<SharedString>,
    posts: u64,
    revising: Vec<SharedString>,
}

/// The whole-space pass: every post's own match count, how far the walk has
/// got, and whether it has finished.
///
/// **[`Self::is_settled`] is the only thing the readout may believe.** While it
/// is false the total is not knowable and the bar says exactly that — never a
/// partial sum, which on screen is indistinguishable from a right one, and
/// never the previous query's total, which is the same lie with a longer fuse.
/// The sibling counts on the minimap are withheld on the same flag, for the
/// same reason: "3 more in this branch" while the pass is still walking is a
/// number a reader would act on.
///
/// **Two halves make it up, because the two have different invalidations.**
/// The posts walk once per [`CountKey`] and are done; a **retained draft's**
/// text moves with no transcript rebuild behind it, so no key can say it is
/// stale and every frame re-asks. Either half being unfinished means the total
/// is not a number yet — which is the whole of what
/// [`SpaceView::count_retained_drafts`] owes for having become budgeted.
#[derive(Default)]
pub(crate) struct SpaceCount {
    /// What [`Self::counts`] is an answer about, once a pass has begun.
    key: Option<CountKey>,
    /// Own match counts per **post** node, filled in as the walk proceeds.
    counts: HashMap<SharedString, usize>,
    /// The next index into `SpaceView::posts` the walk will take.
    next: usize,
    /// Whether the walk has reached the end of the **posts** for [`Self::key`].
    posts_done: bool,
    /// Whether a **draft** the count reaches is waiting on a later chunk —
    /// which is to say, whether the last pass over them stopped for budget
    /// rather than for having finished.
    drafts_pending: bool,
    /// The settled total over this frame's effective tree, recomputed in
    /// `sync_find` and held as an **integer** — the readout formats the words,
    /// because a localized string in state is a cached render decision that a
    /// locale change would leave standing (the localization doctrine's rule).
    pub(crate) total: Option<usize>,
}

impl SpaceCount {
    /// Start over for `key`: no counts, no progress, no total, and — the
    /// load-bearing half — not settled.
    fn restart(&mut self, key: CountKey) {
        self.key = Some(key);
        self.counts.clear();
        self.next = 0;
        self.posts_done = false;
        self.drafts_pending = false;
        self.total = None;
    }

    /// Whether the count is a number anything may show.
    pub(crate) fn is_settled(&self) -> bool {
        self.posts_done && !self.drafts_pending
    }
}

/// Everything one window holds while its find bar is open.
///
/// Window-local by the same rule the composer draft is: two windows on one
/// space are two cursors, so they are two searches (`STATE.md`'s scoping
/// table). The projection cache lives here rather than beside the body
/// editors, which is what makes "no projection while no session is open"
/// structural.
pub(crate) struct FindSession {
    /// The query field. A `gpui_component::Input`, whose preedit path emits no
    /// `InputEvent::Change` — which is exactly why the search is driven by
    /// that event and never by an observer or a render-time `value()` read
    /// (both of which see uncommitted preedit, so a reader composing in
    /// Chinese or Japanese would search fragments they have not chosen).
    pub(crate) input: gpui::Entity<gpui_component::input::InputState>,
    /// The field's subscription — `Change` re-runs the search, `PressEnter`
    /// steps. Held so it dies with the session.
    pub(crate) _sub: gpui::Subscription,
    /// The last **committed** query text, as the search last ran it. Held so
    /// the search can be re-run against a changed transcript without reading
    /// the field (see above).
    pub(crate) text: String,
    /// The prepared query, or `None` while the field is empty.
    pub(crate) query: Option<Query>,
    /// **Which query the results on screen belong to.** Bumped by
    /// [`SpaceView::set_find_query`] — the one door a query changes through,
    /// and one that early-returns on unchanged text, so this moves exactly when
    /// the search does.
    ///
    /// A rendered `ResultFragment` carries the generation it was cut for,
    /// because gpui draws from the platform's frame callback and really does
    /// handle two events between two paints: a `Change` and a card's click land
    /// in that order, the click closure still holds the *old* fragment, and
    /// opening it selected a branch and installed an ordinal from a search that
    /// no longer exists — `sync_find` then resolved the new query from a branch
    /// the reader never chose. An ordinal is meaningless across queries, which
    /// is why identity rather than repair is the answer.
    pub(crate) query_generation: u64,
    /// Every match on the visible branch, in document order and grouped by
    /// node — see [`MatchSet`] for why the grouping is not an optimization.
    pub(crate) matches: MatchSet,
    /// Which match the readout counts as current.
    pub(crate) anchor: Option<MatchAnchor>,
    /// This frame's **visible-branch** node ids.
    ///
    /// Two answers exist for a node's own match count and they are not
    /// interchangeable: the branch pass reads what is on screen (an inline
    /// edit's unsaved buffer, a draft's live text), the whole-space pass reads
    /// the persisted post. This says which of them is authoritative for a
    /// node, so the total, the sibling counts and the highlight layers can
    /// never describe different text (see [`SpaceView::own_match_count`]).
    pub(crate) branch_nodes: HashSet<SharedString>,
    /// The scope node that is in [`Self::branch_nodes`] **only because it
    /// floats** — the active draft whose own branch is not the selected one.
    ///
    /// It is on screen, so it belongs to the *visible* side of the exactness
    /// sum and never to a sibling's "reachable only through this branch". See
    /// [`FindScope`] for the whole rule.
    pub(crate) floating: Option<SharedString>,
    /// The whole-space count and its honest in-progress state.
    pub(crate) space: SpaceCount,
    /// **The Find-all overlay** — the surface behind the total's disclosure.
    ///
    /// Held here, never beside the session, for the same reason the projection
    /// cache is: closing the bar drops the overlay's scroll position, its
    /// measured heights and every editor state it minted, so "no session, no
    /// work" stays structural. See [`super::find_overlay`].
    pub(crate) overlay: super::find_overlay::FindOverlay,
    /// The chunked pass that fills [`Self::space`].
    ///
    /// A task on the session, per `STATE.md`: replace = cancel, and dropping
    /// the session drops the task — which is what keeps "no session, no work"
    /// structural rather than remembered, exactly as the projection cache's
    /// placement does.
    pub(crate) count_task: Option<gpui::Task<()>>,
    /// Per-node searchable projections, keyed by node id, each remembered with
    /// the content it was built from so a post whose text changed re-projects
    /// and one that did not is free — and with the hits it last answered, so
    /// one that did not is free to *scan* as well as to project
    /// ([`CachedProjection`]).
    pub(crate) projections: HashMap<SharedString, CachedProjection>,
    /// A reveal waiting for its post to render for real.
    pub(crate) pending_reveal: Option<PendingReveal>,
    /// **A reveal a new query is owed, once the search has said what the first
    /// match is.**
    ///
    /// A query clears the anchor — a new search re-anchors from the reader's
    /// own place rather than stepping from a match that belonged to a
    /// different one — and only [`SpaceView::sync_find`] can put a new one
    /// back, because the match list is a function of the frame's selected
    /// path. So the moment the query changes there is nothing to reveal *yet*,
    /// and revealing there could only ever find no current match: the reader
    /// was told "1 of N" while match 1 sat off-screen until they stepped. The
    /// intent is recorded instead, and discharged where the anchor is
    /// established.
    pub(crate) reveal_when_anchored: bool,
    /// The bar's own focus handle — the destination when ⌘F re-focuses an open
    /// bar, and the subtree containment is asked of when the bar closes.
    pub(crate) focus: gpui::FocusHandle,
    /// The placeholder the field was last seeded with.
    ///
    /// A localized string held in state would be a cached render decision —
    /// `i18n::apply` refreshes every window and the field would keep painting
    /// the old language. This is not that: it is the *seed*, compared against
    /// a freshly formatted message each render so the field is re-seeded
    /// exactly when the wording moves (the inspector title's shape).
    pub(crate) placeholder: SharedString,
    /// **The inspector field the keyboard came from**, weakly.
    ///
    /// [`SpaceView::keyboard_home`] answers for the conversation — the
    /// reader's tree level, or the composing session that owns the keyboard —
    /// and those it can *derive* at the moment of the question. An inspector
    /// text field it cannot: which of the panel's fields a reader stood in is
    /// not recoverable once the bar has taken the keyboard, and handing it to
    /// the view root instead is not a dead window (the panel's own predicate
    /// is focus-derived, so it stops yielding) but a **wrong** one: the next
    /// character is then type-to-compose, and a reader mid-way through a
    /// system prompt gets a draft.
    ///
    /// So the *lender* is remembered, and it is remembered as the **entity**
    /// rather than its focus handle. That is the whole difference: a
    /// `FocusHandle` recorded here would be the dead slot this window's focus
    /// doctrine is built around — tracked on no element, still reporting
    /// itself focused — where an `InputState` dies with the form that owns it,
    /// so a participant editor closed while the bar stood open simply fails to
    /// upgrade and the handback falls through to `keyboard_home`. Derived at
    /// the moment of use, exactly like the composing arm; only the *identity*
    /// is carried, never the answer.
    pub(crate) returned_input: Option<gpui::WeakEntity<gpui_component::input::InputState>>,
}

/// What one node holds for the current query — [`FindSession::node_result`]'s
/// answer, borrowed from the projection cache rather than copied out of it.
pub(crate) struct NodeResult<'a> {
    /// The markdown the projection was built from: the live buffer for a node
    /// the reader is editing, the persisted row for everything else.
    pub(crate) content: &'a SharedString,
    /// Each rendered block's source range, in document order.
    pub(crate) blocks: &'a [Range<usize>],
    /// The query's hits, in projection order — index *is* the match's ordinal
    /// within the node, by the same construction `MatchSet` relies on.
    pub(crate) hits: &'a [Range<usize>],
    /// Whether the projection was built **cursor-aware** — that is, whether
    /// this node is a live editor (a draft, or the post being edited in place).
    ///
    /// It is the one honest answer to "can this node's text move while its id
    /// stands still": every other node mints a new action id when its text
    /// changes, so its node id is already the discriminator. The overlay reads
    /// it to stamp a fragment's measurement key — see [`ResultFragment::id`].
    pub(crate) live_editor: bool,
}

impl FindSession {
    /// Whether this session owns the keyboard: the query field's own handle,
    /// or anything inside the bar's subtree.
    ///
    /// **One predicate, because its two callers ask on different frames.**
    /// `SpaceView::find_holds_focus` gates the Escape rung and the
    /// transient-overlay predicate; `close_find` asks the same question to
    /// decide whether it owes a handback. Containment alone answers from the
    /// dispatch tree the *last* frame built, so between ⌘F focusing the field
    /// and the bar's first paint it says no — and an Escape arriving in that
    /// window is admitted by the rung (which asks the field) and then dropped
    /// by the close (which did not), taking the session away without putting
    /// the keyboard back where it came from. Written once, the two cannot
    /// disagree about a frame.
    /// **The overlay is the third half**, because it is a surface of the
    /// session painted outside the bar's own subtree: a reader standing in the
    /// results list is inside the find surface, and an Escape there belongs to
    /// find rather than to the conversation behind it.
    fn holds_focus(&self, window: &Window, cx: &gpui::App) -> bool {
        gpui::Focusable::focus_handle(self.input.read(cx), cx).is_focused(window)
            || self.focus.contains_focused(window, cx)
            || self.overlay.holds_focus(window, cx)
    }

    /// One node's text, the blocks its render laid it out in, and the hits the
    /// current query found there — **read from the one cache both passes
    /// fill**, and only when the memo is an answer about the query the bar is
    /// showing.
    ///
    /// This is what keeps the Find-all overlay from being a second source of
    /// truth: `sync_find` fills a visible-branch node's entry from what is on
    /// screen (an inline edit's unsaved buffer, a draft's live text) and the
    /// whole-space count fills every other post's from the persisted row, both
    /// through `hits_of` — so a fragment can only ever show what the readout
    /// counted, and the ordinals a result hands the anchor are the ordinals the
    /// branch's own match list uses.
    ///
    /// `None` for a node the pass has not reached, and for a composing draft
    /// deliberately left unprojected. Neither is a group; the overlay says it
    /// is still counting rather than presenting a partial set as a whole one.
    pub(crate) fn node_result(&self, node: &SharedString) -> Option<NodeResult<'_>> {
        let query = self.query.as_ref()?;
        let entry = self.projections.get(node)?;
        let (answered, hits) = entry.hits.as_ref()?;
        (answered == query).then_some(NodeResult {
            content: &entry.seed.content,
            blocks: &entry.projection.blocks,
            hits,
            live_editor: entry.seed.render_cursor.is_some(),
        })
    }

    /// The current match, if the anchor still names one.
    pub(crate) fn current(&self) -> Option<&Match> {
        current_match(&self.matches, &self.anchor)
    }

    /// The current match's position in the readout, 1-based.
    pub(crate) fn current_index(&self) -> Option<usize> {
        current_position(&self.matches, &self.anchor).map(|i| i + 1)
    }

    /// Re-anchor after the match list has been rebuilt.
    pub(crate) fn reanchor(&mut self, previous: Option<(MatchAnchor, usize)>) {
        reanchor(&self.matches, &mut self.anchor, previous);
    }

    /// Step the anchor by one match, wrapping at both ends.
    pub(crate) fn step(&mut self, forward: bool) -> Option<Match> {
        step_anchor(&self.matches, &mut self.anchor, forward)
    }

    /// **A pending reveal is a promise about the current match**, so it moves
    /// wherever re-anchoring moved that match.
    ///
    /// [`reanchor`] forwards through *item* identity, which is the whole point
    /// of it: an edit or a regeneration mints a new action id for the same
    /// post, and the reader's place survives. But a reveal already in flight
    /// records the **node** — an action id — and `sync_bodies` prunes the
    /// editor of the id that just went away, so the correction could never
    /// obtain geometry for it. The estimate then stood as the final answer and
    /// nothing ever landed on the match. Called right after every re-anchor,
    /// so the two cannot disagree for a frame.
    pub(crate) fn refollow_pending_reveal(&mut self) {
        if self.pending_reveal.is_none() {
            return;
        }
        let Some(m) = current_match(&self.matches, &self.anchor) else {
            // Nothing is current any more; there is nothing left to reveal.
            self.pending_reveal = None;
            return;
        };
        self.pending_reveal = Some(PendingReveal {
            node: m.node.clone(),
            source: m.source.clone(),
        });
    }
}

fn current_position(matches: &[Match], anchor: &Option<MatchAnchor>) -> Option<usize> {
    let anchor = anchor.as_ref()?;
    matches.iter().position(|m| &MatchAnchor::of(m) == anchor)
}

fn current_match<'a>(matches: &'a [Match], anchor: &Option<MatchAnchor>) -> Option<&'a Match> {
    current_position(matches, anchor).map(|i| &matches[i])
}

/// Re-anchor after the match list has been rebuilt: keep the same match if it
/// is still there, else the same **item** clamped to its new count, else the
/// nearest match at or after where the old one stood in document order.
///
/// This is `retarget_tree_focus`'s rule, applied to a different window-local
/// reference to a post — and for the same reason: an edit or a regeneration
/// replaces a post's action id while the post itself stays, so an anchor that
/// could only be matched exactly would jump the reader back to the first match
/// every time a background write landed. `Change::Space` fires on every post,
/// turn, memory write and background summary pass, so that is often.
fn reanchor(
    matches: &[Match],
    anchor: &mut Option<MatchAnchor>,
    previous: Option<(MatchAnchor, usize)>,
) {
    if matches.is_empty() {
        *anchor = None;
        return;
    }
    let Some((was, position)) = previous else {
        *anchor = Some(MatchAnchor::of(&matches[0]));
        return;
    };
    let found = matches
        .iter()
        .find(|m| MatchAnchor::of(m) == was)
        // The item survives; its match count may not. Clamp to the last match
        // the item still has.
        .or_else(|| matches.iter().rfind(|m| MatchAnchor::of(m).key == was.key));
    *anchor = Some(MatchAnchor::of(match found {
        Some(m) => m,
        // The item is gone: the nearest match at or after where it stood.
        None => &matches[position.min(matches.len() - 1)],
    }));
}

/// Drop everything the *previous* query established, so only [`SpaceView::sync_find`]
/// can establish an anchor for the new one.
///
/// The anchor goes because a new query re-anchors from the reader's own place
/// rather than stepping from a match that belonged to a different search.
/// **The match set goes with it, and that is the part that is easy to miss**:
/// the set is rebuilt only by the next render's `sync_find`, so a Return,
/// Shift-Return or arrow handled between `InputEvent::Change` and that render
/// stepped through the *previous* query's matches and left an anchor naming
/// one of them. `sync_find` then read that anchor as where the reader stood
/// and forwarded it into the new results by identity, so a fast
/// query-and-step selected and revealed a match the new search never chose.
/// Emptied here, [`step_anchor`] has nothing to walk and leaves the anchor
/// `None` — which is exactly the state [`reanchor`] turns into "the new
/// query's first match".
///
/// A free function over the three fields, like [`step_anchor`] and
/// [`reanchor`] beside it, because the sequence it guards cannot be driven
/// through a window: gpui's test harness draws every dirty window inside each
/// effect flush (`App::flush_effects`, under `cfg(test)`), which fuses the
/// notified render onto the `Change` that triggered it. Production draws from
/// the platform's frame callback instead, so two events really can be handled
/// between two draws. The rule is asserted here, where that fusion cannot
/// reach it.
fn invalidate_for_new_query(
    matches: &mut MatchSet,
    anchor: &mut Option<MatchAnchor>,
    pending_reveal: &mut Option<PendingReveal>,
) {
    matches.clear();
    *anchor = None;
    *pending_reveal = None;
}

/// Step the anchor by one match, wrapping at both ends — what makes the
/// readout an index rather than a running total.
fn step_anchor(
    matches: &[Match],
    anchor: &mut Option<MatchAnchor>,
    forward: bool,
) -> Option<Match> {
    if matches.is_empty() {
        *anchor = None;
        return None;
    }
    let at = current_position(matches, anchor);
    let next = match (at, forward) {
        (None, true) => 0,
        (None, false) => matches.len() - 1,
        (Some(i), true) => (i + 1) % matches.len(),
        (Some(i), false) => (i + matches.len() - 1) % matches.len(),
    };
    *anchor = Some(MatchAnchor::of(&matches[next]));
    Some(matches[next].clone())
}

// ---------------------------------------------------------------------------
// The view's half: the session's lifecycle, the bar, and the reveal.
// ---------------------------------------------------------------------------

/// The find bar's control row — the height it adds to `doc_reserve` while a
/// session is open, *below* the window's drag band. The bar's surface spans
/// from the window top so it reads as one panel behind the traffic lights; its
/// controls sit under the band so the drag gesture keeps its strip.
pub(crate) const FIND_BAR_H: f32 = 44.0;

/// **What a cached projection was built from** — every input
/// [`searchable_projection`] takes, so "is the cache still good" is one
/// comparison against the scope entry rather than a list of fields a later
/// change can forget to extend.
///
/// The embed map is in here because it really is an input: a marker is hidden
/// wholesale only when its ordinal is *mapped*, and a stored quote's range can
/// stop resolving (its source edited) while the quoting post's own content
/// never moves — flipping the marker between hidden and literal text with
/// nothing about `content` to show for it. `sync_references` re-seeds the
/// editor's map on that frame, so a projection keyed on content alone is a
/// count that disagrees with the post the reader is looking at.
/// The render cursor is an input for the same reason: an *enabled* editor
/// renders cursor-aware, so moving the caret into a construct reveals its
/// delimiters with the content and the embed map both unmoved. Keyed on
/// content alone, the cache would hand back a projection of the published
/// render while the reader looks at the raw markdown. `None` is the published
/// render — every node that is not being edited.
#[derive(PartialEq)]
pub(crate) struct ProjectionSeed {
    content: SharedString,
    embeds: EmbedMap,
    render_cursor: Option<Selection>,
}

/// This frame's search scope, and which of its nodes is in it **only because
/// it floats**.
///
/// The scope admits the active draft whatever branch it belongs to, because an
/// active composer paints over whatever is showing — that is what
/// [`MatchReveal::Composer`] exists for. The tree, meanwhile, still attaches
/// that draft beneath its own branch, so without naming it the sibling
/// aggregate counted it as well: a query matching only the floating draft read
/// "1 of 1" and "1 total" beside a map cell offering "1 more in this branch",
/// which is a match that is not *more* — it is the one on screen.
///
/// So the exactness invariant is stated with the visible side named: **"only
/// through this branch" means not currently visible**, the floating composer
/// belongs to the visible side of the sum, and it is excluded from the sibling
/// subtree exactly when (and only when) it is the active floated one. Reported
/// from the one place that decides it rather than re-derived — see
/// [`SpaceView::find_scope`].
struct FindScope {
    nodes: Vec<ScopeNode>,
    floating: Option<SharedString>,
}

/// One node the search covers, in document order.
struct ScopeNode {
    node: SharedString,
    item_id: Option<SharedString>,
    content: SharedString,
    embeds: EmbedMap,
    /// A draft whose editor is mid-IME-composition. Its buffer holds preedit
    /// the reader has not chosen, so a cached projection is reused and a
    /// missing one is simply not built — the count never flickers against
    /// fragments (`MarkdownEditorState::is_composing`).
    frozen: bool,
    /// The cursor the node is about to be *rendered* with: `Some` for a node
    /// whose editor this frame paints enabled — an inline edit, any draft —
    /// and `None` for a published one, which is the read-only render.
    ///
    /// Derived from the view's own edit state, never read back off the
    /// editor: `disabled` is the element prop echoed during the child's
    /// render, one pass later than the parent's, so the child cannot answer
    /// for a frame that has not painted yet.
    render_cursor: Option<Selection>,
}

impl ScopeNode {
    fn seed(&self) -> ProjectionSeed {
        ProjectionSeed {
            content: self.content.clone(),
            embeds: self.embeds.clone(),
            render_cursor: self.render_cursor,
        }
    }
}

/// One node's projection, the seed it was built from, and **the hits it last
/// answered with**.
///
/// The seed keeps the projection honest about the *text*; `hits` keeps the
/// **scan** from happening again for text and a query that have both stood
/// still. That second half is not an optimization detail: `Query::find_in`
/// folds case by building a whole new [`Projection`] of the haystack, so an
/// un-memoized `projection.find(&query)` is O(bytes) *with an allocation*, and
/// three callers ran one per node per frame — the visible branch's match pass,
/// the whole-space post pass, and the retained-draft pass — with nothing about
/// the conversation or the query having moved. The memo makes an unchanged node
/// cost a comparison and a slice read, which is what leaves the per-frame work
/// proportional to the matches actually shown rather than to the text.
pub(crate) struct CachedProjection {
    /// What [`Self::projection`] is a projection *of*.
    seed: ProjectionSeed,
    projection: NodeProjection,
    /// The last query this node was scanned for, and what it found. Held with
    /// the query rather than cleared on a query change, so the memo cannot go
    /// stale by someone forgetting to invalidate it.
    hits: Option<(Query, Vec<Range<usize>>)>,
}

/// The source ranges `query` matches in `node`'s projection, scanning only when
/// the memo cannot answer.
///
/// A free function over the map rather than a method on [`FindSession`],
/// because its caller in `sync_find` pushes into `session.matches` while
/// holding the returned slice — two disjoint fields, which the borrow checker
/// allows only when the borrow names the field.
fn hits_of<'a>(
    projections: &'a mut HashMap<SharedString, CachedProjection>,
    node: &SharedString,
    query: &Query,
    scans: &std::cell::Cell<usize>,
) -> Option<&'a [Range<usize>]> {
    let entry = projections.get_mut(node)?;
    if !entry.hits.as_ref().is_some_and(|(q, _)| q == query) {
        scans.set(scans.get() + 1);
        let hits = entry.projection.find(query);
        entry.hits = Some((query.clone(), hits));
    }
    Some(&entry.hits.as_ref().expect("just filled").1)
}

/// What one retained draft needs from this chunk.
///
/// **A plan says what work is owed; it never carries the work itself.** A
/// `ProjectionSeed` owns a copy of the draft's whole body, so building one is
/// O(bytes) — and the planning pass visits *every* off-branch draft before the
/// admission loop's first budget check, so seeds built here meant one frame
/// copying the entire stale draft corpus and then deferring all but the first
/// of them. The budget bounded the scans and not the copying that preceded
/// them. So a stale draft is planned as the *fact* that it is stale, and the
/// seed is constructed at admission, where the bytes have already been charged
/// — the same rule the posts half gets for free by checking its budget at the
/// top of each iteration rather than planning ahead.
enum DraftWork {
    /// No text: zero, with nothing to project and nothing to scan.
    Empty,
    /// Its projection is the one its text and cursor call for, and its hits are
    /// already memoized for this query — free.
    Memoized,
    /// A scan, and a projection first when `stale`. Spends budget.
    Scan { stale: bool, bytes: usize },
}

/// One chunk's allowance, shared by both halves of the whole-space pass.
///
/// Two numbers because the two costs differ by an order of magnitude:
/// **projecting** is a parse plus a render pass ([`PROJECTION_CHUNK`]), and
/// **scanning** an already-projected node is a substring walk over its bytes
/// ([`SCAN_CHUNK_BYTES`]). One budget across drafts and posts rather than one
/// each, so a chunk's ceiling is a single stated number however the work is
/// divided between them.
#[derive(Default)]
struct ChunkBudget {
    built: usize,
    scanned: usize,
}

impl ChunkBudget {
    /// Whether this chunk has spent its allowance and owes the rest to the
    /// next one.
    fn spent(&self) -> bool {
        self.built >= PROJECTION_CHUNK || self.scanned >= SCAN_CHUNK_BYTES
    }
}

/// The node ids this frame's scope covers — what a cached projection has to be
/// in to survive the prune.
///
/// **Pruning is a membership question, so it is asked of a set.** The cache
/// holds one projection per node the scope carried, so scanning the scope for
/// each surviving entry is `O(posts²)` — and `sync_find` runs *every frame a
/// session is open*, before the query is even looked at, so a long visible
/// branch paid it while the reader scrolled and while a reveal animated, with
/// nothing about the conversation having changed. Building the set once is one
/// pass over the scope and one hash lookup per entry.
///
/// **Only the lookup moves**: the surviving set is exactly what the scan
/// selected — a projection lives while its node is still in scope, and the
/// cache is still bounded by the scope, which is what keeps closing the bar the
/// thing that drops every projection.
fn live_scope_nodes(scope: &[ScopeNode]) -> HashSet<SharedString> {
    scope.iter().map(|entry| entry.node.clone()).collect()
}

/// A post's embed map: the ordinals whose quoted passage still resolves.
///
/// An input to the projection rather than a detail of it — a marker is hidden
/// wholesale only when its ordinal is *mapped*, and ordinary literal text when
/// it is not — so both passes that project a post build it the same way.
fn post_embed_map(post: &super::model::PostData) -> EmbedMap {
    EmbedMap::new(post.references.iter().filter_map(|r| {
        Some((
            u64::try_from(r.ordinal).ok().filter(|o| *o > 0)?,
            r.snippet.clone()?,
        ))
    }))
}

impl SpaceView {
    /// What the open find bar adds to the document's top reserve.
    pub(crate) fn find_bar_h(&self) -> f32 {
        if self.find.is_some() { FIND_BAR_H } else { 0.0 }
    }

    /// ⌘F — open the bar, or re-focus the field of one already open (the macOS
    /// convention). Opening compensates the page scroll by the reserve it
    /// adds, so the reader's content does not jump out from under them.
    pub(crate) fn open_find(
        &mut self,
        _: &crate::actions::FindInSpace,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // The same predicate the rung and the close ask — a second ⌘F can land
        // on the frame the bar mounts in, where containment alone still
        // describes the tree from before it existed, and answering "no" there
        // re-borrows: the focus query finds no inspector field (the keyboard is
        // in the query field) and clobbers a good lender with `None`.
        let open = self
            .find
            .as_ref()
            .map(|session| (session.input.clone(), session.holds_focus(window, cx)));
        if let Some((input, holds)) = open {
            // **A re-borrow is still a borrow, so the lender is refreshed to
            // match.** The bar is opened once and re-focused many times, and
            // the keyboard it takes on the second ⌘F comes from wherever the
            // reader actually is — an inspector field they stepped into after
            // opening it. Recording the lender only on the creation path left
            // the session pointing at whatever held focus that first time (for
            // a bar opened from the conversation, nothing), so closing handed
            // the keyboard to `keyboard_home` and the reader's next character
            // became a draft instead of returning to the field they were in.
            //
            // **Unless the bar already holds the keyboard**, which is the ⌘F
            // pressed inside the find field itself: nothing new is borrowed
            // there, and the focus query would answer `None` and clobber a
            // good lender with it. Every other case *replaces* the lender —
            // `None` included, since a reader who moved back into the
            // conversation is owed `keyboard_home`, not the field they left.
            if !holds {
                let lender = self.inspector_focused_input(window, cx);
                if let Some(session) = self.find.as_mut() {
                    session.returned_input = lender;
                }
            }
            input.update(cx, |s, cx| s.focus(window, cx));
            cx.notify();
            return;
        }
        let placeholder = crate::i18n::msg::find_placeholder(cx);
        let input = cx.new(|cx| {
            gpui_component::input::InputState::new(window, cx).placeholder(placeholder.clone())
        });
        let sub = cx.subscribe_in(
            &input,
            window,
            |this, state, ev: &gpui_component::input::InputEvent, window, cx| match ev {
                // **The one rule the search is driven by.** `InputState`'s
                // preedit path notifies without emitting `Change`, so reading
                // the value *here* reads committed text — where an observer or
                // a render-time `value()` read would see the spliced preedit.
                gpui_component::input::InputEvent::Change => {
                    let text = state.read(cx).value().to_string();
                    this.set_find_query(text, window, cx);
                }
                // **⌘↩ opens the Find-all overlay**, the disclosure's other
                // door and the one a reader already in the field can reach
                // without leaving it. `InputState` reports the modifier on the
                // event itself, so this needs no binding of its own — and a
                // binding would have had to outrank the composer's own submit
                // chord to be reachable here at all.
                gpui_component::input::InputEvent::PressEnter {
                    secondary: true, ..
                } => {
                    this.toggle_find_overlay(window, cx);
                }
                gpui_component::input::InputEvent::PressEnter { shift, .. } => {
                    this.find_step(!shift, window, cx);
                }
                _ => {}
            },
        );
        self.find = Some(FindSession {
            input: input.clone(),
            _sub: sub,
            text: String::new(),
            query: None,
            query_generation: 0,
            matches: MatchSet::default(),
            anchor: None,
            branch_nodes: HashSet::new(),
            space: SpaceCount::default(),
            overlay: super::find_overlay::FindOverlay::new(cx),
            count_task: None,
            floating: None,
            projections: HashMap::new(),
            pending_reveal: None,
            reveal_when_anchored: false,
            focus: cx.focus_handle(),
            placeholder,
            returned_input: self.inspector_focused_input(window, cx),
        });
        // The document grew a reserve at its top; move the page by the same
        // amount so the words under the reader's eye stay where they were.
        self.set_page_scroll_y(self.page_scroll.offset().y.as_f32() - FIND_BAR_H);
        input.update(cx, |s, cx| s.focus(window, cx));
        cx.notify();
    }

    /// Close the session: drop the projections, clear the match layers, and
    /// hand the keyboard back. Returns whether there was one to close (the
    /// Escape rung's answer).
    #[doc(hidden)]
    pub fn close_find(&mut self, window: &mut Window, cx: &mut Context<Self>) -> bool {
        let Some(session) = self.find.take() else {
            return false;
        };
        // Only from a bar that is actually holding the keyboard — a reader
        // composing beside it never lent it (the handback rule every form in
        // this window owes). The same predicate the Escape rung was admitted
        // by, so the two cannot disagree about the frame the bar mounts in.
        let held = session.holds_focus(window, cx);
        let lender = session.returned_input.clone();
        drop(session);
        self.set_page_scroll_y(self.page_scroll.offset().y.as_f32() + FIND_BAR_H);
        if held {
            // The inspector field the bar borrowed from, **if it is still
            // there** — the weak reference is what answers that (see
            // `FindSession::returned_input`). Everything else is derived now
            // rather than recorded then: `keyboard_home` names whatever is
            // live at the moment of the question, including the composing
            // session, which is the one destination that leaves a reader able
            // to go on writing.
            //
            // **And "still there" is two questions, because one of the
            // panel's fields outlives its panel.** The weak reference answers
            // for a *form* — a participant editor retired while the bar stood
            // open takes its `InputState` with it — but `set_inspector_open`
            // deliberately keeps the title field and the open editor across a
            // close, so a hidden panel's field upgrades perfectly well while
            // its element is unmounted. Focusing it is the dead slot this
            // window's whole focus doctrine is about: the handle still reports
            // itself focused, so `inspector_field_focused` goes on yielding
            // every printable to a field nobody is painting, and the window is
            // silent until the reader clicks somewhere. So the mounting is
            // asked of the panel, with the same predicate the render reads
            // (`inspector_open`, which is exactly what decides whether a panel
            // is painted at all) — derived at the moment of the question, like
            // every other answer here, rather than cleared at the close.
            let back = lender
                .filter(|_| self.inspector_open)
                .and_then(|input| input.upgrade())
                .map(|input| gpui::Focusable::focus_handle(input.read(cx), cx))
                .unwrap_or_else(|| self.keyboard_home(cx));
            window.focus(&back, cx);
        }
        // The match layers are cleared on the next `sync_references`, which now
        // sees no session and writes empty sets.
        cx.notify();
        true
    }

    /// Whether this node's editor geometry describes the layout on screen now.
    ///
    /// **A retained editor keeps the pixels of a layout that is gone.** A body
    /// editor is kept for every live post but *rendered* only near the
    /// viewport, and the editor clears its paint-time geometry only in its own
    /// render — so a post that last painted at another reading-column width,
    /// type scale or gutter scheme goes on answering `content_y_for_offset`
    /// with those pixels, and answers `Some`, which is indistinguishable from
    /// fresh. The reveal's correction then took a stale answer as exact,
    /// dropped the pending reveal, and the rewrapped match landed off-screen
    /// with nothing left to correct it — and the two halves of the sum
    /// disagreed on top of that, since the *slot* term comes from the height
    /// cache, which a width change does invalidate.
    ///
    /// That cache is the answer rather than a new stamp: [`Layout`] is keyed on
    /// exactly the editor's own geometry inputs (reading-column width, type
    /// scale, gutter scheme) and is wiped whenever any of them moves, so an
    /// entry exists only for a node that has painted **since** — which is the
    /// question, asked of state that already exists and cannot drift from the
    /// rendering because the measuring canvas writes it.
    ///
    /// The active draft carries no entry and needs none: it is the one node
    /// that renders every frame, so its geometry is fresh by construction.
    fn node_geometry_is_current(&self, node: &SharedString) -> bool {
        self.layout.measured(node).is_some() || self.active_draft.as_ref() == Some(node)
    }

    /// Open a session on `query` — the scene seam, for a surface that has to
    /// exist before any frame has run.
    ///
    /// It goes through `set_find_query`, the same door the field's own `Change`
    /// event takes, so a scene never stands on a state the production path
    /// cannot produce. Calling it again replaces the query on the open session,
    /// which is what the reader typing over their search does.
    #[doc(hidden)]
    pub fn seed_find_for_test(&mut self, query: &str, window: &mut Window, cx: &mut Context<Self>) {
        if self.find.is_none() {
            self.open_find(&crate::actions::FindInSpace, window, cx);
        }
        if let Some(session) = self.find.as_ref() {
            let input = session.input.clone();
            input.update(cx, |s, cx| s.set_value(query, window, cx));
        }
        self.set_find_query(query.to_string(), window, cx);
    }

    /// Whether the find surface currently owns the keyboard — what gates the
    /// Escape rung, so an Escape in the composer still deactivates the draft,
    /// and what puts the bar in `transient_overlay_open`.
    ///
    /// **Two questions, because containment is a paint-time answer and the
    /// field's own handle is not.** `contains_focused` reads the dispatch tree
    /// the *last* frame built, so on the frame ⌘F mounts the bar it cannot yet
    /// see the input `open_find` has already focused — the bar's container has
    /// not painted. `sync_tree_focus` runs at the head of that same render, so
    /// it read "no overlay", found the reader's post no longer focused, and
    /// cleared their place in the tree; closing the bar then landed on the view
    /// root instead of the post they opened it from. Asking the field first
    /// answers from a fact that is true the instant focus moves, with no frame
    /// in between.
    ///
    /// Containment stays as the second half rather than being replaced: the
    /// bar's close and step verbs are ordinary tab stops, so a reader who Tabs
    /// onto one is still inside the bar with the *field* unfocused, and only
    /// the subtree can say so.
    ///
    /// The pair lives on [`FindSession::holds_focus`] so the two callers who
    /// ask it cannot answer differently — see that method.
    pub(crate) fn find_holds_focus(&self, window: &Window, cx: &gpui::App) -> bool {
        self.find
            .as_ref()
            .is_some_and(|s| s.holds_focus(window, cx))
    }

    /// Apply a committed query. Never called from an observer or a render.
    fn set_find_query(&mut self, text: String, window: &mut Window, cx: &mut Context<Self>) {
        let Some(session) = self.find.as_mut() else {
            return;
        };
        if session.text == text {
            return;
        }
        session.query = Query::new(&text);
        session.text = text;
        // Everything cut for the search that just ended stops being this
        // session's — see [`FindSession::query_generation`].
        session.query_generation = session.query_generation.wrapping_add(1);
        // **The overlay's retained position belongs to the query it was taken
        // in.** Position retention is "while the query is unchanged" — the
        // task's own rule — so a new search starts at the top of a list that is
        // about something else, with no cursor pointing into the old one and no
        // measured height or editor state left over from a fragment that is
        // gone.
        session.overlay.forget_results();
        invalidate_for_new_query(
            &mut session.matches,
            &mut session.anchor,
            &mut session.pending_reveal,
        );
        // The first match of the new search is owed a reveal, but nothing here
        // knows which match that is — `sync_find` rebuilds the list against the
        // frame's selected path and re-anchors. Record the debt; it is
        // discharged there (see `reveal_when_anchored`).
        session.reveal_when_anchored = true;
        // **And the motion the old query started ends with it.** A reveal is a
        // multi-frame `PageGlide`, and clearing the anchor does not stop one:
        // a query narrowed to nothing left the page still travelling towards a
        // match of the search before it while the bar read "No results". A new
        // query that *does* match glides again from wherever this stopped, so
        // cancelling is right either way.
        self.cancel_page_glide();
        // **And a query cleared under the overlay takes the overlay with it**,
        // the other half of the same rule. The disclosure that collapses this
        // surface disappears with the readout it lives beside, so an overlay
        // left standing over an emptied query has lost its own pointer way out
        // while claiming "Nothing matches anywhere" about a search that is no
        // longer being made. Through `close_find_overlay`, so the keyboard goes
        // back to the field the reader is typing in rather than being left on a
        // results list that has stopped existing.
        if self
            .find
            .as_ref()
            .is_some_and(|s| s.query.is_none() && s.overlay.open)
        {
            self.close_find_overlay(window, cx);
        }
        cx.notify();
    }

    /// Return / Shift-Return and the prev/next arrows: step the anchor and
    /// reveal where it landed. Both wrap.
    #[doc(hidden)]
    pub fn find_step(&mut self, forward: bool, window: &mut Window, cx: &mut Context<Self>) {
        let Some(session) = self.find.as_mut() else {
            return;
        };
        session.step(forward);
        self.reveal_find_anchor(window, cx);
        cx.notify();
    }

    /// Recompute the match set against this frame's visible branch.
    ///
    /// Run every frame a session is open rather than invalidated by hand: the
    /// scope is a function of the *selected path*, which the branch scrollers
    /// decide at render time, so there is no event a "recompute now" could
    /// hang off that a wheel gesture would not miss.
    ///
    /// What that costs is bounded by the memo rather than by the text: the
    /// projections are cached per node against the content they were built
    /// from, and the hits against the query they answered, so an unchanged
    /// post on an unchanged query is a comparison and a slice read. A node
    /// whose text moved really is re-projected and re-scanned here, on the
    /// frame and outside any budget — deliberately, because the highlights and
    /// the index describe the branch the reader is looking at *now*, and the
    /// visible branch is the one scope that cannot be deferred.
    pub(crate) fn sync_find(
        &mut self,
        tree: &[TreeNode],
        page_width: Pixels,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.find.is_none() {
            return;
        }
        let FindScope {
            nodes: scope,
            floating,
        } = self.find_scope(tree, page_width, cx);
        let mut session = self.find.take().expect("checked above");
        let previous = session.anchor.clone().map(|a| {
            (
                a,
                current_position(&session.matches, &session.anchor).unwrap_or(0),
            )
        });
        session.matches.clear();
        session.branch_nodes = live_scope_nodes(&scope);
        session.floating = floating;
        let live = self.live_projection_nodes(&session.branch_nodes);
        session.projections.retain(|node, _| live.contains(node));

        if let Some(query) = session.query.clone() {
            for entry in &scope {
                // **A composing draft is never re-projected.** Its buffer holds
                // preedit the reader has not chosen, so the projection it
                // already has — of the text they *have* — is what the count
                // keeps reporting until the composition commits. With nothing
                // cached the draft is simply not searched yet; either way the
                // count never moves against a fragment.
                let seed = entry.seed();
                let cached = session
                    .projections
                    .get(&entry.node)
                    .is_some_and(|c| entry.frozen || c.seed == seed);
                if !cached {
                    if entry.frozen {
                        continue;
                    }
                    let projection =
                        searchable_projection(&entry.content, &entry.embeds, entry.render_cursor);
                    self.projections_built.set(self.projections_built.get() + 1);
                    session.projections.insert(
                        entry.node.clone(),
                        CachedProjection {
                            seed,
                            projection,
                            hits: None,
                        },
                    );
                }
                // The scan happens here only when the memo cannot answer — a
                // node whose text and query have both stood still costs a
                // comparison, where scanning it again is O(bytes) plus the
                // fold's own allocation, every frame the bar is open.
                let Some(hits) = hits_of(
                    &mut session.projections,
                    &entry.node,
                    &query,
                    &self.scans_run,
                ) else {
                    continue;
                };
                let len = entry.content.len().max(1) as f32;
                for (ordinal, source) in hits.iter().cloned().enumerate() {
                    session.matches.push(Match {
                        node: entry.node.clone(),
                        item_id: entry.item_id.clone(),
                        ordinal,
                        fraction: (source.start as f32 / len).clamp(0.0, 1.0),
                        source,
                    });
                }
            }
        }
        session.reanchor(previous);
        session.refollow_pending_reveal();
        // **The new query's own reveal, discharged where the anchor exists.**
        // The match list is final for this frame by now, so an armed debt is
        // either paid or has nothing to pay: a query matching nothing leaves no
        // anchor, and the readout says so instead.
        let owed = std::mem::take(&mut session.reveal_when_anchored) && session.current().is_some();
        self.find = Some(session);
        // The visible branch is settled for this frame; the rest of the space
        // is the count's business, and it may take several.
        self.sync_space_count(cx);
        // One walk of the effective tree per frame, held as an integer for the
        // readout — and `None` for as long as the pass has not finished.
        let total = self.find_space_total(tree);
        if let Some(session) = self.find.as_mut() {
            session.space.total = total;
        }
        if owed {
            self.reveal_current_match(tree, page_width, window, cx);
        }
        self.correct_find_reveal(tree, page_width, window, cx);
    }

    /// Count the drafts the search scope does not cover.
    ///
    /// **A draft is minted for every leaf of the whole tree, and a non-empty
    /// one is never pruned** (`sync_tail_drafts` over `tail_parents`), so a
    /// reader who wrote on one branch and navigated to another leaves prose
    /// behind. It is in the effective tree — every draft is attached — but it
    /// is not in `self.posts` and not in the scope, so without this its matches
    /// were in neither the sibling number nor the total, and appeared the
    /// moment the reader visited that branch with nothing about the
    /// conversation having changed.
    ///
    /// **Asked every frame, but budgeted like everything else.** A draft's text
    /// moves with no transcript rebuild behind it, so it has no honest place in
    /// [`CountKey`] and no key can say it is stale — every frame therefore
    /// re-asks. What that costs is a comparison per draft: an empty one, and
    /// one whose content, cursor and embed map are the ones its projection was
    /// built from *and* whose hits are already memoized for this query, are
    /// both answered without touching a byte of text.
    ///
    /// A draft that is *not* one of those is a scan, and a scan is O(bytes)
    /// with the case fold's own allocation on top — so the **first** look at a
    /// large pasted draft is real work, and it spends the chunk's budget like
    /// any post. Past the budget the rest are deferred to a later chunk, which
    /// is what [`SpaceCount::drafts_pending`] records: while one is owed the
    /// count is **not settled**, the readout says it is counting, and no total
    /// is shown. A budgeted count that still claimed to be settled would be the
    /// partial sum this whole pass exists to refuse.
    ///
    /// The composition guard is kept for symmetry with the scope; an unfocused
    /// editor cannot in fact be composing.
    ///
    /// Returns whether every retained draft is counted — `false` means it
    /// stopped for budget, never that a draft was skipped.
    fn count_retained_drafts(&mut self, budget: &mut ChunkBudget, cx: &mut Context<Self>) -> bool {
        let Some(query) = self.find.as_ref().and_then(|s| s.query.clone()) else {
            return true;
        };
        // Two passes, because deciding what each draft needs reads its editor
        // entity while doing the work needs `&mut self`.
        let mut plan: Vec<(SharedString, DraftWork)> = Vec::new();
        for draft in &self.drafts {
            if self
                .find
                .as_ref()
                .is_some_and(|s| s.branch_nodes.contains(&draft.id))
            {
                continue;
            }
            let editor = draft.editor.read(cx);
            if editor.value().is_empty() {
                plan.push((draft.id.clone(), DraftWork::Empty));
                continue;
            }
            let frozen = editor.is_composing();
            let cached = self
                .find
                .as_ref()
                .and_then(|s| s.projections.get(&draft.id));
            // The freshness question, asked without copying the draft's text:
            // a `SharedString` seed compares against the editor's `&str`
            // directly, so an unchanged draft never allocates its own content.
            let projection_fresh = cached.is_some_and(|c| {
                frozen
                    || (c.seed.content.as_ref() == editor.value()
                        && c.seed.render_cursor == Some(editor.selection())
                        && c.seed.embeds == EmbedMap::new(draft.embed_map()))
            });
            let memoized = projection_fresh
                && cached.is_some_and(|c| matches!(&c.hits, Some((q, _)) if q == &query));
            if memoized {
                plan.push((draft.id.clone(), DraftWork::Memoized));
                continue;
            }
            if frozen && !projection_fresh {
                // Composing with nothing cached: not searched yet, exactly as
                // in the scope. Leaving no entry is the honest state, and the
                // pass's own settling is untouched (drafts are not posts).
                continue;
            }
            // **The plan records that a seed is owed, never the seed.** The
            // freshness question above is already answered against *borrowed*
            // content (`c.seed.content.as_ref() == editor.value()`), so
            // nothing here has to copy anything to decide; building the owned
            // seed is deferred to admission, below.
            let bytes = editor.value().len();
            plan.push((
                draft.id.clone(),
                DraftWork::Scan {
                    stale: !projection_fresh,
                    bytes,
                },
            ));
        }
        for (node, work) in plan {
            let stale = match work {
                DraftWork::Empty => {
                    if let Some(session) = self.find.as_mut() {
                        session.space.counts.insert(node, 0);
                    }
                    continue;
                }
                DraftWork::Memoized => false,
                DraftWork::Scan { stale, bytes } => {
                    if budget.spent() {
                        // The rest are a later chunk's — their seeds unbuilt
                        // and their text uncopied — and the count says so
                        // rather than settling over them.
                        return false;
                    }
                    budget.scanned += bytes;
                    stale
                }
            };
            // **The copy happens here, past the budget check**, so a chunk
            // copies exactly the drafts it is going to project. Reading the
            // editor again is reading the same value the plan measured:
            // `count_chunk` has no await in it, so nothing can type between
            // the two loops.
            let seed = stale.then(|| self.draft_seed(&node, cx)).flatten();
            if let Some(seed) = seed {
                budget.built += 1;
                let projection =
                    searchable_projection(&seed.content, &seed.embeds, seed.render_cursor);
                self.projections_built.set(self.projections_built.get() + 1);
                if let Some(session) = self.find.as_mut() {
                    session.projections.insert(
                        node.clone(),
                        CachedProjection {
                            seed,
                            projection,
                            hits: None,
                        },
                    );
                }
            }
            let scans = &self.scans_run;
            if let Some(session) = self.find.as_mut() {
                let count = hits_of(&mut session.projections, &node, &query, scans).map(<[_]>::len);
                if let Some(count) = count {
                    session.space.counts.insert(node, count);
                }
            }
        }
        true
    }

    /// Build the owned seed for one retained draft — the copy, at the moment
    /// the chunk has agreed to pay for it.
    ///
    /// `None` for a draft that has left `self.drafts` since the plan was made,
    /// which cannot happen inside one `count_chunk` (nothing between the two
    /// loops mutates it) but is the honest answer rather than a panic. Looked
    /// up by **id** and not by the plan's position, the doctrine's own rule: a
    /// reference into a collection is an identity, never a slot.
    fn draft_seed(&self, node: &SharedString, cx: &gpui::App) -> Option<ProjectionSeed> {
        let draft = self.drafts.iter().find(|d| &d.id == node)?;
        let editor = draft.editor.read(cx);
        self.draft_seeds_built.set(self.draft_seeds_built.get() + 1);
        Some(ProjectionSeed {
            content: SharedString::from(editor.value().to_string()),
            embeds: EmbedMap::new(draft.embed_map()),
            // Every draft renders enabled, on its own branch as much as on the
            // selected one — the same reason `draft_scope_node` passes a cursor
            // rather than asking.
            render_cursor: Some(editor.selection()),
        })
    }

    /// The node ids a cached projection may be kept for: this frame's visible
    /// branch, **plus every post in the space**.
    ///
    /// The cache is the whole space's now, because a post off the selected
    /// branch is exactly what the cross-branch count is about — so the prune's
    /// membership set grew to match, and again for the **retained drafts** the
    /// count reaches ([`Self::count_retained_drafts`]); without them their
    /// projections would be dropped and rebuilt on every frame. It is still
    /// bounded by what exists, which is what keeps closing the bar the thing
    /// that drops every projection.
    fn live_projection_nodes(&self, branch: &HashSet<SharedString>) -> HashSet<SharedString> {
        let mut live: HashSet<SharedString> = (0..self.posts.len())
            .map(|i| super::model::node_id(&self.posts, i))
            .collect();
        live.extend(branch.iter().cloned());
        live.extend(self.drafts.iter().map(|d| d.id.clone()));
        live
    }

    /// What this frame's whole-space count would be an answer about.
    fn find_count_key(&self, cx: &gpui::App) -> CountKey {
        let revising: Vec<SharedString> = self
            .space
            .read(cx)
            .streams()
            .iter()
            .filter(|t| t.revising)
            .filter_map(|t| t.target_action_id.clone().map(SharedString::from))
            .collect();
        CountKey {
            query: self
                .find
                .as_ref()
                .filter(|s| s.query.is_some())
                .map(|s| SharedString::from(s.text.clone())),
            posts: self.posts_generation,
            revising,
        }
    }

    /// Bring the whole-space count up to date with this frame — restarting it
    /// when it is an answer about something else, and moving it along when it
    /// is not finished.
    ///
    /// **The frame gets one chunk and no more.** A conversation small enough —
    /// or one whose projections are all warm and whose remaining text is under
    /// [`SCAN_CHUNK_BYTES`] — finishes inside that chunk, which is the "below
    /// the threshold the scan stays on the frame" half of the cost model. One
    /// that does not finish hands the rest to a task on the session, a chunk
    /// per yield, and the readout says "counting" until it lands.
    fn sync_space_count(&mut self, cx: &mut Context<Self>) {
        let key = self.find_count_key(cx);
        let Some(session) = self.find.as_mut() else {
            return;
        };
        if session.space.key.as_ref() != Some(&key) {
            // Replace = cancel: whatever the pass in flight was counting, it
            // was counting something else.
            session.count_task = None;
            session.space.restart(key);
        }
        if session.query.is_none() {
            // Nothing to count. Settled before it starts, and the readout shows
            // no total at all rather than a zero.
            session.space.posts_done = true;
            session.space.drafts_pending = false;
            return;
        }
        // **No settled early-return.** The posts half remembers that it has
        // finished (`count_posts` returns at once), but a retained draft's text
        // moves with nothing in `CountKey` to show for it, so the drafts half
        // has to be re-asked on every frame — which is cheap by construction:
        // an unchanged draft is answered by the memo without a scan.
        if self.count_chunk(cx) {
            return;
        }
        // **The slot answers "is a worker running", so a finished one gives it
        // back** (below). A completed `Task` left sitting here reads exactly
        // like a live one, and this guard would then arm nothing for work a
        // later chunk deferred — the bar stuck on "Counting…" with no total
        // until some unrelated repaint happened along and drained it a chunk
        // at a time. Reachable because the drafts half is re-asked every frame:
        // the posts settle once, and a draft changing afterwards is what asks
        // for a second worker on a slot the first never returned.
        if self
            .find
            .as_ref()
            .is_some_and(|session| session.count_task.is_some())
        {
            return;
        }
        let task = cx.spawn(async move |this: gpui::WeakEntity<Self>, cx| {
            loop {
                // The suspension between chunks. See [`COUNT_YIELD`] for why it
                // cannot be zero. It is also what makes clearing the slot below
                // safe: the future always suspends before it can touch the
                // field, so it can never run ahead of the assignment that puts
                // it there.
                cx.background_executor().timer(COUNT_YIELD).await;
                let finished = this.update(cx, |this, cx| {
                    let finished = this.count_chunk(cx);
                    if finished && let Some(session) = this.find.as_mut() {
                        // **A finished worker hands its slot back, in the same
                        // update it finishes in** — `STATE.md`'s own shape
                        // (`this.balances_task = None` inside the task's
                        // continuation), and the reason that doctrine needs no
                        // generation counter: replace-cancels makes an occupied
                        // slot mean "the current operation", so a slot that
                        // outlives its operation is the one way the invariant
                        // can be broken. A flag beside the slot would answer
                        // the same question twice and is exactly the counter
                        // the doctrine retires. Dropping this task from inside
                        // itself is safe because nothing awaits after it: the
                        // `break` below ends the future in this same poll.
                        session.count_task = None;
                    }
                    // Every chunk moves a number the bar is showing — the
                    // progress while counting, the total when it lands.
                    cx.notify();
                    finished
                });
                if !matches!(finished, Ok(false)) {
                    break;
                }
            }
        });
        if let Some(session) = self.find.as_mut() {
            session.count_task = Some(task);
        }
    }

    /// One yield's worth of the whole-space pass — the posts, then the
    /// retained drafts, under one [`ChunkBudget`]. Returns whether both halves
    /// are finished.
    ///
    /// **The posts go first because that half remembers it is finished.** Once
    /// they are done [`Self::count_posts`] returns at once and spends nothing,
    /// so the steady state — a reader typing into a composer with the bar open
    /// — hands the drafts the whole allowance. The other order starves nobody
    /// either (nothing is shown until both halves land) but it lets a draft's
    /// spend exhaust the chunk *before* the posts are asked, which makes a
    /// chunk's stopping point unattributable to the half that caused it.
    fn count_chunk(&mut self, cx: &mut Context<Self>) -> bool {
        let mut budget = ChunkBudget::default();
        let posts_done = self.count_posts(&mut budget, cx);
        let drafts_done = self.count_retained_drafts(&mut budget, cx);
        if let Some(session) = self.find.as_mut() {
            session.space.drafts_pending = !drafts_done;
        }
        posts_done && drafts_done
    }

    /// The posts half of the pass. Returns whether the walk has reached the end
    /// of the space.
    ///
    /// A re-scan of a warm conversation under [`SCAN_CHUNK_BYTES`] finishes in
    /// the one chunk `sync_space_count` runs on the frame and never reaches the
    /// task at all — the "below the threshold the scan stays on the frame" half
    /// of the cost model.
    fn count_posts(&mut self, budget: &mut ChunkBudget, cx: &mut Context<Self>) -> bool {
        if self
            .find
            .as_ref()
            .is_some_and(|session| session.space.posts_done)
        {
            return true;
        }
        let Some(query) = self.find.as_ref().and_then(|s| s.query.clone()) else {
            return true;
        };
        loop {
            let Some(session) = self.find.as_ref() else {
                return true;
            };
            let i = session.space.next;
            if i >= self.posts.len() {
                if let Some(session) = self.find.as_mut() {
                    session.space.posts_done = true;
                }
                return true;
            }
            if budget.spent() {
                return false;
            }
            let node = super::model::node_id(&self.posts, i);
            let post = &self.posts[i];
            // The streaming rule, reaching the whole space: an answer being
            // replaced is not text the reader can see, so it is out of the
            // count exactly as it is out of the branch's match set. Its
            // exclusion is in `CountKey`, so the pass restarts when it moves.
            let excluded = post
                .action_id
                .as_deref()
                .is_some_and(|id| self.space.read(cx).revising_seq(id).is_some());
            let count = if excluded {
                0
            } else {
                let seed = ProjectionSeed {
                    content: post.content.clone(),
                    embeds: post_embed_map(post),
                    render_cursor: None,
                };
                // **The one node whose projection is not the cache's.** A post
                // under inline edit is cached as the reader has it — live
                // buffer, cursor-aware render — and that is what the branch
                // pass and the highlight layers read. Overwriting it with the
                // published render would make the two thrash each other every
                // frame, so the count's own projection is built and discarded
                // instead, memo and all. At most one node, once per pass.
                let editing = self.editing.as_ref().is_some_and(|e| e.node_id == node);
                let entry = self.find.as_ref().and_then(|s| s.projections.get(&node));
                let fresh = entry.is_some_and(|c| c.seed == seed);
                let memoized = !editing
                    && fresh
                    && entry.is_some_and(|c| matches!(&c.hits, Some((q, _)) if q == &query));
                // Bytes are charged where bytes are read: a post the memo can
                // answer for costs nothing, which is what lets a whole-space
                // re-count over warm projections and an unchanged query finish
                // on the frame its `CountKey` moved.
                if !memoized {
                    budget.scanned += post.content.len();
                }
                if editing {
                    budget.built += 1;
                    let projection = searchable_projection(&seed.content, &seed.embeds, None);
                    self.projections_built.set(self.projections_built.get() + 1);
                    self.scans_run.set(self.scans_run.get() + 1);
                    projection.find(&query).len()
                } else {
                    if !fresh {
                        budget.built += 1;
                        let projection = searchable_projection(&seed.content, &seed.embeds, None);
                        self.projections_built.set(self.projections_built.get() + 1);
                        if let Some(session) = self.find.as_mut() {
                            session.projections.insert(
                                node.clone(),
                                CachedProjection {
                                    seed,
                                    projection,
                                    hits: None,
                                },
                            );
                        }
                    }
                    let scans = &self.scans_run;
                    self.find
                        .as_mut()
                        .and_then(|s| {
                            hits_of(&mut s.projections, &node, &query, scans).map(<[_]>::len)
                        })
                        .unwrap_or(0)
                }
            };
            if let Some(session) = self.find.as_mut() {
                session.space.counts.insert(node, count);
                session.space.next = i + 1;
            }
        }
    }

    /// The nodes the search covers, in document order.
    ///
    /// Every post on the selected path, plus every draft that renders — a
    /// draft on the path, and the **active** one regardless of branch, since it
    /// floats over whatever is showing. A streaming leaf is deliberately out:
    /// its body grows token by token, so matching it would make the count and
    /// the reader's index jitter under their hand for no gain. It enters
    /// through the ordinary path when the turn finalizes and the transcript
    /// reloads.
    ///
    /// A post being **regenerated** is out on the same rule, and has to be
    /// said separately: that turn is the same thing wearing the other shape,
    /// rendering *in place of* the answer it replaces rather than as a leaf
    /// beneath it, so it never reaches the streaming arm below.
    fn find_scope(&self, tree: &[TreeNode], page_width: Pixels, cx: &gpui::App) -> FindScope {
        let mut scope: Vec<ScopeNode> = Vec::new();
        let mut seen: Vec<SharedString> = Vec::new();
        for (sibs, active) in self.selected_levels(tree, page_width) {
            let node = sibs[active];
            match node.src {
                NodeSrc::Msg(i) => {
                    let post = &self.posts[i];
                    // **A post being regenerated is out for the same reason a
                    // streaming leaf is, and it needs saying separately.** A
                    // revising turn is filtered out of the stream overlays
                    // rather than attached as a `NodeSrc::Streaming` leaf —
                    // the pending state renders *in place of* the answer it
                    // replaces, never as a child — so the exclusion below
                    // never covers it, and this arm went on projecting
                    // `post.content`. That text is not on screen at all while
                    // the revision runs: `render_post` swaps the whole value
                    // for `render_revision_body`. Searching it counted matches
                    // in an answer the reader cannot see and aimed the
                    // highlight layers at an editor that is no longer mounted,
                    // while the revision actually on screen went unsearched.
                    //
                    // **Not projected from the stream either**, which is the
                    // streaming rule itself: a body that grows token by token
                    // makes the count and the reader's index jitter under
                    // their hand for no gain. It comes back through the
                    // ordinary path when the turn finalizes and the transcript
                    // reloads.
                    if post
                        .action_id
                        .as_deref()
                        .is_some_and(|id| self.space.read(cx).revising_seq(id).is_some())
                    {
                        continue;
                    }
                    let embeds = post_embed_map(post);
                    let (content, frozen, render_cursor) = self.post_scope_text(&node.id, post, cx);
                    seen.push(node.id.clone());
                    scope.push(ScopeNode {
                        node: node.id.clone(),
                        item_id: post.item_id.clone(),
                        content,
                        embeds,
                        frozen,
                        render_cursor,
                    });
                }
                NodeSrc::Draft => {
                    if let Some(entry) = self.draft_scope_node(&node.id, cx) {
                        seen.push(node.id.clone());
                        scope.push(entry);
                    }
                }
                NodeSrc::Streaming(_) => {}
            }
        }
        // The active draft floats over whatever is showing, so it is in scope
        // even when its own branch is not the selected one — an active
        // composer for a draft belonging to another branch still matches.
        //
        // **And that is exactly the node the sibling aggregates must not
        // claim**, so the decision is reported rather than re-derived: this is
        // the one place that knows the draft joined *despite* the path not
        // carrying it. `seen` holds the selected path's own nodes, so an
        // active draft already on it is not floating and nothing is recorded.
        let mut floating = None;
        if let Some(active) = self.active_draft.clone()
            && !seen.contains(&active)
            && let Some(entry) = self.draft_scope_node(&active, cx)
        {
            floating = Some(entry.node.clone());
            scope.push(entry);
        }
        FindScope {
            nodes: scope,
            floating,
        }
    }

    /// The text to search one post's node, and whether it is frozen.
    ///
    /// **A post being edited is searched as the reader has it, not as the
    /// database has it.** An inline edit session's body editor deliberately
    /// keeps its unsaved buffer — `sync_bodies` skips the editing node, because
    /// that divergence *is* the edit — while `PostData::content` stays the
    /// generation the commit will replace. Searching the persisted text there
    /// reports source ranges of a string nobody is looking at, and the
    /// highlight layer paints them onto the modified buffer: insert or delete
    /// anything ahead of a match and the count describes old text while the
    /// wash lands on unrelated bytes.
    ///
    /// The editor is the same kind of live buffer a draft is, so it takes the
    /// same composition guard: a post mid-IME-composition is not re-projected,
    /// and the count keeps reporting the text the reader has committed (see
    /// [`ScopeNode::frozen`]).
    fn post_scope_text(
        &self,
        node: &SharedString,
        post: &super::model::PostData,
        cx: &gpui::App,
    ) -> (SharedString, bool, Option<Selection>) {
        let editing = self
            .editing
            .as_ref()
            .is_some_and(|e| &e.node_id == node)
            .then(|| self.bodies.get(node))
            .flatten();
        match editing {
            Some(editor) => {
                let editor = editor.read(cx);
                (
                    SharedString::from(editor.value().to_string()),
                    editor.is_composing(),
                    // **The mode is the one this render is about to paint,
                    // asked of the parent rather than of the child.** The
                    // editor's own `disabled` is an echo of the element prop,
                    // written when the child renders — which is *after*
                    // `sync_find` runs in the parent's own render, and without
                    // a notify behind it. On the frame an edit begins, asking
                    // the editor answered with the previous read-only frame,
                    // so the node was projected as published text while the
                    // reader looked at a cursor-aware editor, and nothing
                    // invalidated that until the caret or buffer moved. This
                    // branch is reached on exactly the predicate `post.rs`
                    // passes to `.disabled(!editing)`, so the node is enabled
                    // this frame by construction and its live selection is the
                    // cursor the render will use.
                    Some(editor.selection()),
                )
            }
            None => (post.content.clone(), false, None),
        }
    }

    fn draft_scope_node(&self, id: &SharedString, cx: &gpui::App) -> Option<ScopeNode> {
        let draft = self.drafts.iter().find(|d| &d.id == id)?;
        let editor = draft.editor.read(cx);
        Some(ScopeNode {
            node: draft.id.clone(),
            item_id: None,
            content: SharedString::from(editor.value().to_string()),
            embeds: EmbedMap::new(draft.embed_map()),
            frozen: editor.is_composing(),
            // Every draft renders enabled — the active composer and each
            // in-flow tail draft are built with no `.disabled(..)` at all —
            // so a draft in scope is always cursor-aware, and asking the
            // editor would take the same frame-late answer for no gain.
            render_cursor: Some(editor.selection()),
        })
    }

    /// The match ranges to paint on one node: the ordinary matches, and the
    /// current one on its own layer above them.
    ///
    /// Reads that node's own run and nothing else ([`MatchSet`]) — this runs
    /// once per body editor on every frame the bar is open, so scanning the
    /// whole set here made the paint `O(posts × matches)`.
    pub(crate) fn find_match_ranges(
        &self,
        node: &SharedString,
    ) -> (Vec<Range<usize>>, Vec<Range<usize>>) {
        let Some(session) = self.find.as_ref() else {
            return (Vec::new(), Vec::new());
        };
        let matches = session.matches.of(node);
        let current = session.matches.current_of(node, &session.anchor);
        let mut all = Vec::with_capacity(matches.len());
        let mut active = Vec::new();
        for (i, m) in matches.iter().enumerate() {
            if Some(i) == current {
                active.push(m.source.clone());
            } else {
                all.push(m.source.clone());
            }
        }
        (all, active)
    }

    /// The minimap's tick positions for one node: `(fraction, is_current)`.
    /// Per cell of the selected column, so it takes the same per-node read.
    pub(crate) fn find_ticks(&self, node: &SharedString) -> Vec<(f32, bool)> {
        let Some(session) = self.find.as_ref() else {
            return Vec::new();
        };
        let current = session.matches.current_of(node, &session.anchor);
        session
            .matches
            .of(node)
            .iter()
            .enumerate()
            .map(|(i, m)| (m.fraction, Some(i) == current))
            .collect()
    }

    /// One node's **own** match count, over the whole space.
    ///
    /// Two answers exist and they are not interchangeable, so the branch's
    /// wins wherever it exists: it reads what is on screen — an inline edit's
    /// unsaved buffer, a draft's live text — where the whole-space pass reads
    /// the persisted post. Everything derived from this (the total, a
    /// sibling's subtree count) therefore describes exactly the text the
    /// highlight layers paint, rather than two readings of one conversation.
    pub(crate) fn own_match_count(&self, node: &SharedString) -> usize {
        let Some(session) = self.find.as_ref() else {
            return 0;
        };
        if session.branch_nodes.contains(node) {
            return session.matches.of(node).len();
        }
        session.space.counts.get(node).copied().unwrap_or(0)
    }

    /// The matches reachable through one node: its own plus every
    /// descendant's — "*n* more matches reachable only through this branch".
    ///
    /// **Less the floating composer**, which is on screen rather than down
    /// there; see [`FindScope`]. Its count rejoins the sum on the visible side,
    /// in [`space_total`] and [`levels_account`].
    pub(crate) fn subtree_match_count(&self, node: &TreeNode) -> usize {
        subtree_matches(node, &|id| self.own_match_count(id), self.floating_node())
    }

    /// The node the search reaches only because its composer floats over the
    /// branch the reader is on, if there is one this frame.
    fn floating_node(&self) -> Option<&SharedString> {
        self.find.as_ref()?.floating.as_ref()
    }

    /// The whole space's total, or `None` while the pass is still walking.
    ///
    /// `None` is the honest in-progress state and the only alternative to a
    /// number: a partial sum reads on screen exactly like a settled one, and
    /// the previous query's total is the same lie with a longer fuse.
    ///
    /// One walk of the tree, once per frame a session is open — deliberately
    /// the *same* walk the map's sibling counts take, because it is that
    /// identity that makes the exactness invariant true by construction rather
    /// than by two definitions agreeing. It costs a hash lookup or two per post
    /// (~100 µs at a thousand posts), which buys a number that cannot disagree
    /// with the numbers beside it.
    pub(crate) fn find_space_total(&self, tree: &[TreeNode]) -> Option<usize> {
        let session = self.find.as_ref()?;
        session.space.is_settled().then(|| {
            space_total(
                tree,
                &|id| self.own_match_count(id),
                session.floating.as_ref(),
            )
        })
    }

    /// The count a minimap **sibling** cell carries: the matches reachable
    /// only through that branch. `None` while no session is open, while the
    /// pass has not settled, or where the branch holds nothing — a cell says
    /// nothing rather than "0".
    pub(crate) fn find_branch_count(&self, node: &TreeNode) -> Option<usize> {
        let session = self.find.as_ref()?;
        session
            .space
            .is_settled()
            .then(|| self.subtree_match_count(node))
            .filter(|n| *n > 0)
    }

    /// The left-hand side of the **exactness invariant**: the selected path's
    /// own matches, plus every shown sibling's whole subtree.
    ///
    /// This is what the minimap adds up in front of the reader — the active
    /// column's matches at each level and a number on each inactive one — and
    /// it must equal [`Self::find_space_total`] exactly, because
    /// `selected_levels` exhausts the space: take any post off the selected
    /// path, walk up to the first ancestor that is on it, and the post lies in
    /// the subtree of exactly one non-active child of that ancestor — which is
    /// exactly one of the siblings the map draws.
    pub(crate) fn find_levels_account(&self, tree: &[TreeNode], page_width: Pixels) -> usize {
        levels_account(
            &self.selected_levels(tree, page_width),
            &|id| self.own_match_count(id),
            self.floating_node(),
        )
    }

    /// Phase 1 of the reveal, from a caller with no tree in hand (the step
    /// verbs). Builds this frame's effective tree and delegates.
    fn reveal_find_anchor(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let page_width = self.page_size(window).width;
        let turns = self.stream_overlays(cx);
        let tree = self.effective_tree(page_width, &turns);
        self.reveal_current_match(&tree, page_width, window, cx);
    }

    /// Phase 1 of the reveal: take the reader to where the current match is
    /// *estimated* to be, and record the correction the second phase owes.
    ///
    /// Estimated because only posts intersecting the viewport render a real
    /// editor; everything else is a sized placeholder with no shaped lines and
    /// therefore no per-offset geometry.
    ///
    /// **Which surface scrolls is the match's own question**, not a constant:
    /// the branch never changes — that is what makes ⌘F safe in a tree — so a
    /// match on the selected path is a plain page scroll, while a match in the
    /// **off-branch active composer** is on no page the reader is looking at
    /// and only the composer's own viewport can bring it into view. See
    /// [`MatchReveal`].
    fn reveal_current_match(
        &mut self,
        tree: &[TreeNode],
        page_width: Pixels,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(m) = self.find.as_ref().and_then(|s| s.current().cloned()) else {
            return;
        };
        let window_h = self.page_size(window).height;
        let rem = window.rem_size();
        match self.match_reveal(tree, &m, page_width, window_h, rem, cx) {
            Some(MatchReveal::Page { top, bottom }) => {
                let y = self.find_reveal_offset(top, bottom, window_h);
                self.glide_page_to(y, window, cx);
            }
            Some(MatchReveal::Composer {
                top,
                bottom,
                natural,
            }) => {
                // **A reveal on another surface ends the page's motion.** Only
                // the page arm replaces a glide (`glide_page_to` writes the
                // new trajectory over the old one); every other arm leaves one
                // in flight, and a glide owns `page_scroll` for its whole
                // duration — so the conversation behind the composer went on
                // travelling toward a match that is no longer current while
                // the readout and the highlight had already moved to this one.
                self.cancel_page_glide();
                self.scroll_composer_to(top, bottom, natural, window_h);
            }
            // Nothing measured yet — the correction below is what lands it,
            // and the old motion still ends here: the page is travelling to
            // the *previous* match, and phase 2 stands aside for a glide in
            // flight, so leaving it would strand the correction behind it.
            None => self.cancel_page_glide(),
        }
        // **After** the motion, which itself counts as the reader being taken
        // somewhere and so clears any reveal already pending
        // (`demote_tail_pin_for_reader`).
        if let Some(session) = self.find.as_mut() {
            session.pending_reveal = Some(PendingReveal {
                node: m.node.clone(),
                source: m.source.clone(),
            });
        }
    }

    /// Phase 2: once the target post has rendered for real,
    /// `content_y_for_offset` answers and the position is corrected in place.
    /// The reader takes it back the moment they scroll, navigate or type — the
    /// pending reveal is dropped by the same seams that cancel a glide.
    fn correct_find_reveal(
        &mut self,
        tree: &[TreeNode],
        page_width: Pixels,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(pending) = self.find.as_ref().and_then(|s| s.pending_reveal.clone()) else {
            return;
        };
        // Nothing to correct until the editor has painted **the layout that is
        // on screen** — a stale answer is treated exactly as no answer, so the
        // reveal stays pending and the estimate stands until the post repaints.
        let editor = self
            .node_editor(&pending.node)
            .filter(|_| self.node_geometry_is_current(&pending.node));
        let answered = editor
            .as_ref()
            .and_then(|e| e.read(cx).content_y_for_offset(pending.source.start))
            .is_some();
        if !answered {
            return;
        }
        // **A match can be off to one side as well as off the top.** A fenced
        // code block and a table do not wrap — they clip behind a per-block
        // horizontal scroll — so revealing the match's *row* can leave the
        // match itself outside the strip the reader is looking through, named
        // in the readout and wearing a highlight nobody can see. The editor
        // answers for its own clip and moves nothing for a block that wraps or
        // a match already in view, so this is one call rather than a case
        // analysis here (`reveal_range_horizontally`).
        //
        // It runs beside the vertical correction rather than inside either
        // arm: which *surface* scrolls vertically is the page-versus-composer
        // question below, and the block's own horizontal band is the same
        // either way. It takes the whole span, because the thing that has to
        // be inside the clip is the match's last glyph, not its first.
        if let Some(editor) = editor {
            editor.update(cx, |e, cx| {
                e.reveal_range_horizontally(&pending.source, cx);
            });
        }
        let Some(m) = self.find.as_ref().and_then(|s| s.current().cloned()) else {
            return;
        };
        let window_h = self.page_size(window).height;
        let rem = window.rem_size();
        let Some(reveal) = self.match_reveal(tree, &m, page_width, window_h, rem, cx) else {
            return;
        };
        match reveal {
            MatchReveal::Page { top, bottom } => {
                // A glide already in flight owns the page; correcting under it
                // would be overwritten on its next frame, so the correction
                // waits for it — and the reveal stays pending, or the frame
                // after the glide would have nothing left to correct.
                if self.page_glide.get().is_some() {
                    return;
                }
                if let Some(session) = self.find.as_mut() {
                    session.pending_reveal = None;
                }
                let y = self.find_reveal_offset(top, bottom, window_h);
                self.set_page_scroll_y(y);
            }
            // A glide owns `page_scroll`, never the composer's own handle, so
            // this correction has nothing to wait for.
            MatchReveal::Composer {
                top,
                bottom,
                natural,
            } => {
                if let Some(session) = self.find.as_mut() {
                    session.pending_reveal = None;
                }
                self.scroll_composer_to(top, bottom, natural, window_h);
            }
        }
    }

    /// Scroll the **floating composer's own viewport** so an editor-content
    /// span is in view, with the same minimal-motion rule and margin the page
    /// reveal takes.
    ///
    /// The viewport is the editor's share of the floating bar: the bar less its
    /// top chrome and, in the compact scheme, the bottom action bar. The
    /// docked-byline inset is not a term — a composer whose match this reveals
    /// is off-branch, and an off-branch draft always floats
    /// (`layout::floating_pad`), where the dock reveal is zero.
    fn scroll_composer_to(&mut self, top: f32, bottom: f32, natural: f32, window_h: Pixels) {
        let body_h = (self.composer_float_bar_h(window_h)
            - Self::composer_chrome()
            - self.composer_gutters.get().bottom)
            .max(0.0);
        let scroll_max = (natural - body_h).max(0.0);
        let cur = self.composer_scroll.offset().y.as_f32();
        let next = super::composer::caret_scroll_offset(
            top,
            bottom,
            body_h,
            cur,
            scroll_max,
            FIND_REVEAL_MARGIN,
        );
        if (next - cur).abs() > 0.5 {
            let off = self.composer_scroll.offset();
            self.composer_scroll
                .set_offset(gpui::point(off.x, px(next)));
            // Keep the wheel handler's frozen-offset bookkeeping in step, so a
            // following `ScrollOwner::Body` wheel does not snap the composer
            // back to where it stood before the reveal (`caret_into_view`'s
            // own rule, for the same reason).
            self.composer_prev_off_y = next;
        }
    }

    /// The editor whose buffer a node's source offsets address — a post's body
    /// or a draft's composer.
    fn node_editor(
        &self,
        node: &SharedString,
    ) -> Option<&gpui::Entity<gpui_markdown_editor::MarkdownEditorState>> {
        self.bodies.get(node).or_else(|| {
            self.drafts
                .iter()
                .find(|d| &d.id == node)
                .map(|d| &d.editor)
        })
    }

    /// The page offset that brings a document span into view with the same
    /// minimal-motion rule keyboard navigation uses.
    fn find_reveal_offset(&self, top: f32, bottom: f32, window_h: Pixels) -> f32 {
        super::keyboard::scroll_into_view(
            top,
            bottom,
            super::keyboard::RevealViewport {
                height: window_h.as_f32(),
                // The find bar covers the document's top for its whole height.
                top_inset: self.doc_reserve(),
                bottom_inset: if self.active_draft.is_some() {
                    self.composer_float_bar_h(window_h)
                } else {
                    0.0
                },
            },
            self.page_scroll.offset().y.as_f32(),
            self.scroll_min_y.get(),
            FIND_REVEAL_MARGIN,
        )
    }

    /// Where a match is, and therefore **which surface has to move to show
    /// it**.
    ///
    /// The page answers for everything on the selected path, which is every
    /// match but one. The exception is the search scope's own deliberate
    /// exception: the **active draft is in scope regardless of branch**,
    /// because its composer floats over whatever is showing — and a node that
    /// is not on the selected path has no document top, so the page reveal
    /// could only ever decline. Stepping onto such a match then named it
    /// current in the readout while its highlight stayed wherever the
    /// composer's internal scroll had left it, out of the reader's sight in a
    /// long draft.
    ///
    /// **Scrolling the composer, deliberately, rather than selecting its
    /// branch.** Find never moves the reader's branch — that is the property
    /// that makes ⌘F safe in a tree, and the one the whole "visible branch"
    /// scope is built on. Selecting the draft's branch to reveal a match would
    /// spend it on the one node whose surface can already do the job: the
    /// floating composer has its own viewport and its own scroll handle, which
    /// is exactly what `caret_into_view` scrolls when an edit puts the caret
    /// below the fold.
    fn match_reveal(
        &self,
        tree: &[TreeNode],
        m: &Match,
        page_width: Pixels,
        window_h: Pixels,
        rem_size: Pixels,
        cx: &gpui::App,
    ) -> Option<MatchReveal> {
        let editor = self.node_editor(&m.node);
        let Some(top) = self.selected_path_doc_top(tree, &m.node, page_width, window_h) else {
            // Off the selected path: the only node that can be there is the
            // active floating draft (the scope admits no other), and its own
            // viewport is what shows it.
            if self.active_draft.as_ref() != Some(&m.node) {
                return None;
            }
            let editor = editor?.read(cx);
            let (t, b) = editor.content_y_for_offset(m.source.start)?;
            return Some(MatchReveal::Composer {
                top: t.as_f32(),
                bottom: b.as_f32(),
                natural: editor.content_height().as_f32(),
            });
        };
        let node = super::model::node_ref(tree, &m.node)?;
        let height = self.node_height(node, page_width, window_h);
        let pad = POST_PAD_Y.as_f32();
        let stacked = page_layout(page_width).gutters == GutterPlacement::Stacked;
        let metadata = if stacked {
            compact_gutter_occupancy(rem_size)
        } else {
            0.0
        };
        // Exact once the post's editor has painted (`content_y_for_offset`), an
        // honest estimate before that (the match's byte fraction of the node's
        // height). The editor's own top within the slot is `POST_PAD_Y`, plus
        // the stacked metadata row in the compact scheme — the same two terms
        // the docked composer's caret reveal folds in. **Accepted
        // imprecision:** a post whose reasoning disclosure is open carries that
        // disclosure above its body, so the exact arm lands a disclosure's
        // height high; the reveal margin absorbs it and the highlight is what
        // the reader is looking for.
        match editor
            .filter(|_| self.node_geometry_is_current(&m.node))
            .and_then(|e| e.read(cx).content_y_for_offset(m.source.start))
        {
            Some((t, b)) => {
                let base = top + pad + metadata;
                Some(MatchReveal::Page {
                    top: base + t.as_f32(),
                    bottom: base + b.as_f32(),
                })
            }
            None => {
                let body = (height - 2.0 * pad).max(1.0);
                let y = top + pad + m.fraction * body;
                Some(MatchReveal::Page {
                    top: y,
                    bottom: y + FIND_ESTIMATED_LINE_H,
                })
            }
        }
    }
}

/// Post-order accumulation over the render tree: a node's own matches plus
/// every descendant's.
///
/// The whole space is already in `SpaceView::posts` — `get_space_tree` is
/// `LIMIT`-free and branch-unfiltered — so a per-branch total is one walk over
/// the tree the view already builds, never a database question. The same shape
/// as `subtree_has_draft_content` beside it.
fn subtree_matches(
    node: &TreeNode,
    own: &impl Fn(&SharedString) -> usize,
    floating: Option<&SharedString>,
) -> usize {
    if floating == Some(&node.id) {
        // **The floating composer is on the visible side, so a branch it
        // happens to hang from does not get to claim it.** `effective_tree`
        // attaches the active draft under its own branch whatever the reader
        // is looking at, and this walk is what answers "reachable only through
        // this branch" — a phrase that has to mean *not currently visible*, or
        // the map offers the reader a trip to the one match already under
        // their eyes. Its own count is added once, on the visible side, by
        // [`space_total`] and [`levels_account`], so nothing is lost: this
        // moves the term, it does not drop it.
        //
        // Its descendants, if any, still count — the exclusion is one node,
        // the one the composer is painting.
        return node
            .children
            .iter()
            .map(|child| subtree_matches(child, own, floating))
            .sum();
    }
    own(&node.id)
        + node
            .children
            .iter()
            .map(|child| subtree_matches(child, own, floating))
            .sum::<usize>()
}

/// The right-hand side of the **exactness invariant**: every match in the
/// space, counted once.
///
/// The forest's own walk plus the floating composer, which that walk skips —
/// the one place the two halves of the split are put back together, so the
/// total and the map's account can only ever be read from one definition.
fn space_total(
    roots: &[TreeNode],
    own: &impl Fn(&SharedString) -> usize,
    floating: Option<&SharedString>,
) -> usize {
    roots
        .iter()
        .map(|root| subtree_matches(root, own, floating))
        .sum::<usize>()
        + floating.map_or(0, own)
}

/// The selected path's own matches plus every shown sibling's whole subtree —
/// what the minimap adds up in front of the reader, and the left-hand side of
/// the exactness invariant (see [`SpaceView::find_levels_account`]).
fn levels_account(
    levels: &[(Vec<&TreeNode>, usize)],
    own: &impl Fn(&SharedString) -> usize,
    floating: Option<&SharedString>,
) -> usize {
    // The floating composer is on screen, so the visible side owes it exactly
    // one term — and which term depends on where the levels put it. A draft
    // whose branch the reader is *not* on appears in no level (the sibling
    // subtrees deliberately skip it), so it is added here. One the reader
    // **is** on is an ordinary active node at its own level and is already
    // counted there; adding it again would be the double count this whole
    // split exists to remove, one step along.
    //
    // `find_scope` only ever records the first case, so the second is
    // unreachable through the view — but stating it makes the invariant true
    // for every path rather than for the paths production happens to produce,
    // which is what lets the exactness test walk all of them.
    let already_on_the_path = levels
        .iter()
        .any(|(sibs, active)| Some(&sibs[*active].id) == floating);
    let mut total = if already_on_the_path {
        0
    } else {
        floating.map_or(0, own)
    };
    for (sibs, active) in levels {
        for (i, sib) in sibs.iter().enumerate() {
            total += if i == *active {
                own(&sib.id)
            } else {
                subtree_matches(sib, own, floating)
            };
        }
    }
    total
}

/// Which surface a reveal moves, and the span it has to bring into view.
///
/// Two, because the search scope has two kinds of node in it: everything on the
/// selected path, which the page scrolls to, and the off-branch active draft,
/// which floats over the page and scrolls itself.
enum MatchReveal {
    /// A span in **document** space, revealed by the page.
    Page { top: f32, bottom: f32 },
    /// A span in the composer editor's own **content** space, revealed by the
    /// floating bar's internal scroll. `natural` is the editor's content
    /// height, which is what bounds that scroll.
    Composer { top: f32, bottom: f32, natural: f32 },
}

impl SpaceView {
    /// The floating bar.
    ///
    /// Its **surface** spans from the window top so it reads as one panel
    /// behind the traffic lights; its **controls** sit below
    /// [`TITLE_BAR_RESERVE`], where the drag band — registered after it, and so
    /// winning the hitboxes it covers — leaves them alone. It takes space
    /// rather than floating over the first post ([`SpaceView::doc_reserve`]
    /// grows by [`FIND_BAR_H`]), because a bar that covered the matches it is
    /// counting would be the one thing find must not do.
    pub(crate) fn render_find_bar(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<AnyElement> {
        self.find.as_ref()?;
        // The bar is *above* the overlay it opens, so its own controls stay in
        // the tab order while the overlay covers everything else
        // ([`crate::focus::Covered`]) — **unless something is covering the bar
        // in turn**. The inspector's overlay form paints a full-window scrim
        // after this whole pane, so lifting the guard unconditionally left the
        // bar's verbs reachable by Tab underneath a wash the reader cannot see
        // through. Which surfaces cover which is a function of the window's
        // width, so it is asked rather than assumed.
        let _uncovered = crate::focus::Covered::new(self.inspector_covers_pane(window));
        self.sync_find_placeholder(window, cx);
        let (bg, border, muted) = {
            let theme = cx.theme();
            (theme.background, theme.border, theme.muted_foreground)
        };
        let (index, total) = {
            let session = self.find.as_ref().expect("checked");
            (session.current_index(), session.matches.len())
        };
        let has_query = self.find.as_ref().is_some_and(|s| s.query.is_some());
        let input = self.find.as_ref().expect("checked").input.clone();
        let focus = self.find.as_ref().expect("checked").focus.clone();

        // **The cross-branch total, and its honest in-progress state.** A
        // different number from the index's: that one counts the branch the
        // reader is looking at, this one the whole conversation. While the
        // whole-space pass is still walking there is no number to show —
        // `None` from [`SpaceView::find_space_total`] — and the readout says
        // *that*, because a partial sum reads on screen exactly like a settled
        // one and the previous query's total is the same lie with a longer
        // fuse. A settled zero shows nothing at all: the index's own sentence
        // beside it has already said so, on both counts.
        let overlay_open = self.find_overlay_open();
        let space_total = self.find.as_ref().and_then(|s| s.space.total);
        let settled_total = space_total.filter(|n| *n > 0);
        let total_readout: Option<SharedString> = if !has_query {
            None
        } else {
            match space_total {
                Some(n) if n > 0 => Some(crate::i18n::msg::find_total(cx, n)),
                Some(_) => None,
                None => Some(crate::i18n::msg::find_total_counting(cx)),
            }
        };

        // The readout is a `Label` whose **label and value are the same short
        // sentence** — the notices' shape, so it starts speaking the day gpui
        // gains `aria_live` and is perceivable by review today.
        //
        // **The empty case is qualified by what the rest of the space holds.**
        // "No results" standing beside "3 total" is a contradiction on its
        // face, and worst of all as two `Label` nodes read one after the other,
        // where nothing places them in the same breath. The unqualified
        // sentence stays for the case it is true of — a query nothing anywhere
        // matches — and, deliberately, for the frames while the pass is still
        // walking: nothing is yet *known* to be elsewhere, and claiming a
        // branch is the empty one of many is a settled statement the count has
        // not made yet.
        let readout: SharedString = match (has_query, index) {
            (false, _) => SharedString::default(),
            (true, Some(i)) => crate::i18n::msg::find_count(cx, i, total),
            (true, None) if settled_total.is_some() => {
                crate::i18n::msg::find_no_results_in_branch(cx)
            }
            (true, None) => crate::i18n::msg::find_no_results(cx),
        };
        let steppable = total > 0;

        // **The controls stop where the map begins.** The minimap is painted
        // after the bar, so anything under it is covered — and it *widens* while
        // a session is open, precisely so a sibling column can hold a number, so
        // a flat inset would have put the new total readout behind the strip
        // exactly when the strip grew to meet it. Read from the one accessor,
        // the two cannot overlap whatever width the map takes next.
        let map_inset =
            px(self.minimap_width(self.page_size(window).width).as_f32() + BAR_EDGE_PAD);

        let controls = h_flex()
            .absolute()
            .top(TITLE_BAR_RESERVE)
            .left_0()
            .right_0()
            .h(px(FIND_BAR_H))
            .pl(px(BAR_EDGE_PAD))
            .pr(map_inset)
            .gap_2()
            .items_center()
            // The glyph a sighted reader sees and the sentence a screen reader
            // hears are different things — the × says nothing on its own.
            .child(crate::participants::ghost_button_labeled(
                "space-find-close".into(),
                "space/find/close".into(),
                "✕",
                crate::i18n::msg::find_close(cx),
                false,
                cx,
                cx.listener(|this, _, window, cx| {
                    this.close_find(window, cx);
                }),
            ))
            .child(
                div()
                    .id("space-find-field-wrap")
                    .flex_1()
                    .min_w_0()
                    // The `Input` owns the focus handle and is therefore the
                    // accessible node (the two-regime rule); the wrapper is
                    // bounds-only.
                    .probe_bounds(
                        "space/find/field",
                        gpui::Role::TextInput,
                        crate::i18n::msg::find_field_label(cx),
                    )
                    .child(
                        gpui_component::input::Input::new(&input)
                            .aria_label(crate::i18n::msg::find_field_label(cx)),
                    ),
            )
            .child(self.find_step_button(
                "space-find-prev",
                "space/find/previous",
                "‹",
                crate::i18n::msg::find_previous(cx),
                steppable,
                false,
                cx,
            ))
            .child(self.find_step_button(
                "space-find-next",
                "space/find/next",
                "›",
                crate::i18n::msg::find_next(cx),
                steppable,
                true,
                cx,
            ))
            .child(
                div()
                    .id("space-find-count")
                    .probe_value(
                        "space/find/count",
                        gpui::Role::Label,
                        readout.clone(),
                        readout.clone(),
                    )
                    .flex_none()
                    .min_w(px(64.))
                    .text_sm()
                    .text_color(muted)
                    .child(readout),
            )
            .children(total_readout.map(|sentence| {
                h_flex()
                    .id("space-find-total")
                    // One node, one sentence, label and value alike — the
                    // readout's own shape. The chevron beside it is painted
                    // separately and says nothing of its own.
                    .probe_value(
                        "space/find/total",
                        gpui::Role::Label,
                        sentence.clone(),
                        sentence.clone(),
                    )
                    .flex_none()
                    .gap_1()
                    .items_center()
                    .text_sm()
                    .text_color(muted)
                    .child(sentence)
                    // **The disclosure lives with the readout, not with the
                    // number.** Mounting it only on a settled total meant a
                    // background write that restarted a large count took the
                    // focused control out from under a keyboard reader for the
                    // length of the scan — ordinary keys reaching nothing, Tab
                    // recovering from a dead slot — and left no way to open the
                    // overlay while the pass ran. The overlay's own honest
                    // counting state is what it opens onto, so the control is
                    // as true while counting as after. What it is *not* offered
                    // for is a settled **zero**, which shows no readout either:
                    // that is the one state where there is nothing to show, and
                    // `total_readout` is already exactly that predicate.
                    .children(std::iter::once(()).map(|_| {
                        // **The disclosure is a real control now**, because it
                        // finally does something: it expands the Find-all
                        // overlay. It was a registry-only probe while the
                        // surface behind it did not exist — the step arrows'
                        // rule, that a `Role::Button` with no listener is a
                        // control VoiceOver offers, activates and silently does
                        // nothing with — and it becomes a `Button` in the same
                        // breath as the handler. Its name says what the click
                        // *does*, in both directions, because the glyph alone
                        // says nothing to a screen reader; the sentence beside
                        // it already speaks the number, so this one does not.
                        let open = overlay_open;
                        div()
                            .id("space-find-total-disclosure")
                            .probe(
                                "space/find/total/disclosure",
                                gpui::Role::Button,
                                if open {
                                    crate::i18n::msg::find_hide_all(cx)
                                } else {
                                    crate::i18n::msg::find_show_all(cx)
                                },
                            )
                            .aria_expanded(open)
                            .flex_none()
                            .px_1()
                            .rounded_sm()
                            .text_xs()
                            .cursor_pointer()
                            .text_color(muted.opacity(0.7))
                            .hover(|s| s.text_color(muted))
                            .child(if open { "▼" } else { "▲" })
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.toggle_find_overlay(window, cx);
                            }))
                    }))
            }));

        Some(
            crate::chrome::round_top_client_corners(div(), window)
                .id("space-find-bar")
                .track_focus(&focus)
                // Its own tab region, ahead of the conversation: the reader
                // opened this over the page and is acting in it, so Tab should
                // reach its verbs without walking the transcript first.
                .tab_region(crate::focus::region::FIND)
                .absolute()
                .top_0()
                .left_0()
                .right_0()
                .h(px(TITLE_BAR_RESERVE.as_f32() + FIND_BAR_H))
                .bg(bg)
                .border_b_1()
                .border_color(border)
                // Opaque, with nothing of its own to scroll.
                .contain_mouse(Overlay::Popover)
                .child(controls)
                .into_any_element(),
        )
    }

    /// One of the two step arrows. **One predicate decides the role, the
    /// tab-stopness and the activation**, so a bar that is focused when the
    /// last match disappears cannot keep a live `on_click` for a step that
    /// does nothing — nor a node claiming to be a button for one.
    ///
    /// The arrows deliberately keep painting with no results (a bar losing
    /// controls under the reader's hand is worse than dimmed ones), and that
    /// is exactly why the *role* has to move with the handler: this gpui rev
    /// exposes no `aria_disabled`, and `Window::handle_a11y_action` answers
    /// `Action::Click` by synthesizing a press at the node's centre — so a
    /// `Role::Button` left standing with no listener is a control VoiceOver
    /// offers, activates, and silently does nothing with. `Role::Label` keeps
    /// the glyph and its name readable while claiming nothing about pressing
    /// it, and derives neither focusability nor a tab stop (`focus::is_tab_stop`).
    /// It is `references::footnote_row`'s rule — the role tracks whether a
    /// handler attaches — reaching the one control in this bar that outlives
    /// its own verb.
    #[allow(clippy::too_many_arguments)]
    fn find_step_button(
        &self,
        id: &'static str,
        probe: &'static str,
        glyph: &'static str,
        aria: SharedString,
        enabled: bool,
        forward: bool,
        cx: &mut Context<Self>,
    ) -> gpui::Stateful<gpui::Div> {
        let theme = cx.theme();
        let role = if enabled {
            gpui::Role::Button
        } else {
            gpui::Role::Label
        };
        let el = div()
            .id(id)
            .probe(probe, role, aria)
            .flex_none()
            .px_2()
            .py_1()
            .rounded_md()
            .text_sm();
        if enabled {
            el.cursor_pointer()
                .text_color(theme.muted_foreground)
                .hover(|s| {
                    s.bg(theme.secondary.opacity(0.6))
                        .text_color(theme.foreground)
                })
                .child(glyph)
                .on_click(cx.listener(move |this, _, window, cx| {
                    this.find_step(forward, window, cx);
                }))
        } else {
            el.text_color(theme.muted_foreground.opacity(0.4))
                .tab_stop(false)
                .child(glyph)
        }
    }

    /// Re-seed the field's placeholder when the wording moves — a locale change
    /// refreshes every window, and the placeholder lives inside the field's
    /// state rather than being chosen at render.
    fn sync_find_placeholder(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let fresh = crate::i18n::msg::find_placeholder(cx);
        let Some(session) = self.find.as_mut() else {
            return;
        };
        if session.placeholder == fresh {
            return;
        }
        session.placeholder = fresh.clone();
        let input = session.input.clone();
        input.update(cx, |s, cx| s.set_placeholder(fresh, window, cx));
    }
}

/// How much of the viewport to keep clear around a revealed match — the
/// keyboard reveal's margin, for the same reason: a match flush against the
/// fold reads as cut off.
const FIND_REVEAL_MARGIN: f32 = 24.0;

/// The height an unmeasured match is assumed to occupy, for the estimated
/// first phase only. One prose line is close enough to place the scroll; the
/// correction replaces it as soon as the post paints.
const FIND_ESTIMATED_LINE_H: f32 = 28.0;

/// The bar's own breathing room at each end of its control row — `px_3`'s
/// value, named because the right-hand inset is the minimap's width *plus*
/// this rather than this alone.
const BAR_EDGE_PAD: f32 = 12.0;

/// How long the cross-branch pass waits between chunks.
///
/// **It cannot be zero, and a zero would be silent.** At the pinned gpui
/// revision `BackgroundExecutor::timer` short-circuits a zero duration to
/// `Task::ready(())` (`gpui/src/executor.rs:162`), and awaiting a ready task
/// never returns `Pending` — so a "yield" of zero suspends nothing, and the
/// loop runs every remaining chunk in one poll on the main executor: a long
/// conversation freezes input until it is fully counted, which is exactly what
/// bounding the work per yield exists to avoid. Nothing about the code would
/// look wrong; only the wall clock would.
///
/// A millisecond is this codebase's shape for a real suspension (the theme's
/// clock tick, the minimap's hide delay, the engine teardown grace), and it
/// paces a thousand-post conversation across tens of yields rather than one
/// long stall. It also makes the unfinished state observable: gpui's test
/// scheduler expires a timer only once the clock reaches it, so
/// `run_until_parked` leaves the pass suspended and a test can assert that it
/// *is* suspended before advancing the clock to let it land.
const COUNT_YIELD: std::time::Duration = std::time::Duration::from_millis(1);

/// How many posts one chunk of the whole-space pass will **project**.
///
/// Measured on an M-series Mac over 200 synthetic posts averaging 1.1 KB of
/// ordinary markdown (headings, a link, emphasis, a list, fenced text), the
/// projection being `parse` + `render_readonly` + the walk in
/// [`searchable_projection`]:
///
/// | build | `--release` | dev profile |
/// |---|---|---|
/// | per post | 11.1 µs | 34.4 µs |
/// | per byte | 10.1 ns | 31.2 ns |
///
/// Sixteen posts is ~0.18 ms released (~0.55 ms in dev) at that size, and ~0.35
/// ms for the 2 KB posts the cost model is written against — comfortably inside
/// a frame beside everything else the frame does, while still coarse enough
/// that a thousand-post conversation settles in tens of yields rather than
/// hundreds. Counted in posts rather than bytes because the parse dominates and
/// scales with structure, not length; the residual is a single pathological
/// post, which costs its own chunk and nothing more.
const PROJECTION_CHUNK: usize = 16;

/// How many bytes of post text one chunk will **scan** — and therefore the
/// threshold under which a whole re-scan stays on the frame, because
/// [`SpaceView::sync_space_count`] runs one chunk there before it arms the
/// task.
///
/// Same corpus, scanning warm projections for a lower-case (so
/// case-insensitive, so case-folding) query:
///
/// | scan | `--release` | dev profile |
/// |---|---|---|
/// | per byte | 8.5 ns | 147 ns |
///
/// 128 KB is therefore ~1.1 ms released — one frame's worth of a keystroke, on
/// the keystroke that asked for it. A conversation under that re-counts inside
/// the frame its query changed on and never shows the counting state at all; a
/// 2 MB one takes sixteen yields and does. (The dev profile is ~17× slower here
/// because the fold is a per-character walk in an unoptimized crate; the
/// threshold serves the shipped binary, and dev pays a dropped frame, which is
/// the trade the dev profile already documents.)
const SCAN_CHUNK_BYTES: usize = 128 * 1024;

#[cfg(test)]
mod tests {
    use super::*;

    fn project(source: &str) -> Projection {
        searchable_projection(source, &EmbedMap::default(), None).projection
    }

    /// The projection of a node whose editor is *enabled* with its cursor at
    /// `at` — an inline edit or a draft, which is what the reader is looking
    /// at when they search one.
    fn project_editing(source: &str, at: usize) -> Projection {
        searchable_projection(source, &EmbedMap::default(), Some(Selection::Cursor(at))).projection
    }

    /// How many math overlays the read-only render puts on this source.
    ///
    /// Asserted beside the projection because "no match here" is also what an
    /// input that never parsed as math produces — the two are only told apart
    /// by whether an overlay exists for the projection to judge.
    fn overlay_count(source: &str) -> usize {
        let state = gpui_markdown_editor::EditorState::with_markdown(source);
        let tree = gpui_markdown_editor::parse(&state.markdown);
        let spec = gpui_markdown_editor::render::render_readonly(&state, &tree);
        spec.blocks.iter().map(|b| b.math_overlays.len()).sum()
    }

    fn find_editing(source: &str, at: usize, query: &str) -> Vec<String> {
        let projection = project_editing(source, at);
        let query = Query::new(query).expect("non-empty");
        projection
            .find(&query)
            .into_iter()
            .map(|r| source[r].to_string())
            .collect()
    }

    fn find(source: &str, query: &str) -> Vec<String> {
        let projection = project(source);
        let query = Query::new(query).expect("non-empty");
        projection
            .find(&query)
            .into_iter()
            .map(|r| source[r].to_string())
            .collect()
    }

    #[test]
    fn a_links_url_is_not_searchable_but_its_text_is() {
        let source = "See [the report](https://performance.example/perf).";
        assert!(!project(source).text().contains("performance.example"));
        assert!(find(source, "performance").is_empty());
        assert_eq!(find(source, "report"), vec!["report".to_string()]);
    }

    #[test]
    fn a_phrase_crossing_an_emphasis_delimiter_matches_what_the_reader_sees() {
        let source = "a very **important** thing";
        // The naive source scan finds nothing here — the delimiters are in the
        // way — while the reader plainly sees the phrase.
        assert!(!source.contains("important thing"));
        let hits = find(source, "important thing");
        assert_eq!(hits.len(), 1);
        // Mapped back through the projection the source range covers the
        // delimiters it spans, which is the honest direction: it contains
        // everything the reader was shown.
        assert!(hits[0].contains("important"));
        assert!(hits[0].ends_with("thing"));
    }

    #[test]
    fn an_entity_matches_as_the_character_it_renders_as() {
        let source = "Tom &amp; Jerry";
        assert_eq!(project(source).text(), "Tom & Jerry");
        assert_eq!(find(source, "m & j"), vec!["m &amp; J".to_string()]);
    }

    #[test]
    fn inline_math_and_image_markup_are_not_searchable() {
        // Neither carries a hidden range in the render spec — the element
        // layer suppresses them — so a projection reading `hidden_ranges`
        // alone would make both matchable. This is the arm that pins the
        // projection excluding them itself.
        let math = "before $\\alpha_{beta}$ after";
        assert!(find(math, "beta").is_empty());
        assert_eq!(find(math, "before"), vec!["before".to_string()]);

        let image = "look ![a red kite](https://example/kite.png) here";
        assert!(find(image, "kite").is_empty());
        assert!(find(image, "red").is_empty());
        assert_eq!(find(image, "look"), vec!["look".to_string()]);
    }

    #[test]
    fn math_that_does_not_typeset_is_searchable_as_the_source_the_reader_sees() {
        // The element layer suppresses a math construct's source bytes only by
        // substituting a pad run for typeset math. When the LaTeX does not
        // typeset it shapes the raw `$…$` instead — dim delimiters, mono
        // content — so every one of those bytes is on screen, and excluding
        // them reported no match on text plainly visible.
        // `\\frac` takes two arguments, so RaTeX rejects this — while the
        // construct still parses as math, which is what puts an overlay on the
        // block for the projection to judge.
        let malformed = "before $\\frac{beta}$ after";
        assert!(overlay_count(malformed) == 1);
        assert!(project(malformed).text().contains("$\\frac{beta}$"));
        assert_eq!(find(malformed, "beta"), vec!["beta".to_string()]);
        // The delimiters are shown too, and the projection maps a match back
        // to the source bytes it copied — no substitution is involved.
        assert_eq!(find(malformed, "$\\frac"), vec!["$\\frac".to_string()]);
        assert_eq!(find(malformed, "before"), vec!["before".to_string()]);

        // Math that typesets still shows nothing of its source.
        let typeset = "before $\\alpha_{beta}$ after";
        assert!(overlay_count(typeset) == 1);
        assert!(!project(typeset).text().contains('$'));
        assert!(find(typeset, "beta").is_empty());
    }

    #[test]
    fn a_sole_image_paragraph_is_not_searchable() {
        // The promotion carries neither a hidden range nor an overlay — the
        // element layer hides it once the image loads — so the render spec
        // offers nothing to exclude and the whole markup was copied.
        let source = "![a red kite](https://example/kite.png)";
        assert!(project(source).text().trim().is_empty());
        assert!(find(source, "kite").is_empty());
        assert!(find(source, "example").is_empty());

        // A paragraph that merely *contains* an image keeps its prose, and
        // the image is excluded by the inline-overlay arm as before.
        let inline = "look ![a red kite](https://example/kite.png) here";
        assert_eq!(find(inline, "look"), vec!["look".to_string()]);
        assert!(find(inline, "kite").is_empty());
    }

    #[test]
    fn the_cell_lookup_answers_what_a_scan_would() {
        // `cell_at` replaced two linear scans per hidden range — which made a
        // table quadratic in its cells — with one `partition_point`. Only the
        // cost moved, so the lookup is pinned against the scan it stands in
        // for, over every byte of a grid with gaps at both ends, between every
        // pair of cells, and one cell butting straight onto the next.
        let cells = vec![2..5, 5..9, 12..14, 20..20, 21..30];
        for pos in 0..34 {
            let inside = cells.iter().find(|c| c.start <= pos && pos < c.end);
            let expected = match inside {
                Some(cell) => CellAt::Inside(cell.end),
                None => CellAt::Between(
                    cells
                        .iter()
                        .map(|c| c.start)
                        .filter(|start| *start > pos)
                        .min(),
                ),
            };
            assert_eq!(cell_at(&cells, pos), expected, "at byte {pos}");
        }
        // A block with no grid at all takes the between-cells arm to its end.
        assert_eq!(cell_at(&[], 7), CellAt::Between(None));
    }

    #[test]
    fn a_rendered_atom_keeps_the_text_either_side_of_it_apart() {
        // The element layer replaces a typeset formula and an inline image
        // with something that occupies width, so the text either side of one
        // is two things the reader sees apart. Excluding the source span
        // zero-width joined them into a word that is nowhere on the page —
        // counted in the readout and painted by the highlight layer — which is
        // the table chrome's mistake reached by a third door.
        let math = "left$\\alpha_{beta}$right";
        assert_eq!(overlay_count(math), 1, "the formula really is an overlay");
        assert!(
            find(math, "leftright").is_empty(),
            "the formula stands between them on the page"
        );
        assert_eq!(find(math, "left"), vec!["left".to_string()]);
        assert_eq!(find(math, "right"), vec!["right".to_string()]);

        let image = "left![a red kite](https://example/kite.png)right";
        assert!(
            find(image, "leftright").is_empty(),
            "the picture stands between them on the page"
        );
        assert_eq!(find(image, "left"), vec!["left".to_string()]);
        assert_eq!(find(image, "right"), vec!["right".to_string()]);
    }

    #[test]
    fn a_substitution_inside_a_merged_hide_is_still_searchable() {
        // `merge_hidden_ranges` coalesces the emphasis delimiters with the
        // entity's own hide into one range starting at byte 0, and
        // `build_display_line` interrupts that hide at the substitution — so
        // the reader sees `&`, and a hide taken whole made it unsearchable.
        let source = "**&amp;**";
        assert_eq!(project(source).text(), "&");
        assert_eq!(find(source, "&"), vec!["&amp;".to_string()]);

        let embedded = "a **&amp;** b";
        assert_eq!(project(embedded).text(), "a & b");
        assert_eq!(find(embedded, "a & b"), vec![embedded.to_string()]);

        // The same merge at a table cell's edge, where the hide is split by
        // the cell rule first: the two rules compose rather than race.
        let table = "| a `x`&amp;y | b |\n| --- | --- |\n| 1 | 2 |";
        assert!(project(table).text().contains("x&y"));

        // **But an overlay is atomic.** An entity in an image's alt text is a
        // substitution inside a range the element layer replaces wholesale,
        // so it never reaches the page and must stay unsearchable — the hide
        // rule above must not reach into an overlay. The overlay projects as
        // the barrier every replaced-atom span takes (the picture stands
        // between the words either side of it), which is what the newline is.
        let alt = "look ![a &amp; b](https://e/k.png) here";
        assert_eq!(project(alt).text(), "look \n here");
        assert!(find(alt, "&").is_empty());
    }

    #[test]
    fn an_editable_node_is_projected_with_the_render_mode_it_shows() {
        // A node the reader is editing keeps an *enabled* editor, and an
        // enabled editor renders cursor-aware — so both directions of the
        // read-only/live divergence are visible on the page.
        let link = "The [survey](https://kestrel.example/data) says so.";
        // Published: the URL is hidden, and deliberately unmatchable.
        assert!(find(link, "kestrel.example").is_empty());
        // Editing, caret in the link text: the reader plainly sees the URL.
        assert!(
            project_editing(link, 6)
                .text()
                .contains("https://kestrel.example/data")
        );
        assert_eq!(
            find_editing(link, 6, "kestrel.example"),
            vec!["kestrel.example".to_string()]
        );

        // The other direction: a phrase that only closes up once the
        // delimiters hide must stop matching when they are revealed.
        let emph = "a very **important** thing";
        assert_eq!(find(emph, "important thing").len(), 1);
        assert!(find_editing(emph, 10, "important thing").is_empty());
        assert!(project_editing(emph, 10).text().contains("**important**"));
    }

    #[test]
    fn a_mapped_embed_marker_is_not_searchable_and_an_unmapped_one_is() {
        let source = "{{ embed 1 }}";
        let mapped =
            searchable_projection(source, &EmbedMap::new([(1, "quoted".to_string())]), None)
                .projection;
        assert!(mapped.text().trim().is_empty());
        // An ordinal with no reference behind it is ordinary text — which is
        // also how a marker looks before its reference exists.
        let unmapped = project(source);
        assert!(unmapped.text().contains("embed"));
    }

    #[test]
    fn a_query_never_matches_across_two_blocks() {
        // Nothing separates two paragraphs in the projection but the barrier
        // the builder copies out of the gap; without it `endstart` would be
        // one match spanning a blank line.
        let source = "the end\n\nstart of the next";
        assert!(find(source, "endstart").is_empty());
        assert_eq!(find(source, "the end"), vec!["the end".to_string()]);
    }

    #[test]
    fn a_code_block_is_searchable_and_its_fence_is_not() {
        let source = "```rust\nlet performance = 1;\n```";
        assert_eq!(find(source, "performance"), vec!["performance".to_string()],);
        assert!(find(source, "```").is_empty());
    }

    #[test]
    fn a_table_cell_is_searchable() {
        let source = "| Configuration | Performance |\n| --- | --- |\n| 1x B200 | Moderate |\n";
        assert_eq!(find(source, "performance"), vec!["Performance".to_string()],);
    }

    #[test]
    fn a_query_never_matches_across_a_table_cell_boundary() {
        // The grid's chrome is hidden but it is not *inline* markup: the reader
        // sees two cells, so `leftright` is a phrase nobody can point at. Only
        // the delimiters that really are inline (emphasis, a link's brackets)
        // may close up.
        let source = "| left | right |\n| --- | --- |\n| one | two |\n";
        assert!(find(source, "leftright").is_empty(), "across a column");
        assert!(find(source, "rightone").is_empty(), "across a row");
        assert!(find(source, "onetwo").is_empty(), "and in the body too");
        // The cells themselves are still ordinary searchable text.
        assert_eq!(find(source, "right"), vec!["right".to_string()]);
        assert_eq!(find(source, "two"), vec!["two".to_string()]);
    }

    #[test]
    fn a_barrier_survives_a_substitution_merged_into_it_at_the_cell_edge() {
        // `merge_hidden_ranges` joins an entity's own hidden bytes to the
        // chrome beside it, so one hidden range straddles the cell edge:
        // classified whole it is either all barrier (losing the `&` the reader
        // sees) or all deletion (losing the boundary). Both halves are checked
        // here, at both edges of a cell.
        let trailing = "| left&amp; | right |\n| --- | --- |\n| a | b |\n";
        assert!(
            find(trailing, "left&right").is_empty(),
            "the barrier survives the entity merged into it"
        );
        assert_eq!(
            find(trailing, "left&"),
            vec!["left&amp;".to_string()],
            "…and the entity still displays as the character it renders as"
        );

        let leading = "| left | &amp;right |\n| --- | --- |\n| a | b |\n";
        assert!(
            find(leading, "left&right").is_empty(),
            "and at the other edge, where the chrome comes first"
        );
        assert_eq!(find(leading, "&right"), vec!["&amp;right".to_string()]);
    }

    #[test]
    fn inline_markup_inside_a_cell_still_closes_up() {
        // The other half of the same rule: a barrier is *structural* chrome,
        // and emphasis inside a cell is not — a phrase crossing it matches
        // exactly as it does in a paragraph.
        let source = "| a **bold** claim | second |\n| --- | --- |\n| x | y |\n";
        let hits = find(source, "bold claim");
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert!(hits[0].ends_with("claim"));
    }

    #[test]
    fn a_heading_matches_without_its_hashes() {
        let source = "## Deployment Comparison Table";
        assert!(find(source, "# Deployment").is_empty());
        assert_eq!(find(source, "Deployment"), vec!["Deployment".to_string()],);
    }

    #[test]
    fn a_list_item_matches_without_its_marker() {
        let source = "- alpha\n- beta\n";
        assert_eq!(find(source, "alpha"), vec!["alpha".to_string()]);
        assert!(find(source, "- alpha").is_empty());
    }

    #[test]
    fn every_reported_range_is_a_range_of_the_source() {
        // The whole point of routing through the offset map: a reported range
        // can always be sliced. A projection whose runs were recorded against
        // stale spans would panic here.
        let source = "# Title\n\nSee [x](https://e/y) and `code` and *em* and &amp; ok.\n\n\
                      - one\n- two\n\n```\nfenced text\n```\n";
        let projection = project(source);
        for query in ["e", "o", "t", " "] {
            let Some(query) = Query::new(query) else {
                continue;
            };
            for range in projection.find(&query) {
                assert!(range.end <= source.len(), "{range:?}");
                let _ = &source[range];
            }
        }
    }

    #[test]
    fn the_projection_prune_asks_a_set_what_the_scope_holds() {
        // `sync_find` runs every frame the bar is open and prunes the cache
        // before it looks at the query, so scanning the scope per cached entry
        // was `O(posts²)` on an unchanged conversation. The membership set is
        // what makes it one lookup — and it *is* a set, so a revert to the scan
        // has nothing to call.
        let scope: Vec<ScopeNode> = ["a1", "a2", "draft-1"]
            .into_iter()
            .map(|node| ScopeNode {
                node: node.into(),
                item_id: None,
                content: SharedString::default(),
                embeds: EmbedMap::default(),
                frozen: false,
                render_cursor: None,
            })
            .collect();
        let live = live_scope_nodes(&scope);

        // Exactly the scope's nodes survive — what the scan selected.
        assert_eq!(live.len(), scope.len(), "one entry per scope node");
        for entry in &scope {
            assert!(live.contains(&entry.node), "{} is in scope", entry.node);
        }
        // And a node the branch no longer carries does not, which is what
        // keeps the cache bounded by the scope.
        assert!(!live.contains(&SharedString::from("a3")));
    }

    /// A node with children, for the accumulation tests.
    fn node(id: &str, children: Vec<TreeNode>) -> TreeNode {
        TreeNode {
            src: NodeSrc::Msg(0),
            id: id.into(),
            children,
        }
    }

    /// A deliberately gnarly forest: two thread roots, forks at every depth,
    /// a level whose active child is itself forked, and leaves at three
    /// different depths.
    fn gnarly_forest() -> Vec<TreeNode> {
        vec![
            node(
                "r1",
                vec![
                    node(
                        "a",
                        vec![node("a1", vec![node("a1a", vec![])]), node("a2", vec![])],
                    ),
                    node(
                        "b",
                        vec![
                            node("b1", vec![]),
                            node("b2", vec![node("b2a", vec![]), node("b2b", vec![])]),
                        ],
                    ),
                    node("c", vec![]),
                ],
            ),
            node("r2", vec![node("r2a", vec![])]),
        ]
    }

    /// `selected_levels`' rule, stated over an explicit list of active
    /// indices: level 0 is the roots, and each level after it is the previous
    /// level's active node's children.
    fn levels_of<'a>(roots: &'a [TreeNode], actives: &[usize]) -> Vec<(Vec<&'a TreeNode>, usize)> {
        let mut levels: Vec<(Vec<&'a TreeNode>, usize)> = Vec::new();
        if roots.is_empty() {
            return levels;
        }
        let mut active = actives.first().copied().unwrap_or(0) % roots.len();
        levels.push((roots.iter().collect(), active));
        let mut node = &roots[active];
        let mut depth = 1;
        while !node.children.is_empty() {
            active = actives.get(depth).copied().unwrap_or(0) % node.children.len();
            levels.push((node.children.iter().collect(), active));
            node = &node.children[active];
            depth += 1;
        }
        levels
    }

    #[test]
    fn the_selected_path_and_its_shown_siblings_account_for_the_whole_space() {
        // **The exactness invariant.** `selected_levels` gives, at level 0, all
        // thread roots, and at each level after it all children of the previous
        // level's active node. Take any post off the selected path and walk up
        // to the first ancestor that is on it: the post lies in the subtree of
        // exactly one *non-active* child of that ancestor — which is exactly one
        // of the siblings the minimap draws. So the path's own matches plus
        // those siblings' whole subtrees is the space's total, exactly and
        // without double counting.
        let roots = gnarly_forest();
        // Uneven counts, several zeroes, and a node with matches at every
        // depth — so a walk that skipped a level or double-counted one could
        // not come out equal by luck.
        let counts: HashMap<&str, usize> = [
            ("r1", 2),
            ("a", 3),
            ("a1", 1),
            ("a1a", 4),
            ("a2", 0),
            ("b", 5),
            ("b1", 0),
            ("b2", 2),
            ("b2a", 7),
            ("b2b", 1),
            ("c", 0),
            ("r2", 6),
            ("r2a", 9),
        ]
        .into_iter()
        .collect();
        let own = |id: &SharedString| counts.get(id.as_ref()).copied().unwrap_or(0);
        let total = space_total(&roots, &own, None);
        assert_eq!(
            total,
            counts.values().sum::<usize>(),
            "the forest is the space"
        );

        // Every branch the reader could be on, not one of them.
        let mut checked = 0;
        for r in 0..2 {
            for l1 in 0..3 {
                for l2 in 0..3 {
                    for l3 in 0..3 {
                        let levels = levels_of(&roots, &[r, l1, l2, l3]);
                        assert_eq!(
                            levels_account(&levels, &own, None),
                            total,
                            "path {:?} does not account for the space",
                            levels
                                .iter()
                                .map(|(sibs, active)| sibs[*active].id.as_ref())
                                .collect::<Vec<_>>()
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert_eq!(checked, 54, "every combination was walked");

        // **And the same walk with the off-branch composer floating over it.**
        // `b2b` hangs under `b`, which is a sibling of the `a` the reader is
        // on — so the tree puts it *down there* while the composer paints it
        // *right here*. The invariant has to hold with it on the visible side:
        // its matches are in the total exactly once, they are in every path's
        // account exactly once, and the branch it belongs to no longer offers
        // them as somewhere to go.
        let composer = SharedString::from("b2b");
        let floated = Some(&composer);
        let total_floating = space_total(&roots, &own, floated);
        assert_eq!(
            total_floating, total,
            "moving the composer to the visible side changes no total — the \
             draft is counted once either way"
        );
        let b = &roots[0].children[1];
        assert_eq!(b.id, SharedString::from("b"));
        assert_eq!(
            subtree_matches(b, &own, floated),
            subtree_matches(b, &own, None) - own(&composer),
            "the branch the composer belongs to stops claiming what is on \
             screen — 'only through this branch' means not currently visible"
        );
        let mut checked_floating = 0;
        for r in 0..2 {
            for l1 in 0..3 {
                for l2 in 0..3 {
                    for l3 in 0..3 {
                        let levels = levels_of(&roots, &[r, l1, l2, l3]);
                        assert_eq!(
                            levels_account(&levels, &own, floated),
                            total_floating,
                            "path {:?} does not account for the space with a \
                             floating composer",
                            levels
                                .iter()
                                .map(|(sibs, active)| sibs[*active].id.as_ref())
                                .collect::<Vec<_>>()
                        );
                        checked_floating += 1;
                    }
                }
            }
        }
        assert_eq!(checked_floating, 54);
    }

    #[test]
    fn a_subtree_count_is_a_node_and_everything_under_it() {
        let roots = gnarly_forest();
        let counts: HashMap<&str, usize> = [("b", 5), ("b1", 0), ("b2", 2), ("b2a", 7), ("b2b", 1)]
            .into_iter()
            .collect();
        let own = |id: &SharedString| counts.get(id.as_ref()).copied().unwrap_or(0);
        let b = &roots[0].children[1];
        assert_eq!(b.id, SharedString::from("b"));
        assert_eq!(
            subtree_matches(b, &own, None),
            15,
            "5 of its own plus 0 + 2 + 7 + 1 beneath it"
        );
        // A leaf is its own subtree, and a node nothing matched in contributes
        // nothing rather than being absent.
        assert_eq!(subtree_matches(&roots[1], &own, None), 0);
        // Excluding the floating composer drops that one node and nothing
        // under it — `b2` keeps its own 2 and its other child's 1.
        assert_eq!(
            subtree_matches(b, &own, Some(&SharedString::from("b2a"))),
            8,
            "the excluded node's 7 leaves; its siblings and ancestors stay"
        );
    }

    #[test]
    fn a_restarted_count_is_not_a_settled_one() {
        // The honest-states rule at its smallest: whatever the pass had, it was
        // an answer about something else, so the number goes with it — never
        // left standing while the walk starts over.
        let key = |q: &str| CountKey {
            query: Some(q.into()),
            posts: 0,
            revising: Vec::new(),
        };
        let mut space = SpaceCount::default();
        space.restart(key("kestrel"));
        space.counts.insert("a1".into(), 3);
        space.next = 1;
        space.posts_done = true;
        space.total = Some(3);

        space.restart(key("hawk"));
        assert!(!space.is_settled(), "a new question is not answered yet");
        assert_eq!(space.total, None, "and it shows no number while it walks");
        assert!(space.counts.is_empty());
        assert_eq!(space.next, 0);
    }

    #[test]
    fn a_draft_still_waiting_on_its_scan_is_not_a_settled_count() {
        // The budget's honesty clause: a draft deferred to a later chunk is
        // work owed, so the count is *not* a number anything may show — the
        // same rule the posts half obeys, said about the half that has no
        // `CountKey` to be restarted by.
        let mut space = SpaceCount::default();
        space.restart(CountKey {
            query: Some("kestrel".into()),
            posts: 0,
            revising: Vec::new(),
        });
        space.posts_done = true;
        assert!(space.is_settled(), "the posts are the usual whole of it");

        space.drafts_pending = true;
        assert!(
            !space.is_settled(),
            "a draft owed a scan is a total that is not knowable yet"
        );
        space.drafts_pending = false;
        assert!(space.is_settled(), "and it comes back when the scan lands");
    }

    /// One match, named by node/item/ordinal — the identity the anchor is
    /// built from. The byte range is irrelevant to the anchoring rules.
    fn m(node: &str, item: Option<&str>, ordinal: usize) -> Match {
        Match {
            node: node.into(),
            item_id: item.map(SharedString::from),
            ordinal,
            source: 0..1,
            fraction: 0.0,
        }
    }

    #[test]
    fn a_new_query_leaves_no_old_results_for_a_step_to_anchor_in() {
        // A replay of the production sequence: `InputEvent::Change`
        // invalidates the session, a Return is handled before the notified
        // render, and only then does `sync_find` rebuild and re-anchor. The
        // step must find nothing to walk — otherwise it anchors in the
        // previous query's results and the rebuild honours that as the
        // reader's place.
        let mut matches: MatchSet = [m("a3", Some("i3"), 0)].into_iter().collect();
        let mut anchor = Some(MatchAnchor::of(&matches[0]));
        let mut pending: Option<PendingReveal> = None;

        invalidate_for_new_query(&mut matches, &mut anchor, &mut pending);
        assert!(matches.is_empty(), "the old results go with the old query");

        // The Return that beats the render.
        step_anchor(&matches, &mut anchor, true);
        assert_eq!(anchor, None, "nothing to step through, nothing anchored");

        // The render: `sync_find` reads the previous position, rebuilds the
        // set against the new query, and re-anchors.
        let previous = anchor
            .clone()
            .map(|a| (a, current_position(&matches, &anchor).unwrap_or(0)));
        matches = [
            m("a1", Some("i1"), 0),
            m("a2", Some("i2"), 0),
            m("a3", Some("i3"), 0),
        ]
        .into_iter()
        .collect();
        reanchor(&matches, &mut anchor, previous);

        assert_eq!(
            anchor,
            Some(MatchAnchor::of(&matches[0])),
            "the new query starts at its own first match, not forwarded onto \
             the post the old query's anchor named"
        );
    }

    #[test]
    fn a_nodes_matches_are_read_as_its_own_run() {
        // The per-node surfaces — every body editor's highlight layers, every
        // minimap cell — used to answer by scanning the whole match vector,
        // which is `O(posts × matches)` on every frame the bar is open. They
        // now read a slice, so what has to hold is that the slice is exactly
        // what the scan would have selected, and that the runs account for
        // every match once.
        let matches: MatchSet = [
            m("a1", Some("i1"), 0),
            m("a1", Some("i1"), 1),
            m("a2", Some("i2"), 0),
            m("draft-1", None, 0),
            m("draft-1", None, 1),
            m("draft-1", None, 2),
        ]
        .into_iter()
        .collect();

        let mut accounted = 0;
        for node in ["a1", "a2", "draft-1", "nobody"] {
            let node = SharedString::from(node);
            let scanned: Vec<&Match> = matches.iter().filter(|m| m.node == node).collect();
            let run: Vec<&Match> = matches.of(&node).iter().collect();
            assert_eq!(run, scanned, "the run for {node} is what a scan selects");
            accounted += run.len();
        }
        assert_eq!(
            accounted,
            matches.len(),
            "every match is in exactly one run"
        );

        // **And the read is a borrow of the stored vector, not a filtered
        // copy** — which is what bounds the per-node surfaces to that node's
        // own matches rather than to all of them. A scan cannot answer with a
        // `&[Match]` at all, so this address is the cost claim itself: the
        // run's first element *is* the fourth match, not an equal one.
        let run = matches.of(&"draft-1".into());
        assert!(
            std::ptr::eq(&run[0], &matches[3]),
            "a node's run is a slice of the set, in place"
        );

        // And which of a node's own matches is current is answered from that
        // run alone — an offset into it, never a position in the whole set.
        let anchor = Some(MatchAnchor::of(&matches[4]));
        assert_eq!(
            matches.current_of(&"draft-1".into(), &anchor),
            Some(1),
            "the second of the draft's own matches"
        );
        assert_eq!(
            matches.current_of(&"a1".into(), &anchor),
            None,
            "and no other node claims it"
        );
        assert_eq!(matches.current_of(&"a1".into(), &None), None);
    }

    #[test]
    fn stepping_wraps_at_both_ends() {
        let matches = vec![m("a", Some("i1"), 0), m("b", Some("i2"), 0)];
        let mut anchor = None;
        let node = |m: Option<Match>| m.map(|m| m.node.to_string());
        assert_eq!(
            node(step_anchor(&matches, &mut anchor, true)),
            Some("a".into())
        );
        assert_eq!(
            node(step_anchor(&matches, &mut anchor, true)),
            Some("b".into())
        );
        assert_eq!(
            node(step_anchor(&matches, &mut anchor, true)),
            Some("a".into())
        );
        assert_eq!(
            node(step_anchor(&matches, &mut anchor, false)),
            Some("b".into())
        );
    }

    #[test]
    fn an_edited_post_keeps_the_readers_place_through_its_new_action_id() {
        // The post keeps its item and gets a new action id — which is what an
        // edit or a regeneration does on every commit. Anchored by action id
        // the reader would be thrown back to match 1 of the conversation.
        let before = vec![m("act-1", Some("item-1"), 0), m("act-1", Some("item-1"), 1)];
        let mut anchor = None;
        reanchor(&before, &mut anchor, None);
        step_anchor(&before, &mut anchor, true);
        let previous = (anchor.clone().expect("anchored"), 1);

        let after = vec![m("act-2", Some("item-1"), 0), m("act-2", Some("item-1"), 1)];
        reanchor(&after, &mut anchor, Some(previous));
        assert_eq!(current_position(&after, &anchor), Some(1));
    }

    #[test]
    fn a_post_that_lost_matches_clamps_within_itself() {
        let before = [
            m("a", Some("i"), 0),
            m("a", Some("i"), 1),
            m("a", Some("i"), 2),
        ];
        let mut anchor = Some(MatchAnchor::of(&before[2]));
        let after = vec![m("a", Some("i"), 0)];
        let previous = anchor.clone().expect("anchored");
        reanchor(&after, &mut anchor, Some((previous, 2)));
        assert_eq!(current_position(&after, &anchor), Some(0));
    }

    #[test]
    fn a_post_that_left_falls_to_the_nearest_match_in_document_order() {
        let before = [
            m("a", Some("ia"), 0),
            m("b", Some("ib"), 0),
            m("c", Some("ic"), 0),
        ];
        let mut anchor = Some(MatchAnchor::of(&before[1]));
        // `b` is gone; the reader's place was position 1.
        let after = vec![m("a", Some("ia"), 0), m("c", Some("ic"), 0)];
        let previous = anchor.clone().expect("anchored");
        reanchor(&after, &mut anchor, Some((previous, 1)));
        assert_eq!(
            current_match(&after, &anchor).map(|m| m.node.to_string()),
            Some("c".into())
        );
    }

    #[test]
    fn no_matches_leaves_no_anchor() {
        let mut anchor = Some(MatchAnchor {
            key: "a".into(),
            ordinal: 0,
        });
        reanchor(&[], &mut anchor, None);
        assert!(anchor.is_none());
        assert!(step_anchor(&[], &mut anchor, true).is_none());
    }
}
