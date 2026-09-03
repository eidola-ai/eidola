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

use std::collections::HashMap;
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
pub(crate) fn searchable_projection(
    content: &str,
    embeds: &EmbedMap,
    cursor: Option<Selection>,
) -> Projection {
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
    let mut prev_end: Option<usize> = None;
    for block in &spec.blocks {
        let block_range = clamp(&block.source_range, content.len());
        if block_range.start >= block_range.end {
            continue;
        }
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
    builder.finish()
}

/// What the walk over one block's source finds at a given byte.
enum Mark<'a> {
    /// Bytes the render replaces with display text of its own.
    Substitute(&'a str),
    /// Bytes the render shows nothing for.
    Skip,
}

/// The barrier a table's grid chrome projects as — the block gap's newline,
/// reached by the other door.
///
/// **Not every hidden range is zero-width formatting.** A read-only table hides
/// the pipes and padding between cells, so deleting them wholesale would
/// concatenate the visible cell texts and let `| left | right |` match
/// `leftright` — a phrase occupying two cells the reader plainly sees apart.
/// The chrome is *structural*, the way a paragraph gap is, so it projects as
/// the same thing: one newline, which a query typed into a one-line field can
/// never contain.
///
/// It is a **substitution**, not a fabricated run: the newline stands for the
/// chrome's own source bytes, so the projected text still maps back to a range
/// of the post — the one rule [`ProjectionBuilder`] exists to keep. (Nothing
/// can ever match *into* it, so the atom-coverage semantics never come up.)
const TABLE_CELL_BARRIER: &str = "\n";

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
    for math in &block.math_overlays {
        let r = clamp(&math.source_range, content.len());
        if r.start < r.end && gpui_markdown_editor::math_overlay_typesets(block, content, math) {
            marks.push((r, Mark::Skip));
        }
    }
    for image in &block.image_overlays {
        let r = clamp(&image.source_range, content.len());
        if r.start < r.end {
            marks.push((r, Mark::Skip));
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
/// `cells` is in source order and non-overlapping (the render's own geometry),
/// and every edge is a character boundary of the source, so each piece is a
/// range [`ProjectionBuilder`] will accept.
fn split_at_cell_edges<'a>(
    range: Range<usize>,
    cells: &[Range<usize>],
    sub_starts: &[usize],
    out: &mut Vec<(Range<usize>, Mark<'a>)>,
) {
    let mut pos = range.start;
    while pos < range.end {
        match cells.iter().find(|c| c.start <= pos && pos < c.end) {
            // Inside a cell — inline markup, up to that cell's end.
            Some(cell) => {
                let end = cell.end.min(range.end);
                push_hide(pos..end, sub_starts, out);
                pos = end;
            }
            // Between cells — the grid, up to wherever the next cell begins.
            None => {
                let end = cells
                    .iter()
                    .map(|c| c.start)
                    .filter(|start| *start > pos)
                    .min()
                    .unwrap_or(range.end)
                    .min(range.end);
                out.push((pos..end, Mark::Substitute(TABLE_CELL_BARRIER)));
                pos = end;
            }
        }
    }
}

/// The content ranges of every cell a table block carries, or an empty vec for
/// any other block. The delimiter row has no cells to name — the render hides
/// its whole line, which is then outside every cell and so a barrier like the
/// rest of the grid.
fn table_cells(block: &gpui_markdown_editor::RenderBlock) -> Vec<Range<usize>> {
    let gpui_markdown_editor::BlockKind::Table { geometry, .. } = &block.kind else {
        return Vec::new();
    };
    geometry
        .rows
        .iter()
        .filter(|row| row.kind != gpui_markdown_editor::table::RowKind::Delimiter)
        .flat_map(|row| row.cells.iter().cloned())
        .collect()
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
    pub(crate) offset: usize,
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
    /// Every match on the visible branch, in document order.
    pub(crate) matches: Vec<Match>,
    /// Which match the readout counts as current.
    pub(crate) anchor: Option<MatchAnchor>,
    /// Per-node searchable projections, keyed by node id, each remembered with
    /// the content it was built from so a post whose text changed re-projects
    /// and one that did not is free.
    pub(crate) projections: HashMap<SharedString, (ProjectionSeed, Projection)>,
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

impl FindSession {
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
            offset: m.source.start,
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
    matches: &mut Vec<Match>,
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
        let open = self.find.as_ref().map(|session| {
            (
                session.input.clone(),
                session.focus.contains_focused(window, cx),
            )
        });
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
                    this.set_find_query(text, cx);
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
            matches: Vec::new(),
            anchor: None,
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
        // this window owes).
        let held = session.focus.contains_focused(window, cx);
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

    /// Whether the find surface currently owns the keyboard — what gates the
    /// Escape rung, so an Escape in the composer still deactivates the draft.
    pub(crate) fn find_holds_focus(&self, window: &Window, cx: &gpui::App) -> bool {
        self.find
            .as_ref()
            .is_some_and(|s| s.focus.contains_focused(window, cx))
    }

    /// Apply a committed query. Never called from an observer or a render.
    fn set_find_query(&mut self, text: String, cx: &mut Context<Self>) {
        let Some(session) = self.find.as_mut() else {
            return;
        };
        if session.text == text {
            return;
        }
        session.query = Query::new(&text);
        session.text = text;
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
    /// hang off that a wheel gesture would not miss. The cost is the scan, not
    /// the projections — those are cached per node against the content they
    /// were built from, so an unchanged post is a string comparison.
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
        let scope = self.find_scope(tree, page_width, cx);
        let mut session = self.find.take().expect("checked above");
        let previous = session.anchor.clone().map(|a| {
            (
                a,
                current_position(&session.matches, &session.anchor).unwrap_or(0),
            )
        });
        session.matches.clear();
        session
            .projections
            .retain(|node, _| scope.iter().any(|s| &s.node == node));

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
                    .is_some_and(|(built_from, _)| entry.frozen || built_from == &seed);
                if !cached {
                    if entry.frozen {
                        continue;
                    }
                    let projection =
                        searchable_projection(&entry.content, &entry.embeds, entry.render_cursor);
                    self.projections_built.set(self.projections_built.get() + 1);
                    session
                        .projections
                        .insert(entry.node.clone(), (seed, projection));
                }
                let Some((_, projection)) = session.projections.get(&entry.node) else {
                    continue;
                };
                let len = entry.content.len().max(1) as f32;
                for (ordinal, source) in projection.find(&query).into_iter().enumerate() {
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
        if owed {
            self.reveal_current_match(tree, page_width, window, cx);
        }
        self.correct_find_reveal(tree, page_width, window, cx);
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
    fn find_scope(&self, tree: &[TreeNode], page_width: Pixels, cx: &gpui::App) -> Vec<ScopeNode> {
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
                    let embeds = EmbedMap::new(post.references.iter().filter_map(|r| {
                        Some((
                            u64::try_from(r.ordinal).ok().filter(|o| *o > 0)?,
                            r.snippet.clone()?,
                        ))
                    }));
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
        if let Some(active) = self.active_draft.clone()
            && !seen.contains(&active)
            && let Some(entry) = self.draft_scope_node(&active, cx)
        {
            scope.push(entry);
        }
        scope
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
    pub(crate) fn find_match_ranges(
        &self,
        node: &SharedString,
    ) -> (Vec<Range<usize>>, Vec<Range<usize>>) {
        let Some(session) = self.find.as_ref() else {
            return (Vec::new(), Vec::new());
        };
        let current = session.current();
        let mut all = Vec::new();
        let mut active = Vec::new();
        for m in session.matches.iter().filter(|m| &m.node == node) {
            if Some(m) == current {
                active.push(m.source.clone());
            } else {
                all.push(m.source.clone());
            }
        }
        (all, active)
    }

    /// The minimap's tick positions for one node: `(fraction, is_current)`.
    pub(crate) fn find_ticks(&self, node: &SharedString) -> Vec<(f32, bool)> {
        let Some(session) = self.find.as_ref() else {
            return Vec::new();
        };
        let current = session.current();
        session
            .matches
            .iter()
            .filter(|m| &m.node == node)
            .map(|m| (m.fraction, Some(m) == current))
            .collect()
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
                offset: m.source.start,
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
        // Nothing to correct until the editor has painted.
        let answered = self
            .node_editor(&pending.node)
            .and_then(|e| e.read(cx).content_y_for_offset(pending.offset))
            .is_some();
        if !answered {
            return;
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
        match editor.and_then(|e| e.read(cx).content_y_for_offset(m.source.start)) {
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

        // The readout is a `Label` whose **label and value are the same short
        // sentence** — the notices' shape, so it starts speaking the day gpui
        // gains `aria_live` and is perceivable by review today.
        let readout: SharedString = match (has_query, index) {
            (false, _) => SharedString::default(),
            (true, Some(i)) => crate::i18n::msg::find_count(cx, i, total),
            (true, None) => crate::i18n::msg::find_no_results(cx),
        };
        let steppable = total > 0;

        let controls = h_flex()
            .absolute()
            .top(TITLE_BAR_RESERVE)
            .left_0()
            .right_0()
            .h(px(FIND_BAR_H))
            .px_3()
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
            );

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

    /// One of the two step arrows. **One predicate decides both tab-stopness
    /// and activation**, so a bar that is focused when the last match
    /// disappears cannot keep a live `on_click` for a step that does nothing.
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
        let el = div()
            .id(id)
            .probe(probe, gpui::Role::Button, aria)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn project(source: &str) -> Projection {
        searchable_projection(source, &EmbedMap::default(), None)
    }

    /// The projection of a node whose editor is *enabled* with its cursor at
    /// `at` — an inline edit or a draft, which is what the reader is looking
    /// at when they search one.
    fn project_editing(source: &str, at: usize) -> Projection {
        searchable_projection(source, &EmbedMap::default(), Some(Selection::Cursor(at)))
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
        // rule above must not reach into an overlay.
        let alt = "look ![a &amp; b](https://e/k.png) here";
        assert_eq!(project(alt).text(), "look  here");
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
            searchable_projection(source, &EmbedMap::new([(1, "quoted".to_string())]), None);
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
        let mut matches = vec![m("a3", Some("i3"), 0)];
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
        matches = vec![
            m("a1", Some("i1"), 0),
            m("a2", Some("i2"), 0),
            m("a3", Some("i3"), 0),
        ];
        reanchor(&matches, &mut anchor, previous);

        assert_eq!(
            anchor,
            Some(MatchAnchor::of(&matches[0])),
            "the new query starts at its own first match, not forwarded onto \
             the post the old query's anchor named"
        );
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
