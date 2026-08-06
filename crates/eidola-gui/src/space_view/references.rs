//! Quoted references — the space view's half of the ordinal seam.
//!
//! Three surfaces, one shared key. The **ordinal** (`action_antecedent.ordinal`
//! ↔ `{{ embed N }}` ↔ the editor's embed map ↔ the footnote index) is the only
//! thing the editor, app-core, and this module agree on; nothing else crosses.
//!
//! - **Quote creation.** A selection inside any post's read-only editor is
//!   tracked as a [`PostSelection`] (which post, which content block, which
//!   byte range, what it says). `Edit > Quote` attaches it to the active draft
//!   — minting the next ordinal, pushing a
//!   [`PendingReference`](super::PendingReference), handing the editor an
//!   embed map so the marker renders as a real quote block, and injecting the
//!   marker at the caret. `Edit > Quote in Reply` does the same into a fresh
//!   reply draft on the quoted post.
//! - **The footnote rail.** Below a post's (or draft's) body, outside the
//!   markdown: a quiet, ruled list of its references — ordinal, the quoted
//!   post's byline, the passage. A reference whose stored range no longer maps
//!   reads "quoted an earlier version" rather than guessing (ranges are never
//!   remapped). Clicking a row navigates to the quoted post. A draft's rows
//!   carry a remove affordance; a post's rows become removable chips inside
//!   the existing per-post Edit session.
//! - **Source highlights.** A post whose passages other posts have quoted
//!   paints them with the editor's opaque highlight plugin; a plain click
//!   navigates to the referencing post — or, when several quoted the same
//!   passage, opens a small picker. The data is
//!   [`Space::incoming_references`](crate::space::Space::incoming_references):
//!   per-space domain state on the shared entity, not a view field, so two
//!   windows on one space paint identically and `Change::Space` refreshes both.
//!
//! Ordinal 0 is app-core's reserved `reply` edge. It never appears in a
//! `PostNode.references` list, is never minted here, and is not removable
//! through any of these surfaces — the structural reply is the thread, not an
//! annotation on it.

use std::ops::Range;

use gpui::{
    AnyElement, Context, Focusable as _, InteractiveElement, IntoElement, ParentElement,
    SharedString, StatefulInteractiveElement, Styled, Window, div, prelude::FluentBuilder as _, px,
};
use gpui_component::{ActiveTheme, h_flex, v_flex};

use eidola_app_core::error::AppError;

use crate::actions::{Quote, QuoteInReply};
use crate::overlay::{Contain as _, Overlay};
use crate::probe::Probe as _;

use super::model::{NodeSrc, TreeNode};
use super::{GUTTER_GAP, PendingReference, SpaceView};

/// A live selection inside one post's read-only body — everything a
/// [`ReferenceSpec`](eidola_app_core::ReferenceSpec) needs, resolved at the
/// moment the user made it.
///
/// The body editor's buffer is the post's content blocks joined with no
/// separator, so a buffer byte range maps to `(block, block-relative range)`
/// by subtracting the block's span start. A selection that **crosses block
/// boundaries names no single block** and is therefore not quotable — the
/// schema's reference edge points at one block, and inventing a synthetic
/// one would make the stored range a lie.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PostSelection {
    /// The tree node id (= the post's action id) whose editor holds it.
    pub(crate) node_id: SharedString,
    /// The **concrete generation** quoted — references never remap to tips.
    pub(crate) action_id: String,
    /// The quoted content block.
    pub(crate) block_id: String,
    /// Byte range *within that block's text* (what the edge stores).
    pub(crate) range: Range<usize>,
    /// The quoted markdown itself.
    pub(crate) snippet: SharedString,
    /// The quoted post's byline, for the footnote row's attribution.
    pub(crate) byline: SharedString,
}

impl PostSelection {
    /// The write-side spec for this selection.
    pub(crate) fn spec(&self) -> eidola_app_core::ReferenceSpec {
        eidola_app_core::ReferenceSpec {
            antecedent_action_id: self.action_id.clone(),
            content_block_id: Some(self.block_id.clone()),
            range_start: Some(self.range.start as i64),
            range_end: Some(self.range.end as i64),
            annotation: None,
        }
    }
}

/// What a reader is told when a quote points into a conversation they take no
/// part in (task 37's denial, the human arm).
///
/// It may confirm that the passage came from somewhere — existence is public
/// **within this conversation**, which is the one the reader is in — and names
/// nothing else: no title, no participants, not a byte of what is there. The
/// wording is ours rather than the error's, so no future change to a typed
/// error's `Display` can widen what this surface says.
pub(crate) const FOLLOW_DENIED_HERE: SharedString = SharedString::new_static(
    "That passage was quoted from a conversation you don't take part in, so there is nowhere \
     to go from here.",
);

/// The open "quote into another conversation" picker (task 37's creation UI).
///
/// It holds the passage, because the reader's selection in the source post is
/// dropped the moment the quote lands somewhere and because the picker outlives
/// a click that lands elsewhere in the page. `confirming` is the second step:
/// a chosen destination, held so the **visibility statement can name it** —
/// which is the whole reason this is two steps and not a menu of one-click
/// verbs.
#[derive(Clone, Debug)]
pub(crate) struct QuoteDestination {
    pub(crate) selection: PostSelection,
    pub(crate) confirming: Option<(String, SharedString)>,
    /// The popover subtree's focus handle. The destination rows are tab stops
    /// and **arming replaces them** with the statement and its two verbs, so
    /// the pressed row unmounts: without somewhere inside the surface to put
    /// the keyboard, a reader who chose a destination from the keyboard is left
    /// on a dead handle (the rule `set_inspector_promote_confirm` states).
    pub(crate) focus: gpui::FocusHandle,
}

/// **The sentence the creation UI must show** (task 37): what quoting into
/// `title` means, in two facts a reader needs before they do it — who will be
/// able to read the passage, and that it is a *copy* (references name concrete
/// generations and are never remapped, so what leaves this conversation is
/// exactly the bytes chosen, once).
///
/// Pure, so the wording is asserted directly rather than through a painted
/// band, and so no destination can ever be shown without it.
pub(crate) fn visibility_statement(title: &str) -> SharedString {
    SharedString::from(format!(
        "This passage will be visible to everyone in {title}. Quoting copies it — later edits \
         here won't change it there."
    ))
}

/// How a conversation reads in the destination picker — the Library's own
/// rule (title, else the opening line, else "Untitled space"), so the two
/// surfaces name the same conversation the same way.
fn space_label(space: &eidola_app_core::SpaceInfo) -> SharedString {
    match (&space.title, &space.snippet) {
        (Some(t), _) => SharedString::from(t.clone()),
        (None, Some(s)) => SharedString::from(s.clone()),
        (None, None) => SharedString::from("Untitled space"),
    }
}

/// Map a body-editor byte range onto the single content block that fully
/// contains it, returning the block id and the block-relative range.
///
/// Pure (the spans come from the post snapshot), so it is unit-tested below.
/// Returns `None` for an empty range or one that spans two blocks — see
/// [`PostSelection`] for why a cross-block selection is not quotable.
pub(crate) fn block_range_for(
    blocks: &[crate::space::PostBlockSpan],
    range: &Range<usize>,
) -> Option<(String, Range<usize>)> {
    if range.start >= range.end {
        return None;
    }
    let block = blocks
        .iter()
        .find(|b| b.range.start <= range.start && range.end <= b.range.end)?;
    Some((
        block.block_id.clone(),
        (range.start - block.range.start)..(range.end - block.range.start),
    ))
}

/// One footnote row's presentation — the shape both the post rail and the
/// draft rail render, so the two can't drift.
pub(crate) struct FootnoteRow {
    /// Display index (`1.`, `2.`, …) — the rail's own numbering, which stays
    /// sequential even when the underlying ordinals have gaps.
    pub(crate) index: usize,
    /// The durable ordinal (a post's) — what a removal names.
    pub(crate) ordinal: i64,
    pub(crate) byline: SharedString,
    /// What this row says about the reference.
    pub(crate) body: FootnoteBody,
    /// The quoted post — the click target.
    pub(crate) antecedent_action_id: String,
}

/// What a footnote row can honestly say. The three cases are genuinely
/// different states, not one nullable string: a quote that still resolves, a
/// quote whose stored range no longer maps onto the generation it named (never
/// remapped, never approximated), and a plain backlink that never carried a
/// range at all.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum FootnoteBody {
    /// The passage, collapsed to one line.
    Quote(SharedString),
    /// A range was recorded but no longer maps onto that generation's block.
    Unresolvable,
    /// A reference to the whole post — no range was ever recorded.
    Backlink,
}

/// A zero-height, zero-visual **in-flow** probe that records its own flow
/// position into `cell`.
///
/// Two of these, bracketing the composer's footnote rail, measure the rail's
/// exact vertical occupancy — top margin, rule, padding and all — as the
/// difference of two real painted positions. That is the honest answer to "how
/// tall is the rail", and the reason the composer no longer carries a
/// row-count formula that drifts the moment the rail's styling changes. It
/// also degenerates correctly: with no rail rendered the two marks coincide
/// and the height is zero.
///
/// Deliberately **not** an absolute `size_full` child of the rail (the
/// `record_height` idiom): that resolves against the parent's *padding* box,
/// silently dropping the rail's margin and rule — a quiet under-count of the
/// same kind this replaces.
pub(crate) fn flow_mark(cell: std::rc::Rc<std::cell::Cell<f32>>) -> impl IntoElement {
    gpui::canvas(
        |_, _, _| {},
        move |bounds: gpui::Bounds<gpui::Pixels>, _, _, _| {
            cell.set(bounds.origin.y.as_f32());
        },
    )
    .w_full()
    .h_0()
}

/// Drop a post's **recognized** embed blocks from a text preview, leaving its
/// own prose. Used wherever a post's content is summarized as chrome (the
/// highlight picker's rows): the `{{ embed N }}` marker is rendered content,
/// never a string a user should read, so a preview that quoted it back would
/// leak the wire format into the UI.
///
/// Recognition is the editor's own `embed_blocks` over the post's embed map —
/// the same set that renders as quote blocks, so a marker the author defused
/// (fenced) or one with no mapping stays in the preview as the literal text it
/// literally is.
pub(crate) fn strip_embed_blocks(
    content: &str,
    references: &[eidola_app_core::PostReference],
) -> String {
    let map = gpui_markdown_editor::EmbedMap::new(
        references
            .iter()
            .filter_map(|r| Some((u64::try_from(r.ordinal).ok()?, r.snippet.clone()?))),
    );
    if map.is_empty() {
        return content.to_string();
    }
    let mut out = content.to_string();
    for block in gpui_markdown_editor::embed::embed_blocks(content, &map)
        .into_iter()
        .rev()
    {
        out.replace_range(block.range, "");
    }
    out
}

/// Drop the recognized `{{ embed N }}` blocks for `removed` ordinals from an
/// edited body, closing the paragraph gap each one leaves behind.
///
/// A footnote removed in an Edit session drops its *edge*; the marker that
/// addressed it must go with it, or the reloaded post renders the bare wire
/// syntax as literal text — and, worse, sends it upstream literally, since
/// `expand_embed_strings` has no edge to expand it against. Recognition is the
/// editor's own `embed_blocks` over a map of just the removed ordinals, so a
/// marker the author defused (fenced) or one belonging to a *surviving*
/// reference is left exactly where it is.
///
/// Pure over the submitted string — the live buffer is never touched, so Cancel
/// still restores the stashed original and a submit the space *rejects* (busy)
/// leaves the user's text intact.
pub(crate) fn strip_removed_markers(content: &str, removed: &[i64]) -> String {
    let map = gpui_markdown_editor::EmbedMap::new(
        removed
            .iter()
            .filter_map(|&o| Some((u64::try_from(o).ok()?, String::new()))),
    );
    if map.is_empty() {
        return content.to_string();
    }
    let mut out = content.to_string();
    for block in gpui_markdown_editor::embed::embed_blocks(content, &map)
        .into_iter()
        .rev()
    {
        let seam = block.range.start;
        out.replace_range(block.range, "");
        close_paragraph_gap(&mut out, seam);
    }
    out.trim().to_string()
}

/// Collapse the blank-line run spanning `seam` to a single paragraph break —
/// the marker's own blank-line delimiters would otherwise stack into a gap.
/// A run that reaches either end of the document is trimmed away entirely.
fn close_paragraph_gap(text: &mut String, seam: usize) {
    let is_gap = |b: u8| b == b'\n' || b == b' ' || b == b'\t' || b == b'\r';
    let bytes = text.as_bytes();
    let mut start = seam.min(bytes.len());
    while start > 0 && is_gap(bytes[start - 1]) {
        start -= 1;
    }
    let mut end = seam.min(bytes.len());
    while end < bytes.len() && is_gap(bytes[end]) {
        end += 1;
    }
    if start == end {
        return;
    }
    let replacement = if start == 0 || end == bytes.len() {
        ""
    } else {
        "\n\n"
    };
    text.replace_range(start..end, replacement);
}

/// Collapse a quoted passage to one quiet line: whitespace runs folded, then
/// truncated on a word boundary with an ellipsis. The rail is an index, not a
/// second copy of the quote — the body's embed block is where you read it.
pub(crate) fn footnote_snippet(text: &str) -> String {
    snippet_to(text, 96)
}

/// [`footnote_snippet`] with an explicit budget — the minimap's a11y labels
/// take a shorter one.
pub(crate) fn snippet_to(text: &str, max: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max {
        return flat;
    }
    let cut = flat
        .char_indices()
        .nth(max)
        .map(|(i, _)| i)
        .unwrap_or(flat.len());
    let head = &flat[..cut];
    let head = head.rsplit_once(' ').map(|(a, _)| a).unwrap_or(head);
    format!("{head}…")
}

impl SpaceView {
    // -- Per-frame sync -----------------------------------------------------

    /// Keep each post body editor's reference decorations current: its **embed
    /// map** (so `{{ embed N }}` markers materialize as the quoted passages)
    /// and its **highlight set** (the passages other posts quoted). Also
    /// requests the incoming-reference index for the posts that actually
    /// rendered last frame.
    ///
    /// Both setters `notify()`, so both are guarded by an equality check
    /// against what the editor already holds — writing unconditionally every
    /// frame would be an infinite render loop.
    pub(crate) fn sync_references(&mut self, cx: &mut Context<Self>) {
        // Drain last frame's visible posts into lazy per-post fetches (see
        // `wants_incoming_refs`). `ensure_incoming_references` is idempotent.
        let wanted: Vec<SharedString> = self.wants_incoming_refs.borrow_mut().drain().collect();
        if !wanted.is_empty() {
            self.space.update(cx, |space, cx| {
                for id in &wanted {
                    space.ensure_incoming_references(id, cx);
                }
            });
        }

        for i in 0..self.posts.len() {
            let id = super::model::node_id(&self.posts, i);
            let Some(editor) = self.bodies.get(&id).cloned() else {
                continue;
            };
            let entries: Vec<(u64, String)> = self.posts[i]
                .references
                .iter()
                .filter(|r| r.ordinal > 0)
                .filter_map(|r| Some((u64::try_from(r.ordinal).ok()?, r.snippet.clone()?)))
                .collect();
            let ranges = self.highlight_ranges(i, cx);
            let embeds = gpui_markdown_editor::EmbedMap::new(entries.clone());
            let highlights = gpui_markdown_editor::HighlightSet::new(ranges.clone());
            let (embeds_stale, highlights_stale) = {
                let e = editor.read(cx);
                (*e.embeds() != embeds, *e.highlights() != highlights)
            };
            if embeds_stale || highlights_stale {
                editor.update(cx, |e, cx| {
                    if embeds_stale {
                        e.set_embeds(entries, cx);
                    }
                    if highlights_stale {
                        e.set_highlights(ranges, cx);
                    }
                });
            }
        }
    }

    /// Note that post `node_id` rendered for real this frame, so the next
    /// frame requests its incoming-reference index (see
    /// [`SpaceView::wants_incoming_refs`]). Callable from the shared `&self`
    /// render path.
    pub(crate) fn want_incoming_refs(&self, node_id: &SharedString) {
        self.wants_incoming_refs
            .borrow_mut()
            .insert(node_id.clone());
    }

    // -- Selection tracking -------------------------------------------------

    /// Record (or clear) the quotable selection in post `node_id`'s read-only
    /// body. Called from each body editor's subscription on
    /// `SelectionChanged`.
    ///
    /// Notifies **only when quotability flips**, not on every drag step: the
    /// stored value is refreshed either way (the Quote handler reads the
    /// latest), but the only thing a re-render buys is the Edit menu's
    /// enablement, which depends on `is_some()` alone.
    pub(crate) fn note_body_selection(&mut self, node_id: &SharedString, cx: &mut Context<Self>) {
        let was = self.post_selection.is_some();
        self.post_selection = self.resolve_body_selection(node_id, cx);
        if was != self.post_selection.is_some() {
            cx.notify();
        }
    }

    /// Resolve post `node_id`'s current editor selection into a
    /// [`PostSelection`], or `None` when it is collapsed, crosses blocks, or
    /// the post isn't persisted (an optimistic row has nothing to quote yet).
    fn resolve_body_selection(
        &self,
        node_id: &SharedString,
        cx: &gpui::App,
    ) -> Option<PostSelection> {
        let editor = self.bodies.get(node_id)?;
        let editor = editor.read(cx);
        let range = editor.selection().selection_range();
        let post = self
            .posts
            .iter()
            .find(|p| p.action_id.as_deref() == Some(node_id.as_ref()))?;
        let action_id = post.action_id.clone()?;
        let (block_id, block_range) = block_range_for(&post.blocks, &range)?;
        let snippet = editor.value().get(range.clone())?.to_string();
        Some(PostSelection {
            node_id: node_id.clone(),
            action_id: action_id.to_string(),
            block_id,
            range: block_range,
            snippet: snippet.into(),
            byline: post.byline.clone(),
        })
    }

    /// The current quotable selection, if any — what gates the Edit menu's
    /// Quote items and what they act on. Public for behavior tests.
    pub fn post_selection_action_id(&self) -> Option<String> {
        self.post_selection.as_ref().map(|s| s.action_id.clone())
    }

    // -- Quote --------------------------------------------------------------

    /// `Edit > Quote` — attach the current post selection to the **active**
    /// draft (activating the selected branch's tail draft, or opening one at
    /// its leaf, when no composer is open) and inject the embed marker at the
    /// caret.
    pub fn quote(&mut self, _: &Quote, window: &mut Window, cx: &mut Context<Self>) {
        let Some(selection) = self.post_selection.clone() else {
            return;
        };
        let Some(draft_id) = self.draft_for_quote(window, cx) else {
            return;
        };
        self.attach_quote(&draft_id, selection, window, cx);
    }

    /// `Edit > Quote in Reply` — the same attachment, but into a draft
    /// replying to the **quoted post**, so the answer branches where the
    /// passage is. Reuses an existing empty draft on that post (the tail
    /// draft's job) rather than stacking a second one beside it; otherwise
    /// forks a new branch there.
    pub fn quote_in_reply(
        &mut self,
        _: &QuoteInReply,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(selection) = self.post_selection.clone() else {
            return;
        };
        let parent = SharedString::from(selection.action_id.clone());
        let existing = self
            .drafts
            .iter()
            .find(|d| d.parent.as_ref() == Some(&parent) && d.editor.read(cx).is_empty())
            .map(|d| d.id.clone());
        let draft_id = match existing {
            Some(id) => {
                self.activate_draft(id.clone(), cx);
                let focus = self
                    .drafts
                    .iter()
                    .find(|d| d.id == id)
                    .map(|d| d.editor.read(cx).focus_handle(cx));
                if let Some(focus) = focus {
                    window.focus(&focus, cx);
                }
                id
            }
            None => {
                self.create_draft(Some(parent), window, cx);
                let Some(id) = self.active_draft.clone() else {
                    return;
                };
                id
            }
        };
        self.attach_quote(&draft_id, selection, window, cx);
    }

    /// Which draft a plain `Quote` lands in: the active composer if one is
    /// open; otherwise the draft sitting on the currently selected branch (the
    /// tail composer `sync_tail_drafts` keeps there), activated; otherwise a
    /// fresh draft replying to that branch's last post.
    fn draft_for_quote(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<SharedString> {
        if let Some(id) = self.active_draft.clone() {
            return Some(id);
        }
        let viewport = self.page_size(window);
        let turns = self.stream_overlays(cx);
        let tree = self.effective_tree(viewport.width, &turns);
        let path = self.selected_path_nodes(&tree, viewport.width);

        // Deepest draft on the selected path — the branch's tail composer.
        if let Some(id) = path
            .iter()
            .rev()
            .find(|n| matches!(n.src, NodeSrc::Draft))
            .map(|n| n.id.clone())
        {
            self.activate_draft(id.clone(), cx);
            let focus = self
                .drafts
                .iter()
                .find(|d| d.id == id)
                .map(|d| d.editor.read(cx).focus_handle(cx));
            if let Some(focus) = focus {
                window.focus(&focus, cx);
            }
            return Some(id);
        }

        // No draft on this branch (streaming, or a transcript still loading):
        // open one at its last real post.
        let leaf = path
            .iter()
            .rev()
            .find(|n| matches!(n.src, NodeSrc::Msg(_)))
            .map(|n| n.id.clone());
        self.create_draft(leaf, window, cx);
        self.active_draft.clone()
    }

    /// The selected root→leaf path as owned nodes (the levels walk borrows the
    /// tree, which the caller is about to mutate through).
    fn selected_path_nodes(&self, roots: &[TreeNode], page_width: gpui::Pixels) -> Vec<TreeNode> {
        self.selected_levels(roots, page_width)
            .into_iter()
            .map(|(sibs, active)| sibs[active].clone())
            .collect()
    }

    /// Push `selection` onto `draft_id` as its next reference and inject the
    /// marker.
    fn attach_quote(
        &mut self,
        draft_id: &SharedString,
        selection: PostSelection,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.attach_reference(
            draft_id,
            selection.spec(),
            selection.byline.clone(),
            selection.snippet.clone(),
            window,
            cx,
        );
    }

    /// Push a reference onto `draft_id` and inject the marker: the editor
    /// learns the embed map first (so the marker materializes as a quote block
    /// the instant it lands), then the marker is inserted at the caret through
    /// the editor's normal update pipeline (one undo step; a marker dropped
    /// into a verbatim region degrades to literal text, which is the documented
    /// honest behavior).
    ///
    /// Takes the reference's three parts rather than a [`PostSelection`],
    /// because a **cross-space** quote arrives from another window with no
    /// selection of ours behind it — the spec names a concrete generation, and
    /// that is all that travels.
    fn attach_reference(
        &mut self,
        draft_id: &SharedString,
        spec: eidola_app_core::ReferenceSpec,
        byline: SharedString,
        snippet: SharedString,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(draft) = self.drafts.iter_mut().find(|d| &d.id == draft_id) else {
            return;
        };
        let ordinal = draft.next_ordinal();
        draft.references.push(PendingReference {
            ordinal,
            spec,
            byline,
            snippet,
        });
        let embeds = draft.embed_map();
        let editor = draft.editor.clone();
        editor.update(cx, |e, cx| {
            e.set_embeds(embeds, cx);
            e.insert_embed_marker(ordinal, cx);
        });
        // The quote has left the source post; drop the selection so a second
        // Quote can't silently re-attach the same passage.
        self.post_selection = None;
        let focus = editor.read(cx).focus_handle(cx);
        window.focus(&focus, cx);
        cx.notify();
    }

    // -- Quote into another conversation ------------------------------------

    /// `Edit > Quote in Another Conversation…` — the cross-space arm of the
    /// quote affordances (task 37's creation UI).
    ///
    /// **The mechanism is a destination picker, not a drag**: a quote is a
    /// write into a conversation, so it is chosen from a list of conversations
    /// (the Library's own index), and the reader confirms a sentence naming
    /// the one they picked. Cross-window drag would be a second, gestural way
    /// to say the same thing with nowhere to put the sentence.
    pub fn quote_elsewhere(
        &mut self,
        _: &crate::actions::QuoteElsewhere,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(selection) = self.post_selection.clone() else {
            return;
        };
        self.quote_destination = Some(QuoteDestination {
            selection,
            confirming: None,
            focus: cx.focus_handle(),
        });
        // Ask for a fresh index as the picker opens — what `OpenLibrary` does
        // on every invocation, and the only thing that re-reads a `Failed` one
        // besides a bus signal. A fresh scroll for a freshly opened list.
        self.quote_destination_scroll = gpui::ScrollHandle::new();
        self.stores.spaces.update(cx, |s, cx| s.refresh(cx));
        cx.notify();
    }

    /// Close the destination picker (Escape, click-out, Cancel). Returns
    /// whether it was open — the Escape rung's answer.
    pub fn close_quote_destination(&mut self, cx: &mut Context<Self>) -> bool {
        if self.quote_destination.take().is_some() {
            cx.notify();
            return true;
        }
        false
    }

    /// Arm the confirmation for one destination: the picker keeps the passage
    /// and grows the sentence that says who will be able to read it.
    fn arm_quote_destination(
        &mut self,
        space_id: String,
        title: SharedString,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(dest) = self.quote_destination.as_mut() else {
            return;
        };
        dest.confirming = Some((space_id, title));
        // The chosen row unmounts with the list it was in; the surface itself
        // survives, so the keyboard stays on it — and only if it was the
        // surface holding it, so a pointer press elsewhere takes nothing.
        let focus = dest.focus.clone();
        if focus.contains_focused(window, cx) {
            window.focus(&focus, cx);
        }
        cx.notify();
    }

    /// Send the quote to the confirmed destination: hand it to that space's
    /// entity and present that conversation's window.
    ///
    /// The two windows share nothing but the [`Space`](crate::space::Space)
    /// entity (a draft is window-local by design), so the entity is the
    /// courier — see [`Space::offer_quote`](crate::space::Space::offer_quote).
    /// Nothing durable happens here: the passage lands in a **draft**, which
    /// the reader still has to post, and app-core validates the reference at
    /// that write (a quote into a conversation you have left is refused there,
    /// with zero trace, rather than being second-guessed here).
    ///
    /// **The window presented is the window holding the quote.** The
    /// destination may already be open — the `Space` registry joins windows on
    /// one entity — so this raises that window rather than opening a second one
    /// onto the same conversation, and *addresses* the offer to it so the
    /// one-shot mailbox cannot be drained by a window the reader never sees
    /// (Codex review, PR #280: opening unconditionally left the reader looking
    /// at a fresh, empty composer while an existing window had taken the
    /// passage). This is a targeted navigation, not a `⌘N`/Library open, which
    /// is what the window model's "no window dedup" residual covers; the
    /// raise-or-open shape is [`crate::open_record_request`]'s.
    fn send_quote_to_destination(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(dest) = self.quote_destination.take() else {
            return;
        };
        let Some((space_id, _)) = dest.confirming else {
            return;
        };
        let quote = crate::space::OfferedQuote {
            spec: dest.selection.spec(),
            byline: dest.selection.byline.clone(),
            snippet: dest.selection.snippet.clone(),
        };
        // The same space may already be open in another window; `open` is
        // get-or-create, so the offer reaches the one shared entity either way.
        let space = self
            .stores
            .spaces
            .update(cx, |spaces, cx| spaces.open(space_id.clone(), cx));
        // The newest window on this conversation, if any — the one a reader who
        // has several open was last working in.
        let target = space.update(cx, |space, cx| space.open_windows(cx).last().copied());
        space.update(cx, |space, cx| {
            space.offer_quote(quote, target.map(|w| w.window_id()), cx)
        });
        // The quote has left this post; drop the selection so the same passage
        // can't be sent twice by a second press (the in-space quote's rule).
        self.post_selection = None;
        let stores = self.stores.clone();
        let intent = crate::lifecycle::intend_to_open(cx);
        cx.defer(move |cx: &mut gpui::App| {
            if !intent.still_wanted(cx) {
                return;
            }
            if let Some(handle) = target {
                if crate::raise_space_window(cx, handle) {
                    return;
                }
                // The window went away between the offer and this defer. Its
                // address now names nobody, so hand the offer back to whoever
                // draws this space next — the window opened just below.
                space.update(cx, |space, cx| space.readdress_offer_to_any_window(cx));
            }
            crate::open_space_window(cx, stores, space_id);
        });
        let _ = window;
        cx.notify();
    }

    /// Take a quote another window handed this space and attach it to a draft
    /// — the receiving half of the handoff, drained once per offer at the head
    /// of `render`.
    ///
    /// It lands exactly where `Edit > Quote` would put it (the branch's tail
    /// composer, activated), because from here on it *is* an ordinary pending
    /// reference: same ordinal minting, same footnote row, same
    /// accept-before-consume.
    ///
    /// The offer is addressed, so a window that shares this space with the one
    /// the sender presented draws straight past it (see
    /// [`Space::take_offered_quote`](crate::space::Space::take_offered_quote)).
    pub(crate) fn adopt_offered_quote(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let here = window.window_handle().window_id();
        if !self.space.read(cx).has_offered_quote_for(here) {
            return;
        }
        // **Wait for the tail this belongs to.** Quoting into a conversation
        // that was not already open lands in a window whose first frames run
        // before its transcript has answered, and `sync_tail_drafts` mints no
        // composer in that state — so taking the offer there made
        // `draft_for_quote` pick a parent from a tree with no posts in it: a
        // *root* draft, attached to nothing on screen, submitting with no
        // `reply_to` and persisted under whatever the tail turned out to be
        // (Codex review, PR #280). The mailbox already survives frames, so the
        // offer simply stays in it — no second pending mechanism — and is taken
        // on the first frame there is somewhere for it to land.
        //
        // The same predicate `sync_tail_drafts` gates on, deliberately without
        // its *streaming* half: a streaming space has a loaded tree, so
        // `draft_for_quote`'s fallback picks the branch's last real post — a
        // parent that exists — where waiting would leave the reader looking at
        // the window they were just shown with an empty composer.
        //
        // Rethreading cannot cure this after the fact (the doctrine's usual
        // move): `rethread_drafts` forwards a draft through the *identity* its
        // parent named, and a root draft names nothing — "root because this
        // space is empty" and "root because we had not read it yet" are the
        // same value. A draft minted against an unloaded tree is a guess; the
        // doctrine's drafts attach to what exists.
        if !self.space.read(cx).transcript_loaded() {
            return;
        }
        let Some(offer) = self
            .space
            .update(cx, |space, _| space.take_offered_quote(here))
        else {
            return;
        };
        let Some(draft_id) = self.draft_for_quote(window, cx) else {
            return;
        };
        self.attach_reference(
            &draft_id,
            offer.spec,
            offer.byline,
            offer.snippet,
            window,
            cx,
        );
    }

    /// The destination picker: which conversation to quote into, and — once one
    /// is chosen — **the sentence that says what choosing it means**.
    ///
    /// The statement is the point of the surface (task 37): references name
    /// concrete generations and are never remapped, so the source space leaks
    /// exactly the bytes deliberately quoted, once — and the person doing the
    /// quoting is the flow-control point. So the destination is named in the
    /// sentence, and the verb below it is the only way through.
    pub(crate) fn render_quote_destination(&self, cx: &Context<Self>) -> Option<AnyElement> {
        let dest = self.quote_destination.as_ref()?;
        let theme = cx.theme();
        let here = self.space.read(cx).id().map(str::to_string);

        let mut col = v_flex()
            .id("space-quote-destination")
            .track_focus(&dest.focus)
            .probe(
                "space/quote-destination",
                gpui::Role::Group,
                "Quote into another conversation",
            )
            // An opaque popover over the page (see `crate::overlay`).
            .contain_mouse(Overlay::Popover)
            .absolute()
            .right(GUTTER_GAP)
            .bottom(px(96.))
            .w(px(320.))
            .p_1()
            .gap_0p5()
            .rounded_md()
            .border_1()
            .border_color(theme.border)
            .bg(theme.popover)
            .on_mouse_down_out(cx.listener(|this, _, _, cx| {
                this.close_quote_destination(cx);
            }));

        if let Some((_, title)) = dest.confirming.clone() {
            // The mandatory statement, naming the destination.
            let statement = visibility_statement(&title);
            col = col
                .child(
                    div()
                        .id("space-quote-destination-note")
                        .px_1()
                        .py_0p5()
                        .text_xs()
                        .text_color(theme.muted_foreground)
                        .probe_value(
                            "space/quote-destination/note",
                            gpui::Role::Label,
                            "What quoting there means",
                            statement.clone(),
                        )
                        .child(statement),
                )
                .child(
                    h_flex()
                        .gap_1()
                        .px_1()
                        .py_0p5()
                        .child(
                            div()
                                .id("space-quote-destination-confirm")
                                .probe(
                                    "space/quote-destination/confirm",
                                    gpui::Role::Button,
                                    SharedString::from(format!("Quote into {title}")),
                                )
                                .px_2()
                                .py_0p5()
                                .rounded_md()
                                .text_xs()
                                .cursor_pointer()
                                .text_color(theme.foreground)
                                .hover(|s| s.bg(theme.muted))
                                .child("Quote there")
                                .on_click(cx.listener(|this, _, window, cx| {
                                    this.send_quote_to_destination(window, cx);
                                })),
                        )
                        .child(
                            div()
                                .id("space-quote-destination-cancel")
                                .probe(
                                    "space/quote-destination/cancel",
                                    gpui::Role::Button,
                                    "Cancel",
                                )
                                .px_2()
                                .py_0p5()
                                .rounded_md()
                                .text_xs()
                                .cursor_pointer()
                                .text_color(theme.muted_foreground)
                                .hover(|s| s.bg(theme.muted))
                                .child("Cancel")
                                .on_click(cx.listener(|this, _, _, cx| {
                                    this.close_quote_destination(cx);
                                })),
                        ),
                );
            return Some(col.into_any_element());
        }

        col = col.child(
            div()
                .px_1()
                .pb_0p5()
                .text_xs()
                .text_color(theme.muted_foreground)
                .child("Quote into"),
        );

        // **The index has states, and this surface has to read all of them**
        // (Codex review, PR #280). `list()` answers `&[]` for a failed read
        // exactly as it does for a real empty Library — the Library window's
        // own "Failed is not empty" rule — so collapsing them said "No other
        // conversations yet" about a read that failed, with nothing to press:
        // quoting elsewhere was simply dead for the session, since the index is
        // re-read only on a bus signal or an `OpenLibrary`.
        let (load_error, has_listing, destinations) = {
            let store = self.stores.spaces.read(cx);
            let cell = store.index();
            let rows: Vec<(String, SharedString)> = store
                .list()
                .iter()
                .filter(|s| here.as_deref() != Some(s.id.as_str()))
                .map(|s| (s.id.clone(), space_label(s)))
                .collect();
            (cell.error().map(|e| e.to_string()), cell.has_value(), rows)
        };

        if destinations.is_empty() {
            let (line, retry) = match (&load_error, has_listing) {
                // A failed *initial* read: say so, and offer the door back. The
                // quiet retry line rather than the full `load_error_panel` — a
                // 320px popover is the Library's "couldn't refresh" idiom, not
                // its centred panel.
                (Some(_), false) => ("Couldn't load your conversations.", true),
                // Nothing has answered yet. A read in flight knows nothing, and
                // an unanswered index is not an empty one.
                (None, false) => ("Loading…", false),
                // Genuinely empty (or empty-but-stale, which the retry says).
                _ => ("No other conversations yet.", load_error.is_some()),
            };
            col = col.child(
                div()
                    .id("space-quote-destination-empty")
                    // A static readout rides its own `Label` node (the a11y
                    // rule): three states, three sentences, and which one is
                    // showing is the whole point of Finding C.
                    .probe_value(
                        "space/quote-destination/empty",
                        gpui::Role::Label,
                        "Conversations",
                        line,
                    )
                    .px_1()
                    .py_0p5()
                    .text_xs()
                    .text_color(theme.muted_foreground)
                    .child(line),
            );
            if retry {
                col = col.child(self.quote_destination_retry(cx));
            }
            return Some(col.into_any_element());
        }

        // **A bounded, scrolling list** — the Library is unbounded, and an
        // unbounded column inside a popover clipped its own overflow: every
        // conversation past the first handful was unreachable, and each frame
        // built a row for all of them. The model picker's shape, because a
        // dropdown of conversations is a dropdown (Codex review, PR #280).
        let mut menu = v_flex()
            .id("space-quote-destination-list")
            .w_full()
            .max_h(px(220.))
            .overflow_y_scroll()
            .track_scroll(&self.quote_destination_scroll)
            .gap_0p5();
        for (offered, (id, label)) in destinations.into_iter().enumerate() {
            let for_arm = label.clone();
            menu = menu.child(
                div()
                    .id(SharedString::from(format!(
                        "space-quote-destination-{offered}"
                    )))
                    .probe(
                        format!("space/quote-destination/{offered}"),
                        gpui::Role::Button,
                        label.clone(),
                    )
                    .w_full()
                    .px_1()
                    .py_0p5()
                    .rounded_sm()
                    .text_xs()
                    .truncate()
                    .cursor_pointer()
                    .hover(|s| s.bg(theme.muted))
                    .child(label)
                    .on_click(cx.listener(move |this, _, window, cx| {
                        this.arm_quote_destination(id.clone(), for_arm.clone(), window, cx);
                    })),
            );
        }
        col = col.child(div().relative().w_full().child(menu).child(
            crate::scrollbar::vertical_floating(
                "quote-destination-scrollbar",
                &self.quote_destination_scroll,
            ),
        ));
        // A failed *refresh* over a listing we still hold keeps the rows — they
        // are honest as of the last read — and says the last read is no longer
        // the last word.
        if load_error.is_some() {
            col = col.child(self.quote_destination_retry(cx));
        }
        Some(col.into_any_element())
    }

    /// The picker's quiet retry line — the Library's "couldn't refresh" strip,
    /// in a popover. It is the **only** door to a fresh index from here: the
    /// store re-reads on a bus `Change::SpaceIndex` or an `OpenLibrary`, so
    /// without it a reader whose index failed has to know that opening the
    /// Library is what fixes quoting elsewhere.
    fn quote_destination_retry(&self, cx: &Context<Self>) -> AnyElement {
        let theme = cx.theme();
        div()
            .id("space-quote-destination-retry")
            .probe("space/quote-destination/retry", gpui::Role::Button, "Retry")
            .px_1()
            .py_0p5()
            .text_xs()
            .cursor_pointer()
            .text_color(theme.muted_foreground)
            .hover(|s| s.text_color(theme.foreground))
            .child("Couldn't refresh — retry")
            .on_click(cx.listener(|this, _, _, cx| {
                this.stores.spaces.update(cx, |s, cx| s.refresh(cx));
            }))
            .into_any_element()
    }

    /// Drop a pending reference from a draft: remove the row **and** its embed
    /// marker from the body, so the quote block disappears with its footnote
    /// (a stranded marker would render as literal `{{ embed N }}` text).
    /// Surviving ordinals are **not** renumbered — their markers address them.
    pub fn remove_draft_reference(
        &mut self,
        draft_id: &SharedString,
        ordinal: u64,
        cx: &mut Context<Self>,
    ) {
        let Some(draft) = self.drafts.iter_mut().find(|d| &d.id == draft_id) else {
            return;
        };
        draft.references.retain(|r| r.ordinal != ordinal);
        let embeds = draft.embed_map();
        let editor = draft.editor.clone();
        editor.update(cx, |e, cx| {
            e.remove_embed_marker(ordinal, cx);
            e.set_embeds(embeds, cx);
        });
        cx.notify();
    }

    /// Toggle a persisted reference's removal mark inside the active edit
    /// session (the rail's chips). Ordinal 0 — the reply edge — is refused
    /// here as well as core-side: the reply is the thread, not an annotation.
    pub fn toggle_reference_removal(&mut self, ordinal: i64, cx: &mut Context<Self>) {
        if ordinal <= 0 {
            return;
        }
        let Some(editing) = self.editing.as_mut() else {
            return;
        };
        if let Some(pos) = editing
            .removed_references
            .iter()
            .position(|o| *o == ordinal)
        {
            editing.removed_references.remove(pos);
        } else {
            editing.removed_references.push(ordinal);
        }
        cx.notify();
    }

    // -- The footnote rail --------------------------------------------------

    /// The rail under a **persisted post**: its `reference` edges as ruled
    /// footnote rows. Rendered outside the markdown body (the editor never
    /// learns what a reference is), quiet by default; inside an Edit session
    /// each row grows a removal chip.
    pub(crate) fn render_post_footnotes(
        &self,
        i: usize,
        node: &TreeNode,
        editing: bool,
        cx: &Context<Self>,
    ) -> Option<AnyElement> {
        let post = &self.posts[i];
        let rows: Vec<FootnoteRow> = post
            .references
            .iter()
            // Defense in depth: ordinal 0 is the structural reply edge and
            // never arrives in a references list.
            .filter(|r| r.ordinal > 0)
            .enumerate()
            .map(|(idx, r)| FootnoteRow {
                index: idx + 1,
                ordinal: r.ordinal,
                byline: self.reference_byline(&r.antecedent_action_id),
                body: match (r.snippet.as_deref(), r.range_start) {
                    (Some(s), _) => FootnoteBody::Quote(footnote_snippet(s).into()),
                    (None, Some(_)) => FootnoteBody::Unresolvable,
                    (None, None) => FootnoteBody::Backlink,
                },
                antecedent_action_id: r.antecedent_action_id.clone(),
            })
            .collect();
        if rows.is_empty() {
            return None;
        }
        let removed = self
            .editing
            .as_ref()
            .filter(|e| e.node_id == node.id)
            .map(|e| e.removed_references.clone())
            .unwrap_or_default();

        let mut rail = self.rail_frame(cx);
        for row in &rows {
            let marked = removed.contains(&row.ordinal);
            let mut el = self.footnote_row(
                format!("space-fn-{}-{}", node.id, row.ordinal),
                format!("space/post/{i}/footnote/{}", row.index),
                row,
                marked,
                !editing,
                cx,
            );
            if editing {
                let ordinal = row.ordinal;
                el = el.child(
                    div()
                        .id(SharedString::from(format!(
                            "space-fn-rm-{}-{}",
                            node.id, row.ordinal
                        )))
                        .probe(
                            format!("space/post/{i}/footnote/{}/remove", row.index),
                            gpui::Role::Button,
                            if marked {
                                "Keep this reference"
                            } else {
                                "Remove this reference"
                            },
                        )
                        .flex_none()
                        .px_1()
                        .text_xs()
                        .text_color(if marked {
                            cx.theme().foreground
                        } else {
                            cx.theme().muted_foreground
                        })
                        .cursor_pointer()
                        .child(if marked { "undo" } else { "×" })
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.toggle_reference_removal(ordinal, cx);
                        })),
                );
            } else {
                let target = row.antecedent_action_id.clone();
                el = el.on_click(cx.listener(move |this, _, window, cx| {
                    this.navigate_to_action(target.clone(), window, cx);
                }));
            }
            rail = rail.child(el);
        }
        Some(rail.into_any_element())
    }

    /// The rail under a **draft**: its pending references, each with an
    /// **embed** affordance (re-place the quote's marker in the body when it
    /// isn't there) and a remove affordance (dropping the row also drops its
    /// marker).
    ///
    /// `measure` records the rail's painted height into
    /// [`SpaceView::composer_rail_h`] — set only for the *active* draft, whose
    /// bar has to reserve room for it; an inactive draft's rail rides its
    /// post-shaped frame and needs no reservation.
    pub(crate) fn render_draft_footnotes(
        &self,
        draft_id: &SharedString,
        measure: bool,
        cx: &Context<Self>,
    ) -> Option<AnyElement> {
        let draft = self.drafts.iter().find(|d| &d.id == draft_id)?;
        if draft.references.is_empty() {
            return None;
        }
        // Which ordinals the body already carries as *recognized* embed
        // blocks — the ones whose "embed" affordance would duplicate a marker
        // that is already there. Same recognition the composer's compaction
        // and the editor's rendering use, so the affordance appears exactly
        // when the quote block is missing from the draft.
        let body = draft.editor.read(cx).value();
        let map = gpui_markdown_editor::EmbedMap::new(draft.embed_map());
        let placed: std::collections::HashSet<u64> =
            gpui_markdown_editor::embed::embed_blocks(body, &map)
                .into_iter()
                .map(|b| b.ordinal)
                .collect();
        let mut refs = draft.references.clone();
        refs.sort_by_key(|r| r.ordinal);

        let mut rail = self.rail_frame(cx);
        if measure {
            // The active composer's rail is the last thing in the bar, so it
            // carries the bar's bottom breath — [`composer::rail_breath`], a
            // post's full bottom pad (the half-pad that mirrors the chrome
            // above the byline reads as crowding under a ruled row). Keeping it
            // *inside* the measured rail (rather than folding it in as a
            // separate term) is what stops the last footnote row sitting flush
            // against the window edge — and it is why `record_height` takes the
            // **max** of this measured span and the bare breath rather than
            // their sum: the breath below the composer is drawn once, here.
            rail = rail.pb(px(super::composer::rail_breath()));
        }
        for (idx, r) in refs.iter().enumerate() {
            let row = FootnoteRow {
                index: idx + 1,
                ordinal: r.ordinal as i64,
                byline: r.byline.clone(),
                body: FootnoteBody::Quote(footnote_snippet(&r.snippet).into()),
                antecedent_action_id: r.spec.antecedent_action_id.clone(),
            };
            let ordinal = r.ordinal;
            let mut el = self.footnote_row(
                format!("space-draft-fn-{}-{}", draft.id, ordinal),
                format!("space/draft/footnote/{}", row.index),
                &row,
                false,
                // A draft's rail rows never navigate — the quoted post is not
                // yet reachable from an unposted draft — so they are rows, not
                // links; the re-embed and remove chips are the affordances.
                false,
                cx,
            );
            // "embed" — put the quote back into the body. A reference and its
            // marker are separable (deleting the block leaves the reference,
            // which is what makes the footnote a backlink), so the rail owns
            // the way back. Offered only while the marker is *absent*: with the
            // block already in the body a second one would render the same
            // quote twice and confuse removal.
            if !placed.contains(&ordinal) {
                let draft_id = draft_id.clone();
                el = el.child(
                    div()
                        .id(SharedString::from(format!(
                            "space-draft-fn-embed-{}-{}",
                            draft.id, ordinal
                        )))
                        .probe(
                            format!("space/draft/footnote/{}/embed", row.index),
                            gpui::Role::Button,
                            "Embed this quote in the draft",
                        )
                        .flex_none()
                        .px_1()
                        .text_xs()
                        .text_color(cx.theme().muted_foreground)
                        .cursor_pointer()
                        .hover(|s| s.text_color(cx.theme().foreground))
                        .child("embed")
                        .on_click(cx.listener(move |this, _, window, cx| {
                            this.embed_draft_reference(&draft_id, ordinal, window, cx);
                        })),
                );
            }
            let draft_id = draft_id.clone();
            let el = el.child(
                div()
                    .id(SharedString::from(format!(
                        "space-draft-fn-rm-{}-{}",
                        draft.id, ordinal
                    )))
                    .probe(
                        format!("space/draft/footnote/{}/remove", row.index),
                        gpui::Role::Button,
                        "Remove this quote",
                    )
                    .flex_none()
                    .px_1()
                    .text_xs()
                    .text_color(cx.theme().muted_foreground)
                    .cursor_pointer()
                    .hover(|s| s.text_color(cx.theme().foreground))
                    .child("×")
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.remove_draft_reference(&draft_id, ordinal, cx);
                    })),
            );
            rail = rail.child(el);
        }
        Some(rail.into_any_element())
    }

    /// Place the marker for a pending reference the draft's body no longer
    /// carries — the rail's "embed" affordance, and the inverse of the
    /// deletion that made the quote a bare backlink. The editor owns the
    /// splice (`insert_embed_marker` pads it into its own paragraph), so the
    /// host never touches marker bytes.
    pub fn embed_draft_reference(
        &mut self,
        draft_id: &SharedString,
        ordinal: u64,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(draft) = self.drafts.iter().find(|d| &d.id == draft_id) else {
            return;
        };
        if !draft.references.iter().any(|r| r.ordinal == ordinal) {
            return;
        }
        let embeds = draft.embed_map();
        let editor = draft.editor.clone();
        editor.update(cx, |e, cx| {
            e.set_embeds(embeds, cx);
            e.insert_embed_marker(ordinal, cx);
        });
        // The marker lands at the caret, so the composer must have it: focus
        // the editor the way `attach_quote` does.
        let focus = editor.read(cx).focus_handle(cx);
        window.focus(&focus, cx);
        cx.notify();
    }

    /// The rail's container: a hairline rule above a tight column — a book's
    /// footnote apparatus, not a card.
    fn rail_frame(&self, cx: &Context<Self>) -> gpui::Div {
        v_flex()
            .mt_2()
            .pt_1p5()
            .gap_0p5()
            .border_t_1()
            .border_color(cx.theme().border)
    }

    /// One footnote row: the index, the attribution, and the passage — all in
    /// the quiet register. A reference whose stored range no longer maps says
    /// so plainly rather than guessing at a remap.
    ///
    /// `navigable` decides the **role**, which is what decides whether the row
    /// is a tab stop — a rail row is a `Link` only where the caller actually
    /// attaches the navigate handler (a post's rail at rest, never a draft's
    /// and never one mid-edit-session, where the row carries only its removal
    /// chip). `space_view::traces` already switches role the same way. Without
    /// it those rows were focusable, activatable-looking `Link`s with no click
    /// listener at all — the dead-tab-stop shape `crate::focus` exists to
    /// prevent.
    fn footnote_row(
        &self,
        element_id: String,
        probe: String,
        row: &FootnoteRow,
        marked: bool,
        navigable: bool,
        cx: &Context<Self>,
    ) -> gpui::Stateful<gpui::Div> {
        let theme = cx.theme();
        let (text, italic, color) = match &row.body {
            FootnoteBody::Quote(s) => (s.clone(), true, theme.muted_foreground),
            // Honest states: we say what we know, never a guess at what the
            // passage says now.
            FootnoteBody::Unresolvable => (
                SharedString::from("quoted an earlier version"),
                false,
                theme.muted_foreground.opacity(0.75),
            ),
            FootnoteBody::Backlink => (
                SharedString::from("referenced"),
                false,
                theme.muted_foreground.opacity(0.75),
            ),
        };
        let aria = format!("Reference {}: {} — {}", row.index, row.byline, text);
        h_flex()
            .id(SharedString::from(element_id))
            .probe(
                probe,
                if navigable {
                    gpui::Role::Link
                } else {
                    gpui::Role::ListItem
                },
                aria,
            )
            .w_full()
            .items_baseline()
            .gap_1p5()
            .text_xs()
            .when(marked, |d| d.opacity(0.45))
            .when(navigable, |d| d.cursor_pointer())
            .child(
                div()
                    .flex_none()
                    .w(px(14.))
                    .text_color(theme.muted_foreground)
                    .child(format!("{}.", row.index)),
            )
            .child(
                div()
                    .flex_none()
                    .text_color(theme.muted_foreground)
                    .child(row.byline.clone()),
            )
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .truncate()
                    .italic_when(italic)
                    .text_color(color)
                    .child(text),
            )
    }

    /// The byline to attribute a reference to: the quoted post's own, when it
    /// lives in this space. A cross-space reference resolves to nothing local,
    /// so it reads as what it is.
    fn reference_byline(&self, action_id: &str) -> SharedString {
        self.posts
            .iter()
            .find(|p| p.action_id.as_deref() == Some(action_id))
            .map(|p| p.byline.clone())
            .unwrap_or_else(|| SharedString::from("another space"))
    }

    // -- Navigation ---------------------------------------------------------

    /// Go to `action_id`: select its branch and scroll it to rest when it is
    /// in this space; otherwise resolve its home space and open that window
    /// (references are the cross-space mechanism — the Library/Record
    /// precedent for leaving the current window).
    pub fn navigate_to_action(
        &mut self,
        action_id: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.highlight_picker = None;
        let page_width = self.page_width(window);
        let turns = self.stream_overlays(cx);
        let tree = self.effective_tree(page_width, &turns);
        if super::model::node_ref(&tree, &action_id).is_some() {
            self.select_path_to(&tree, &action_id, page_width);
            self.scroll_node_into_view(&tree, &action_id, window, cx);
            cx.notify();
            return;
        }
        self.navigate_to_absent_action(action_id, window, cx);
    }

    /// A quoted post that isn't in the current-tip tree. It is **not**
    /// necessarily foreign: references name a *concrete generation*, so a post
    /// quoted and later edited or regenerated leaves the tree even though it
    /// still lives here — its item does not. So resolve the action's
    /// `(item, space)` and try the item's current tip in this tree first;
    /// only a genuinely cross-space post falls through to opening its own
    /// window (for a same-space edit that fallback would open a *duplicate*
    /// window on this space with nothing selected).
    ///
    /// The resolve is a pure read owned in the view's own slot (it dies with
    /// the window, per STATE.md's owner-is-blast-radius rule — a cancelled
    /// navigation strands nothing).
    fn navigate_to_absent_action(
        &mut self,
        action_id: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(app_core) = self.stores.app_core() else {
            return;
        };
        let rx = crate::bridge::action_location(app_core, action_id);
        let stores = self.stores.clone();
        let intent = crate::lifecycle::intend_to_open(cx);
        self.navigate_task = Some(cx.spawn_in(window, async move |this, cx| {
            let located = match rx.await {
                Ok(Ok(located)) => located,
                // **The denial the reader is allowed to see** (task 37 rule 4):
                // the resolve is membership-gated, so a quote into a
                // conversation this reader takes no part in refuses here. It is
                // said in the app's voice and says nothing about that space —
                // the typed error carries no title, participant or content, and
                // the copy adds none. Any other failure is an ordinary error
                // band; a cancelled receiver is a closing window, and silent.
                Ok(Err(err)) => {
                    this.update(cx, |this, cx| this.report_navigation_failure(err, cx))
                        .ok();
                    return;
                }
                Err(_) => return,
            };
            let Some((item_id, space_id)) = located else {
                return;
            };
            let selected = this
                .update_in(cx, |this, window, cx| {
                    this.select_item_tip(&item_id, window, cx)
                })
                .unwrap_or(false);
            if selected {
                return;
            }
            let _ = cx.update(|_, cx| {
                if intent.still_wanted(cx) {
                    crate::open_space_window(cx, stores.clone(), space_id);
                }
            });
        }));
    }

    /// A follow that could not be taken. A **refusal** is not a failure: it
    /// gets the quiet notice (the cascade band's family — muted, dismissible),
    /// worded in the app's own voice so nothing about the refused conversation
    /// can ride along in a rendered error. Anything else is a real error and
    /// takes the danger band.
    fn report_navigation_failure(&mut self, err: AppError, cx: &mut Context<Self>) {
        match err {
            AppError::NotAParticipant { .. } => {
                self.reference_notice = Some(FOLLOW_DENIED_HERE);
            }
            other => self.error = Some(other.to_string()),
        }
        cx.notify();
    }

    /// Select the post now carrying `item_id`'s current generation, if this
    /// space renders it — how a reference to a since-edited post still lands
    /// on the content it quoted. Returns whether it found one.
    ///
    /// The stored range is **not** remapped onto the new generation (the
    /// footnote already says "quoted an earlier version" when it no longer
    /// resolves); this only takes the reader to where that post now lives.
    #[doc(hidden)]
    pub fn select_item_tip(
        &mut self,
        item_id: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        let Some(tip) = self
            .space
            .read(cx)
            .messages()
            .iter()
            .find(|m| m.item_id.as_deref() == Some(item_id))
            .and_then(|m| m.action_id.clone())
        else {
            return false;
        };
        let page_width = self.page_width(window);
        let turns = self.stream_overlays(cx);
        let tree = self.effective_tree(page_width, &turns);
        if super::model::node_ref(&tree, &tip).is_none() {
            return false;
        }
        self.select_path_to(&tree, &tip, page_width);
        self.scroll_node_into_view(&tree, &tip, window, cx);
        cx.notify();
        true
    }

    /// Scroll the page so `node_id` rests near the top of the reading area —
    /// enough to read the quoted passage in place without hunting for it.
    ///
    /// **Glided, not jumped** (task 46, bug 4): the reader asked to be taken
    /// somewhere, and the travel is what tells them it is the same conversation
    /// rather than a new page. See [`super::nav::PageGlide`].
    fn scroll_node_into_view(
        &mut self,
        roots: &[TreeNode],
        node_id: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let viewport = self.page_size(window);
        let Some(doc_top) =
            self.selected_path_doc_top(roots, node_id, viewport.width, viewport.height)
        else {
            return;
        };
        let target = super::TITLE_BAR_RESERVE.as_f32() + 24.0;
        let y = (target - doc_top).min(0.0);
        self.glide_page_to(y, window, cx);
    }

    // -- Source highlights --------------------------------------------------

    /// The highlight ranges to paint on post `i`'s body: each incoming
    /// reference's stored range mapped from `(block, block-relative)` back
    /// into the body editor's buffer offsets, keyed by its index in the post's
    /// incoming list (the opaque `u64` the editor hands back on a click).
    ///
    /// A range that no longer maps onto a live block is **dropped**, never
    /// approximated — the same honesty the footnote rail's "quoted an earlier
    /// version" row states out loud.
    pub(crate) fn highlight_ranges(&self, i: usize, cx: &gpui::App) -> Vec<(Range<usize>, u64)> {
        let post = &self.posts[i];
        let Some(action_id) = post.action_id.as_deref() else {
            return Vec::new();
        };
        self.space
            .read(cx)
            .incoming_references(action_id)
            .iter()
            .enumerate()
            .filter_map(|(key, r)| {
                let block_id = r.content_block_id.as_deref()?;
                let start = usize::try_from(r.range_start?).ok()?;
                let end = usize::try_from(r.range_end?).ok()?;
                let span = post.blocks.iter().find(|b| b.block_id == block_id)?;
                let (lo, hi) = (span.range.start + start, span.range.start + end);
                (lo < hi && hi <= span.range.end).then_some((lo..hi, key as u64))
            })
            .collect()
    }

    /// A plain click on highlighted text in post `node_id`: one referencing
    /// post navigates straight there; several open the picker so the choice is
    /// the user's rather than a guess.
    pub(crate) fn on_highlight_click(
        &mut self,
        node_id: SharedString,
        keys: &[u64],
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(post) = self
            .posts
            .iter()
            .find(|p| p.action_id.as_deref() == Some(node_id.as_ref()))
        else {
            return;
        };
        let Some(action_id) = post.action_id.clone() else {
            return;
        };
        let incoming = self.space.read(cx).incoming_references(&action_id).to_vec();
        let chosen: Vec<_> = keys
            .iter()
            .filter_map(|k| incoming.get(*k as usize))
            .collect();
        match chosen.as_slice() {
            [] => {}
            [only] => {
                let target = only.action_id.clone();
                self.navigate_to_action(target, window, cx);
            }
            many => {
                self.highlight_picker = Some(super::HighlightPicker {
                    choices: many
                        .iter()
                        .map(|r| {
                            (
                                r.action_id.clone(),
                                r.space_id.clone(),
                                self.referencer_label(&r.action_id),
                            )
                        })
                        .collect(),
                });
                cx.notify();
            }
        }
    }

    /// How a referencing post reads in the picker: its byline plus the opening
    /// of what it *says*, so the choice is about the reply, not an opaque id.
    ///
    /// The post's own quote blocks are elided first — a marker is rendered
    /// content, never chrome text, and a label reading "`{{ embed 1 }}`" would
    /// leak the wire format into the UI.
    fn referencer_label(&self, action_id: &str) -> SharedString {
        match self
            .posts
            .iter()
            .find(|p| p.action_id.as_deref() == Some(action_id))
        {
            Some(p) => {
                let head = footnote_snippet(&strip_embed_blocks(&p.content, &p.references));
                if head.is_empty() {
                    p.byline.clone()
                } else {
                    SharedString::from(format!("{}: {head}", p.byline))
                }
            }
            None => SharedString::from("A post in another space"),
        }
    }

    /// The picker: a small popover of the posts that quoted the clicked
    /// passage. Dismissed by click-out or a choice — the band-menu pattern.
    pub(crate) fn render_highlight_picker(&self, cx: &Context<Self>) -> Option<AnyElement> {
        let picker = self.highlight_picker.as_ref()?;
        let theme = cx.theme();
        let mut col = v_flex()
            .id("space-highlight-picker")
            .probe(
                "space/highlight/picker",
                gpui::Role::Group,
                "Posts quoting this passage",
            )
            // An opaque popover over the page (see `crate::overlay`): a click
            // on a row must not also land in the post beneath it.
            .contain_mouse(Overlay::Popover)
            .absolute()
            .right(GUTTER_GAP)
            .bottom(px(96.))
            .w(px(280.))
            .p_1()
            .gap_0p5()
            .rounded_md()
            .border_1()
            .border_color(theme.border)
            .bg(theme.popover)
            .on_mouse_down_out(cx.listener(|this, _, _, cx| {
                this.highlight_picker = None;
                cx.notify();
            }))
            .child(
                div()
                    .px_1()
                    .pb_0p5()
                    .text_xs()
                    .text_color(theme.muted_foreground)
                    .child("Quoted by"),
            );
        for (idx, (action_id, _space_id, label)) in picker.choices.iter().enumerate() {
            let target = action_id.clone();
            col = col.child(
                div()
                    .id(SharedString::from(format!("space-highlight-pick-{idx}")))
                    .probe(
                        format!("space/highlight/picker/{idx}"),
                        gpui::Role::Button,
                        label.clone(),
                    )
                    .w_full()
                    .px_1()
                    .py_0p5()
                    .rounded_sm()
                    .text_xs()
                    .truncate()
                    .cursor_pointer()
                    .hover(|s| s.bg(theme.muted))
                    .child(label.clone())
                    .on_click(cx.listener(move |this, _, window, cx| {
                        this.navigate_to_action(target.clone(), window, cx);
                    })),
            );
        }
        Some(col.into_any_element())
    }

    // -- Test seams ---------------------------------------------------------

    /// Whether the quote-destination picker is open, and the statement it is
    /// showing (`None` while it is still a list of conversations).
    #[doc(hidden)]
    pub fn quote_destination_for_test(&self) -> Option<Option<SharedString>> {
        let dest = self.quote_destination.as_ref()?;
        Some(
            dest.confirming
                .as_ref()
                .map(|(_, title)| visibility_statement(title)),
        )
    }

    /// Open the destination picker on a made-up selection — the driver and the
    /// visual harness build a scene before any frame has minted the post
    /// editors a real selection would come from (the same reason
    /// `seed_draft_quote_for_test` exists).
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn seed_quote_destination_for_test(
        &mut self,
        action_id: &str,
        block_id: &str,
        byline: &str,
        snippet: &str,
        armed: Option<(&str, &str)>,
        cx: &mut Context<Self>,
    ) {
        self.quote_destination = Some(QuoteDestination {
            selection: PostSelection {
                node_id: SharedString::from(action_id.to_string()),
                action_id: action_id.to_string(),
                block_id: block_id.to_string(),
                range: 0..snippet.len(),
                snippet: SharedString::from(snippet.to_string()),
                byline: SharedString::from(byline.to_string()),
            },
            confirming: armed
                .map(|(id, title)| (id.to_string(), SharedString::from(title.to_string()))),
            focus: cx.focus_handle(),
        });
        cx.notify();
    }

    /// Choose a destination without a pointer (the rows are painted from the
    /// Library index; the behavior tier drives the transition, the driver the
    /// pixels).
    #[doc(hidden)]
    pub fn arm_quote_destination_for_test(
        &mut self,
        space_id: &str,
        title: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.arm_quote_destination(
            space_id.to_string(),
            SharedString::from(title.to_string()),
            window,
            cx,
        );
    }

    /// The open destination picker's subtree focus handle (tests).
    #[doc(hidden)]
    pub fn quote_destination_focus_handle(&self) -> Option<gpui::FocusHandle> {
        self.quote_destination.as_ref().map(|d| d.focus.clone())
    }

    /// Confirm the armed destination — the "Quote there" verb.
    #[doc(hidden)]
    pub fn confirm_quote_destination_for_test(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.send_quote_to_destination(window, cx);
    }

    /// Feed a resolve failure through the follow path's reporting rule — what
    /// the app *says* when a follow is refused. The refusal itself is app-core's
    /// (`action_location` is membership-gated, and its own tests pin both the
    /// gate and the non-leaking payload); this seam pins the voice.
    #[doc(hidden)]
    pub fn report_navigation_failure_for_test(&mut self, err: AppError, cx: &mut Context<Self>) {
        self.report_navigation_failure(err, cx);
    }
}

/// `italic_when` — gpui's `Styled` has `italic()` but no conditional form, and
/// the rail needs one for the honest "earlier version" row.
trait ItalicWhen: Styled + Sized {
    fn italic_when(self, yes: bool) -> Self {
        if yes { self.italic() } else { self }
    }
}
impl<T: Styled + Sized> ItalicWhen for T {}

/// `GUTTER_GAP` is re-exported for the rail's alignment with the reading
/// column; referenced here so the import isn't dead when the rail is styled
/// without it.
#[allow(dead_code)]
const _RAIL_GUTTER: gpui::Pixels = GUTTER_GAP;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::PostBlockSpan;

    fn spans() -> Vec<PostBlockSpan> {
        vec![
            PostBlockSpan {
                block_id: "a".into(),
                range: 0..10,
            },
            PostBlockSpan {
                block_id: "b".into(),
                range: 10..25,
            },
        ]
    }

    #[test]
    fn a_range_inside_one_block_resolves_block_relative() {
        let (id, r) = block_range_for(&spans(), &(12..18)).expect("resolves");
        assert_eq!(id, "b");
        assert_eq!(r, 2..8);
    }

    #[test]
    fn a_range_spanning_two_blocks_is_not_quotable() {
        assert!(block_range_for(&spans(), &(5..15)).is_none());
    }

    #[test]
    fn an_empty_range_is_not_quotable() {
        assert!(block_range_for(&spans(), &(7..7)).is_none());
        // An inverted range, written structurally (a reversed literal is a
        // clippy deny) — the selection machinery normalizes, but the guard
        // must hold for any caller.
        assert!(block_range_for(&spans(), &Range { start: 9, end: 3 }).is_none());
    }

    #[test]
    fn a_whole_block_selection_resolves() {
        let (id, r) = block_range_for(&spans(), &(0..10)).expect("resolves");
        assert_eq!(id, "a");
        assert_eq!(r, 0..10);
    }

    #[test]
    fn strip_removed_markers_drops_only_the_removed_ordinals_and_closes_the_gap() {
        let doc = "Before.\n\n{{ embed 1 }}\n\nBetween.\n\n{{ embed 2 }}\n\nAfter.";

        // The removed ordinal's marker goes, its paragraph gap closes, and the
        // surviving reference's marker is untouched — its footnote still
        // addresses it.
        let out = strip_removed_markers(doc, &[1]);
        assert_eq!(out, "Before.\n\nBetween.\n\n{{ embed 2 }}\n\nAfter.");

        // Removing every reference leaves clean prose, not a run of blanks.
        assert_eq!(
            strip_removed_markers(doc, &[1, 2]),
            "Before.\n\nBetween.\n\nAfter."
        );

        // Nothing removed is a verbatim pass-through.
        assert_eq!(strip_removed_markers(doc, &[]), doc);

        // A marker the author defused by fencing it is literal text, not a
        // block — the editor renders it literally, so removal must not touch
        // it either (the UI and the wire never disagree).
        let fenced = "Before.\n\n```\n{{ embed 1 }}\n```\n\nAfter.";
        assert_eq!(strip_removed_markers(fenced, &[1]), fenced);
    }

    #[test]
    fn strip_removed_markers_handles_document_edges() {
        // Leading marker: the gap it leaves is trimmed rather than left as an
        // opening blank line.
        assert_eq!(
            strip_removed_markers("{{ embed 1 }}\n\nBody.", &[1]),
            "Body."
        );
        // Trailing marker: likewise at the end.
        assert_eq!(
            strip_removed_markers("Body.\n\n{{ embed 1 }}", &[1]),
            "Body."
        );
        // A body that is *only* a marker empties out — `commit_edit` treats an
        // empty submission as a no-op rather than persisting a blank post.
        assert_eq!(strip_removed_markers("{{ embed 1 }}", &[1]), "");
    }

    #[test]
    fn strip_embed_blocks_elides_only_recognized_markers() {
        let r = |ordinal: i64, snippet: Option<&str>| eidola_app_core::PostReference {
            antecedent_action_id: "x".into(),
            ordinal,
            content_block_id: Some("b".into()),
            range_start: Some(0),
            range_end: Some(4),
            annotation: None,
            snippet: snippet.map(String::from),
        };
        // A mapped marker standing as its own paragraph is elided.
        let doc = "before\n\n{{ embed 1 }}\n\nafter";
        assert_eq!(
            footnote_snippet(&strip_embed_blocks(doc, &[r(1, Some("quoted"))])),
            "before after"
        );
        // An *unmapped* ordinal is literal text everywhere — including here.
        assert!(strip_embed_blocks(doc, &[r(2, Some("quoted"))]).contains("{{ embed 1 }}"));
        // So is one the author defused inside a fence.
        let fenced = "before\n\n```\n{{ embed 1 }}\n```\n\nafter";
        assert!(strip_embed_blocks(fenced, &[r(1, Some("quoted"))]).contains("{{ embed 1 }}"));
        // No references at all: untouched.
        assert_eq!(strip_embed_blocks(doc, &[]), doc);
    }

    #[test]
    fn footnote_snippet_flattens_and_truncates_on_a_word_boundary() {
        assert_eq!(footnote_snippet("  a\n\nb   c "), "a b c");
        let long = "word ".repeat(40);
        let out = footnote_snippet(&long);
        assert!(out.ends_with('…'));
        assert!(out.chars().count() <= 97);
        assert!(!out.contains("  "));
        // Never cuts mid-word.
        assert!(out.trim_end_matches('…').ends_with("word"));
    }
}
