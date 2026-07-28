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

use crate::actions::{Quote, QuoteInReply};
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
fn strip_embed_blocks(content: &str, references: &[eidola_app_core::PostReference]) -> String {
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
    const MAX: usize = 96;
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= MAX {
        return flat;
    }
    let cut = flat
        .char_indices()
        .nth(MAX)
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
        let viewport = crate::chrome::content_size(window);
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
    /// marker: the editor learns the embed map first (so the marker
    /// materializes as a quote block the instant it lands), then the marker is
    /// inserted at the caret through the editor's normal update pipeline (one
    /// undo step; a marker dropped into a verbatim region degrades to literal
    /// text, which is the documented honest behavior).
    fn attach_quote(
        &mut self,
        draft_id: &SharedString,
        selection: PostSelection,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(draft) = self.drafts.iter_mut().find(|d| &d.id == draft_id) else {
            return;
        };
        let ordinal = draft.next_ordinal();
        draft.references.push(PendingReference {
            ordinal,
            spec: selection.spec(),
            byline: selection.byline.clone(),
            snippet: selection.snippet.clone(),
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
            // carries the bar's bottom breath — [`composer::bottom_breath`],
            // the mirror of the `half_pad` chrome above the byline. Keeping it
            // *inside* the measured rail (rather than folding it in as a
            // separate term) is what stops the last footnote row sitting flush
            // against the window edge — and it is why `record_height` takes the
            // **max** of this measured span and the bare breath rather than
            // their sum: the breath below the composer is drawn once, here.
            rail = rail.pb(px(super::composer::bottom_breath()));
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
    fn footnote_row(
        &self,
        element_id: String,
        probe: String,
        row: &FootnoteRow,
        marked: bool,
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
            .probe(probe, gpui::Role::Link, aria)
            .w_full()
            .items_baseline()
            .gap_1p5()
            .text_xs()
            .when(marked, |d| d.opacity(0.45))
            .cursor_pointer()
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
        let page_width = crate::chrome::content_size(window).width;
        let turns = self.stream_overlays(cx);
        let tree = self.effective_tree(page_width, &turns);
        if super::model::node_ref(&tree, &action_id).is_some() {
            self.select_path_to(&tree, &action_id, page_width);
            self.scroll_node_into_view(&tree, &action_id, window);
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
        self.navigate_task = Some(cx.spawn_in(window, async move |this, cx| {
            let Ok(Ok(Some((item_id, space_id)))) = rx.await else {
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
                crate::open_space_window(cx, stores.clone(), space_id);
            });
        }));
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
        let page_width = crate::chrome::content_size(window).width;
        let turns = self.stream_overlays(cx);
        let tree = self.effective_tree(page_width, &turns);
        if super::model::node_ref(&tree, &tip).is_none() {
            return false;
        }
        self.select_path_to(&tree, &tip, page_width);
        self.scroll_node_into_view(&tree, &tip, window);
        cx.notify();
        true
    }

    /// Scroll the page so `node_id` rests near the top of the reading area —
    /// enough to read the quoted passage in place without hunting for it.
    fn scroll_node_into_view(&self, roots: &[TreeNode], node_id: &str, window: &mut Window) {
        let viewport = crate::chrome::content_size(window);
        let Some(doc_top) =
            self.selected_path_doc_top(roots, node_id, viewport.width, viewport.height)
        else {
            return;
        };
        let target = super::TITLE_BAR_RESERVE.as_f32() + 24.0;
        let y = (target - doc_top).min(0.0);
        let off = self.page_scroll.offset();
        self.page_scroll.set_offset(gpui::point(off.x, px(y)));
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
