//! Drafts + the composer — the unsent-reply model.
//!
//! A band's "+" creates a [`Draft`](super::Draft): a local UI node with its own
//! editor, attached to the persisted tree as a leaf of the post it replies to.
//! Drafts **persist** when deselected (they keep their content and their place
//! in the tree, taking real vertical space and tinting their branch's
//! navigation dots), exactly like the `examples/eidola_space.rs` mockup.
//!
//! One draft is *active* at a time — the focused one — and adopts the
//! floating/docking composer behavior: its editor floats over the bottom, its
//! in-flow slot a placeholder. Focusing another draft (or Escape) retires the
//! current one; a retired **empty** draft is deleted, leaving no trace.
//!
//! `⌘↩` (`Send`) **posts** the active draft — the space's participants decide
//! who responds (notify policies drive one streaming turn per responder);
//! `⌘⇧↩` (`PostOnly`, the ⌥-revealed "Post quietly") persists it without
//! notifying anyone. Both consume the draft. The composer carries no model
//! picker — who answers, and with what model, is Participants configuration
//! (`Space > Participants…`); explicit asks live on the separators.

use gpui::{
    AnyElement, App, AppContext, Bounds, BoxShadow, Context, Element, Entity, Focusable,
    GlobalElementId, InspectorElementId, InteractiveElement, IntoElement, KeyDownEvent, LayoutId,
    ParentElement, Pixels, ScrollWheelEvent, SharedString, StatefulInteractiveElement, Styled,
    TouchPhase, Window, div, hsla, linear_color_stop, linear_gradient, point,
    prelude::FluentBuilder as _, px, size,
};
use gpui_component::{ActiveTheme, h_flex};
use gpui_markdown_editor::{MarkdownEditor, MarkdownEditorEvent, MarkdownEditorState};

use std::collections::HashSet;

use crate::loadable::Loadable;

use super::context_menu::ContextTarget;
use super::layout::body_width;
use super::model::{self, TreeNode};

/// Vertical offset of the floating composer's drop shadow (negative = cast
/// upward, over the conversation) and its blur radius. Named so the stacking
/// layer that hosts the composer can dilate its bounds to match the shadow's
/// visible reach — see the `layered(..)` call in `render_active_draft`.
const SHADOW_OFFSET_Y: Pixels = px(-3.);
const SHADOW_BLUR: Pixels = px(18.);
use super::nav::ScrollOwner;
use super::{Draft, GUTTER_GAP, POST_PAD_Y, PostOnly, Send, SpaceView, prose_style};

impl SpaceView {
    // -- Draft lifecycle ---------------------------------------------------

    /// Mint a draft node replying to `parent`: a fresh editor wired to activate
    /// on focus and submit on the Enter chords, pushed into `drafts`. Returns its
    /// id. Shared by the focused-fork path ([`Self::create_draft`]) and the
    /// auto-ensured docked tail drafts ([`Self::sync_tail_drafts`]).
    fn create_draft_node(
        &mut self,
        parent: Option<gpui::SharedString>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> gpui::SharedString {
        self.next_draft_seq += 1;
        let id = gpui::SharedString::from(format!("draft-{}", self.next_draft_seq));
        let editor = cx.new(|cx| MarkdownEditorState::new(window, cx));

        // The draft's editor drives both activation (focus) and submit
        // (`PressEnter`): focusing it makes it the active draft; the modified
        // Enter chords route to the save-vs-ask gestures.
        let sub_id = id.clone();
        let sub =
            cx.subscribe_in(
                &editor,
                window,
                move |this, _editor, event, window, cx| match event {
                    MarkdownEditorEvent::Focus if this.active_draft.as_ref() != Some(&sub_id) => {
                        this.activate_draft(sub_id.clone(), cx);
                    }
                    // A buffer edit in the active draft: arm the caret
                    // scroll-into-view (consumed by the composer body's
                    // `caret_into_view` canvas next paint, once the post-edit
                    // layout is fresh). Only the active draft renders as the
                    // scrollable composer, so ignore Changes from any other.
                    MarkdownEditorEvent::Change if this.active_draft.as_ref() == Some(&sub_id) => {
                        this.composer_caret_scroll_pending.set(true);
                        cx.notify();
                    }
                    // Keyboard caret movement with no buffer change (arrows,
                    // Home/End, word moves) must scroll the caret into view too —
                    // same arming as an edit, same active-draft guard.
                    MarkdownEditorEvent::SelectionChanged
                        if this.active_draft.as_ref() == Some(&sub_id) =>
                    {
                        this.composer_caret_scroll_pending.set(true);
                        cx.notify();
                    }
                    MarkdownEditorEvent::PressEnter {
                        secondary: true,
                        shift,
                    } => {
                        if *shift {
                            this.post_only(&PostOnly, window, cx);
                        } else {
                            this.submit(&Send, window, cx);
                        }
                    }
                    _ => {}
                },
            );

        self.drafts.push(Draft {
            id: id.clone(),
            parent,
            editor,
            references: Vec::new(),
            _sub: sub,
        });
        id
    }

    /// Test/scene seam: mint an **active** draft already carrying a body and
    /// pending quoted references, without replaying the selection gesture.
    ///
    /// The gesture itself (`Edit > Quote`) is covered by the behavior tests;
    /// this exists for the driver's composing scene, which cannot perform it —
    /// a real quote needs the post's body editor and the branch's tail draft,
    /// both minted during `render`, and the driver's headless dispatcher does
    /// not pump `on_next_frame`. Seeding declaratively (like every other
    /// scene's `set_post_tree_for_test`) renders the same state.
    ///
    /// `quotes` are `(ordinal, byline, snippet)` triples; the body is expected
    /// to carry the matching `{{ embed N }}` markers.
    #[doc(hidden)]
    pub fn seed_draft_quote_for_test(
        &mut self,
        parent: Option<&str>,
        body: &str,
        quotes: Vec<(u64, &str, &str)>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let id = self.create_draft_node(parent.map(gpui::SharedString::from), window, cx);
        let Some(draft) = self.drafts.iter_mut().find(|d| d.id == id) else {
            return;
        };
        for (ordinal, byline, snippet) in quotes {
            draft.references.push(super::PendingReference {
                ordinal,
                spec: eidola_app_core::ReferenceSpec {
                    antecedent_action_id: parent.unwrap_or_default().to_string(),
                    content_block_id: Some("blk-1".into()),
                    range_start: Some(0),
                    range_end: Some(snippet.len() as i64),
                    annotation: None,
                },
                byline: byline.into(),
                snippet: snippet.into(),
            });
        }
        let embeds = draft.embed_map();
        let editor = draft.editor.clone();
        let body = body.to_string();
        editor.update(cx, |e, cx| {
            e.set_embeds(embeds, cx);
            e.set_value(body, cx);
        });
        self.activate_draft(id, cx);
        let focus = editor.read(cx).focus_handle(cx);
        window.focus(&focus, cx);
    }

    /// Open a draft replying to `parent` (a band's "+" / ⌘N): make it the active
    /// (floating) composer and focus it. The branch selection + page dock happen
    /// on the next render (`pending_select`), against the real tree.
    pub(crate) fn create_draft(
        &mut self,
        parent: Option<gpui::SharedString>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let id = self.create_draft_node(parent, window, cx);
        let focus = self.drafts.last().unwrap().editor.read(cx).focus_handle(cx);
        self.activate_draft(id.clone(), cx);
        self.pending_select = Some(id);
        window.focus(&focus, cx);
    }

    /// Ensure every branch leaf has a **tail draft** — the always-present,
    /// docked end-of-branch composer that replaces the leaf "+": a draft whose
    /// parent is a leaf (a post nothing replies to; the root `None` for a blank
    /// space). Empty tail drafts persist (docked); empty, inactive drafts whose
    /// parent is *not* a leaf (orphans, or a fork whose branch was committed) are
    /// pruned. Skipped while streaming and until the transcript has loaded (so we
    /// know the real leaves). Runs each frame in `render` (cheap + idempotent).
    pub(crate) fn sync_tail_drafts(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.space.read(cx).is_streaming() {
            return;
        }
        if !matches!(self.space.read(cx).transcript(), Loadable::Loaded { .. }) {
            return;
        }

        let leaf_parents = self.tail_parents(); // leaf action ids (Some-parents)
        let want_root = self.posts.is_empty(); // blank space → root (None) tail draft

        // Ensure the blank-space root tail draft.
        if want_root && !self.drafts.iter().any(|d| d.parent.is_none()) {
            self.create_draft_node(None, window, cx);
        }
        // Ensure a docked tail draft for each leaf that lacks one.
        for parent in &leaf_parents {
            let has = self
                .drafts
                .iter()
                .any(|d| d.parent.as_deref() == Some(parent.as_ref()));
            if !has {
                self.create_draft_node(Some(parent.clone()), window, cx);
            }
        }

        // Prune empty, inactive drafts whose parent is no longer a tail parent
        // (an orphan, or a fork whose branch was committed). Compute the tail-set
        // as a local so the filter borrows no `self` method.
        let leaf_set: HashSet<&str> = leaf_parents.iter().map(|s| s.as_ref()).collect();
        let active = self.active_draft.clone();
        let stale: Vec<gpui::SharedString> = self
            .drafts
            .iter()
            .filter(|d| {
                let is_tail = match d.parent.as_deref() {
                    None => want_root,
                    Some(p) => leaf_set.contains(p),
                };
                Some(&d.id) != active.as_ref() && !is_tail && d.editor.read(cx).is_empty()
            })
            .map(|d| d.id.clone())
            .collect();
        for id in stale {
            self.delete_draft(&id);
        }
    }

    /// Make `id` the active (floating) draft, retiring the previous one (which
    /// is deleted if it was an empty *fork*; an empty tail draft just docks).
    /// Resets the shared composer scroll.
    pub(crate) fn activate_draft(&mut self, id: gpui::SharedString, cx: &mut Context<Self>) {
        if self.active_draft.as_ref() != Some(&id) {
            self.retire_active_draft(cx);
        }
        // A reader who has started composing again owns the viewport: from here
        // the composer's own caret-into-view path drives the page, and the
        // post-submit pin must not race it (see `follow_streaming_tail`).
        self.tail_pin = false;
        // An editing session is beginning: seed the accessible value from the
        // draft as it stands. This is the seam every session passes through —
        // the band's Reply, a click on an inactive draft, the editor's own
        // `Focus` event, the quote paths — which is why it is *here* and not at
        // retirement. Without it a draft retired and re-opened (Escape, then
        // clicking back in) reports its pre-typing text forever: retirement
        // clears `active_draft` before any frame renders the composer
        // unfocused, and re-opening focuses the *same* draft, so neither of
        // `composer_aria_value`'s render-time conditions can ever fire.
        self.seed_composer_aria_value(&id, cx);
        self.active_draft = Some(id);
        self.composer_scroll.set_offset(point(px(0.), px(0.)));
        self.composer_prev_off_y = 0.0;
        self.composer_dock_runway = None;
        cx.notify();
    }

    /// Write the draft's current buffer into the accessible-value snapshot.
    /// A no-op when the id names no live draft (the caller's own guard clauses
    /// already cover that; this just refuses to invent a value).
    fn seed_composer_aria_value(&self, id: &gpui::SharedString, cx: &Context<Self>) {
        let Some(draft) = self.drafts.iter().find(|d| d.id == *id) else {
            return;
        };
        let text = SharedString::from(draft.editor.read(cx).value().to_string());
        *self.composer_aria_value.borrow_mut() = (id.clone(), text);
    }

    /// Re-activate a persisted (inactive) draft by index — the same call the
    /// inactive-draft click and the editor's `Focus` event make.
    #[doc(hidden)]
    pub fn activate_draft_for_test(&mut self, index: usize, cx: &mut Context<Self>) {
        if let Some(id) = self.drafts.get(index).map(|d| d.id.clone()) {
            self.activate_draft(id, cx);
        }
    }

    /// Deactivate the active draft (Escape / external request). An empty **tail**
    /// draft stays as the docked end-of-branch composer; an empty **fork** draft
    /// (a transient new branch) is deleted; a draft with content stays inline.
    pub(crate) fn deactivate_active_draft(&mut self, cx: &mut Context<Self>) {
        if self.active_draft.is_some() {
            self.retire_active_draft(cx);
            cx.notify();
        }
    }

    /// Clear the active draft; delete it only if it was an **empty fork** (an
    /// abandoned new branch). An empty tail draft is kept (docked) — it's the
    /// always-present composer at the end of its branch.
    fn retire_active_draft(&mut self, cx: &mut Context<Self>) {
        let Some(id) = self.active_draft.take() else {
            return;
        };
        // Any open band menu belongs to the retiring interaction — and so
        // does an exact-height resize: deactivating (or switching drafts)
        // reverts the sizing to the resting Max.
        self.band_menu = None;
        self.reset_composer_sizing();
        let Some(draft) = self.drafts.iter().find(|d| d.id == id) else {
            return;
        };
        let empty = draft.editor.read(cx).is_empty();
        let parent = draft.parent.clone();
        if empty && !self.is_tail_parent(parent.as_deref()) {
            self.delete_draft(&id);
        }
    }

    /// Remove a draft from the tree and forget its editor/state. The parent
    /// scroller re-clamps onto a still-valid branch at render time
    /// (`active_child_index` clamps to the new child count).
    fn delete_draft(&mut self, id: &str) {
        self.drafts.retain(|d| d.id != id);
        self.layout.retain(&|live| live != id);
    }

    // -- Post --------------------------------------------------------------

    /// Route the composer's outward events when dispatched as actions (tests /
    /// menu). The editor's own subscription (see `create_draft`) is the
    /// production path; this keeps `Send`/`PostOnly` dispatchable.
    pub(crate) fn submit(&mut self, _: &Send, window: &mut Window, cx: &mut Context<Self>) {
        self.send_active_draft(false, window, cx);
    }

    pub(crate) fn post_only(&mut self, _: &PostOnly, window: &mut Window, cx: &mut Context<Self>) {
        self.send_active_draft(true, window, cx);
    }

    /// Persist the active draft — **Post** drives the space's notification
    /// plan (participants with matching notify policies respond, concurrently);
    /// `quiet` skips the plan (nobody is asked). Consumes the draft **only when
    /// the space accepts it**. A no-op with no active draft or on an empty
    /// draft; a rejected submit (the space is busy) leaves the draft intact and
    /// active.
    fn send_active_draft(&mut self, quiet: bool, window: &mut Window, cx: &mut Context<Self>) {
        let Some(active) = self.active_draft.clone() else {
            return;
        };
        let Some(draft) = self.drafts.iter().find(|d| d.id == active) else {
            return;
        };
        let editor = draft.editor.clone();
        let parent = draft.parent.clone();
        let mut pending = draft.references.clone();
        pending.sort_by_key(|r| r.ordinal);
        // Compact the draft's ordinals onto `1..=N` at the durability
        // boundary. Draft ordinals may gap (removing a pending reference must
        // never renumber the survivors, whose markers already address them),
        // but app-core assigns edge ordinals `1..=N` **in supplied order** —
        // so the body's markers are rewritten to match in the same move. See
        // `compact_draft_references`.
        let (prompt, references) = compact_draft_references(editor.read(cx).value(), &pending);
        let prompt = prompt.trim().to_string();
        if prompt.is_empty() {
            return;
        }

        // The reply antecedent is the draft's parent, but only a real persisted
        // action id is valid (a `None`/synthetic parent → app-core uses the
        // tail).
        let reply_to = parent.and_then(|p| {
            self.posts
                .iter()
                .find(|post| post.action_id.as_deref() == Some(p.as_ref()))
                .and_then(|post| post.action_id.clone())
                .map(|s| s.to_string())
        });

        // Attempt the save/post FIRST, and consume the draft only if the space
        // **accepts** it. `Space::submit`/`post_only` return `false` whenever
        // the space is busy — a streaming turn *or* an in-flight save/plan whose
        // `post_runner` is occupied but which isn't streaming yet. Consuming the
        // draft up front (the previous behavior, which only checked
        // `is_streaming`) permanently lost the typed content when a Post landed
        // in that save window: the draft was deleted and the `false` ignored. A
        // rejected submit must leave the draft intact and active. Both mutations
        // and the consume below run in this one synchronous handler, so no frame
        // ever renders the draft beside the optimistic post.
        //
        // The draft's pending references ride the same accept-before-consume
        // contract: a rejected post leaves the draft **and its references**
        // exactly as they were (the compaction above is computed, never
        // written back), so nothing about the quote is lost.
        let accepted = self.space.update(cx, |s, cx| {
            if quiet {
                s.post_only(prompt, reply_to, references, cx)
            } else {
                s.submit(prompt, reply_to, references, cx)
            }
        });
        if !accepted {
            return;
        }

        // Consume the draft (it's becoming a persisted post). Posting is a
        // deactivation, so the exact-height resize reverts with it.
        self.active_draft = None;
        self.reset_composer_sizing();
        self.delete_draft(&active);
        self.error = None;
        self.band_menu = None;
        self.cascade_notice = None;

        self.settle_on_new_post(window, cx);
        cx.notify();
    }

    /// Land the reader on the post they just made: re-derive the snapshot so
    /// the new turn is in it, select the branch it joined, park the page at the
    /// end of that branch, and arm the tail pin.
    ///
    /// Each step is load-bearing, and the order is:
    ///
    /// - **Rebuild first.** `Space::submit`/`post_only` append the optimistic
    ///   user turn and *emit* `MessagesChanged`; gpui delivers that event after
    ///   this handler returns, so `self.posts` is still the pre-post snapshot
    ///   here and every measurement below would describe the document *before*
    ///   the post — scrolling to a "tail" a post short of the real one, which is
    ///   what left the window above the end (and tail-following disengaged).
    /// - **Then select.** The post lands where the draft was, so the branch is
    ///   usually already right — but saying so explicitly is what guarantees the
    ///   response attaches and renders where the reader is looking, the same
    ///   select-before-render rule the explicit asks follow (PR #218).
    /// - **Then scroll**, against the selected branch.
    /// - **Then pin**, because the document is not done growing: the persisted
    ///   reload re-keys the post (new node id → unmeasured → estimate, then the
    ///   warm pass measures it) and the response's own leaf follows. See
    ///   [`SpaceView::follow_streaming_tail`].
    fn settle_on_new_post(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.rebuild(cx);

        let page_width = crate::chrome::content_size(window).width;
        let turns = self.stream_overlays(cx);
        let tree = self.effective_tree(page_width, &turns);
        // The optimistic turn is the last row of the fresh snapshot.
        if let Some(last) = self.posts.len().checked_sub(1) {
            let node = model::node_id(&self.posts, last);
            if model::node_ref(&tree, &node).is_some() {
                self.select_path_to(&tree, &node, page_width);
            }
        }
        self.scroll_to_tail(window, cx);
        self.tail_pin = true;
    }

    // -- Asks --------------------------------------------------------------

    /// Route an explicit ask — a separator's `Ask ▸ <participant>`, the
    /// cascade notice's "Ask to continue", or a retry — to the shared
    /// [`Space::ask`]. Closes any open band menu + the cascade notice, applies
    /// the tail-draft rule, selects the target's branch so the streaming node
    /// renders where the reply will land, and scrolls it into view.
    ///
    /// **The tail-draft rule** (asking at the end of a branch): an **empty**
    /// draft replying to the target is discarded — the UI tracks the new tail
    /// (once the response lands, `sync_tail_drafts` docks a fresh composer
    /// under it); a draft **with content** is kept exactly as it is — it
    /// becomes its own sibling branch beside the incoming response and stays
    /// active in the composer if it was.
    pub fn ask_participant(
        &mut self,
        participant_id: String,
        target_action_id: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.band_menu = None;
        self.cascade_notice = None;
        self.error = None;

        // The tail-draft rule: discard an *empty* draft on the ask's target.
        let empty_draft = self
            .drafts
            .iter()
            .find(|d| {
                d.parent.as_deref() == Some(target_action_id.as_str())
                    && d.editor.read(cx).is_empty()
            })
            .map(|d| d.id.clone());
        if let Some(id) = empty_draft {
            if self.active_draft.as_ref() == Some(&id) {
                self.active_draft = None;
                self.reset_composer_sizing();
                window.focus(&self.focus_handle, cx);
            }
            self.delete_draft(&id);
        }

        let accepted = self.space.update(cx, |s, cx| {
            s.ask(participant_id, target_action_id.clone(), cx)
        });
        if accepted {
            // Select the target's branch *before* the next render so the
            // streaming node attaches under the asked post, not whatever
            // branch was selected (the PR #218 retry lesson).
            let page_width = crate::chrome::content_size(window).width;
            let roots = model::build_tree(&self.posts);
            if model::node_ref(&roots, &target_action_id).is_some() {
                self.select_path_to(&roots, &target_action_id, page_width);
            }
            self.scroll_to_tail(window, cx);
        }
        cx.notify();
    }

    // -- Scrolling ---------------------------------------------------------

    /// Scroll the page so the bottom of the selected branch (the composer /
    /// streaming leaf) sits at the window bottom.
    pub(crate) fn scroll_to_tail(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let viewport = crate::chrome::content_size(window);
        let turns = self.stream_overlays(cx);
        let tree = self.effective_tree(viewport.width, &turns);
        let total = self.selected_total_height(&tree, viewport.width, viewport.height);
        let doc = self.doc_reserve() + total;
        let y = (viewport.height.as_f32() - doc).min(0.0);
        let off = self.page_scroll.offset();
        self.page_scroll.set_offset(point(off.x, px(y)));
    }

    /// Dock the active draft at its "home": its slot top around 40% of the
    /// window, computed from the (already-selected) tree.
    pub(crate) fn dock_active_draft(
        &self,
        roots: &[TreeNode],
        page_width: gpui::Pixels,
        window_h: gpui::Pixels,
    ) {
        let doc_top = self.placeholder_doc_top(roots, page_width, window_h);
        let target = window_h.as_f32() * 0.4;
        let y = (target - doc_top).min(0.0);
        let off = self.page_scroll.offset();
        self.page_scroll.set_offset(point(off.x, px(y)));
    }

    /// "See in context": dock the active draft back at its place in the branch.
    pub(crate) fn go_home(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let viewport = crate::chrome::content_size(window);
        let turns = self.stream_overlays(cx);
        let tree = self.effective_tree(viewport.width, &turns);
        if let Some(active) = self.active_draft.clone()
            && model::node_ref(&tree, &active).is_some()
        {
            self.select_path_to(&tree, &active, viewport.width);
            self.dock_active_draft(&tree, viewport.width, viewport.height);
        }
        cx.notify();
    }

    /// The pre-dock glide: ease a scrolled floating composer's internal offset
    /// toward its top as the page approaches the dock threshold, so the content
    /// sits at exactly its top the moment the composer docks — instead of
    /// docking mid-scroll and unwinding afterwards.
    ///
    /// **Zone-gated.** The glide engages only while the dock threshold sits
    /// *under* the floating bar: the draft's would-be dock top (`slot_top +
    /// half_pad`, the same value the dock decision in
    /// [`Self::render_active_draft`] compares against the float line) between
    /// the float line and the window bottom — the last `float_bar_h` (at most
    /// half a window) of page travel before docking. Outside that zone a page
    /// scroll never moves the internal content ([`dock_runway`] saturates, so
    /// consecutive frames read equal runways and step nothing).
    ///
    /// **Robust by form, not by bookkeeping.** Each step scales the offset by
    /// the ratio of runway remaining ([`approach_glide_offset`]), and ratios
    /// telescope: any monotone descent through the zone yields the same offset
    /// at a given runway no matter where the scroll stopped and restarted, a
    /// retreat (scrolling away from the dock) deliberately leaves the offset
    /// alone — the glide never reverses — and a step that lands on (or jumps
    /// past) the threshold multiplies by zero, so the offset is **exactly**
    /// zero at dock by construction, with no accumulated error to correct. The
    /// dock ramp ([`composer_bar_h`]) remains the geometric backstop: once
    /// docked, the growing scroll viewport clamps out any residual offset from
    /// a path this glide can't see (a mid-zone resize, a document reflow).
    ///
    /// Runs once per `render`, after the frame's page offset is final and
    /// before the frozen-offset baseline (`composer_prev_off_y`) is taken — so
    /// every page-scroll source drives it (wheel anywhere in the window,
    /// minimap drag, tail-follow, programmatic docks like [`Self::go_home`]),
    /// not just wheel events over the bar, and the wheel handler's
    /// frozen-offset restore can never undo an eased step.
    pub(crate) fn glide_composer_toward_dock(
        &mut self,
        roots: &[TreeNode],
        page_width: gpui::Pixels,
        window_h: gpui::Pixels,
    ) {
        // Only an active draft on its own branch ever docks; anything else
        // resets the tracking so a stale runway can't seed a bogus first step
        // when the composer next lands on-path.
        let on_path = match self.active_draft.as_ref() {
            Some(active) => self.selected_leaf_id(roots, page_width).as_ref() == Some(active),
            None => false,
        };
        if !on_path {
            self.composer_dock_runway = None;
            return;
        }
        // A resize drag moves the float line itself: runway deltas during it
        // are the handle, not page travel toward the dock, and easing on them
        // would unwind scroll the user didn't approach with. Suspend and
        // re-baseline once the drag ends.
        if self.composer_resize.is_some() {
            self.composer_dock_runway = None;
            return;
        }

        let win = window_h.as_f32();
        let float_bar_h = self.composer_float_bar_h(window_h);
        let float_top = win - float_bar_h;
        let half_pad = POST_PAD_Y.as_f32() / 2.0;
        let dock_top = self.placeholder_doc_top(roots, page_width, window_h)
            + self.clamped_scroll_y()
            + half_pad;
        let runway = dock_runway(dock_top, float_top, win);
        if let Some(prev) = self.composer_dock_runway {
            let off = self.composer_scroll.offset();
            let eased = approach_glide_offset(off.y.as_f32(), prev, runway);
            if (eased - off.y.as_f32()).abs() > 0.01 {
                self.composer_scroll.set_offset(point(off.x, px(eased)));
            }
        }
        self.composer_dock_runway = Some(runway);
    }

    // -- The separator resize handle -----------------------------------------

    /// Begin a separator-handle resize drag: snapshot the grab (pointer `y` +
    /// the bar's current height) and switch the sizing to **Exact**
    /// immediately — grabbing the handle *means* sizing the bar independent of
    /// its content, including in excess of it. The fraction snapshots the
    /// bar's *current* ratio rather than the stored cap, so nothing jumps
    /// under the grab: a Max bar resting below its cap starts the drag exactly
    /// where it rests.
    pub(crate) fn start_composer_resize(
        &mut self,
        pointer_y: f32,
        bar_h: f32,
        win: f32,
        cx: &mut Context<Self>,
    ) {
        if win <= 0.0 {
            return;
        }
        self.composer_fraction = (bar_h / win).clamp(0.01, COMPOSER_FRACTION_MAX);
        self.composer_sizing = ComposerSizing::Exact;
        self.composer_resize = Some(ComposerResizeDrag {
            start_y: pointer_y,
            start_bar_h: bar_h,
        });
        cx.notify();
    }

    /// One motion step of the resize drag: the bar's top edge follows the
    /// pointer as a delta from the grab (up = taller), clamped to the
    /// fraction bounds. A no-op with no drag in flight, so the window-global
    /// listeners can stay registered unconditionally (the minimap pattern).
    pub(crate) fn update_composer_resize(
        &mut self,
        pointer_y: f32,
        win: f32,
        cx: &mut Context<Self>,
    ) {
        let Some(drag) = self.composer_resize else {
            return;
        };
        if win <= 0.0 {
            return;
        }
        let bar_h = drag.start_bar_h + (drag.start_y - pointer_y);
        let fraction = (bar_h / win).clamp(COMPOSER_FRACTION_MIN, COMPOSER_FRACTION_MAX);
        if (fraction - self.composer_fraction).abs() > f32::EPSILON {
            self.composer_fraction = fraction;
            cx.notify();
        }
    }

    /// End the resize drag (mouse-up, or a move with no button held). The
    /// sizing stays **Exact** — only deactivation reverts it.
    pub(crate) fn end_composer_resize(&mut self, cx: &mut Context<Self>) {
        if self.composer_resize.take().is_some() {
            cx.notify();
        }
    }

    /// Revert the sizing mode to the resting **Max** and drop any in-flight
    /// resize drag. Called whenever the composer deactivates (Escape, a post
    /// consuming the draft, an ask discarding it) or a different draft is
    /// selected. The *fraction* deliberately survives — it remains this
    /// window's cap until the window closes or the handle is dragged again;
    /// only the exact-height pin reverts.
    pub(crate) fn reset_composer_sizing(&mut self) {
        self.composer_sizing = ComposerSizing::Max;
        self.composer_resize = None;
    }

    /// The composer's accessible value, **frozen while it holds keyboard
    /// focus**.
    ///
    /// The composer is the one place in the app whose text changes on every
    /// keystroke, and assistive technology re-reads a focused control's entire
    /// value whenever it changes — so a value wired straight to the buffer
    /// would re-speak the whole draft per character. Zed's own text field
    /// freezes for exactly this reason, and the audit (§4) makes it the rule
    /// for us.
    ///
    /// So it refreshes at settled moments only. Two are decided here: a frame
    /// where the composer does **not** hold focus — precisely when a
    /// screen-reader user is reviewing the draft rather than writing it — and
    /// (as a backstop) a frame whose draft is not the one the snapshot names,
    /// for any path that makes a draft active without going through
    /// [`Self::activate_draft`]. The third and most important is decided
    /// *there*: **activation seeds the snapshot**, which is what makes a
    /// retired-and-re-opened draft report the text it actually holds.
    fn composer_aria_value(
        &self,
        draft_id: &SharedString,
        editor: &Entity<MarkdownEditorState>,
        window: &Window,
        cx: &Context<Self>,
    ) -> SharedString {
        let state = editor.read(cx);
        let mut snapshot = self.composer_aria_value.borrow_mut();
        if snapshot.0 != *draft_id || !state.focus_handle(cx).is_focused(window) {
            snapshot.0 = draft_id.clone();
            snapshot.1 = SharedString::from(state.value().to_string());
        }
        snapshot.1.clone()
    }

    /// The composer's current accessible value — the frozen snapshot
    /// [`Self::composer_aria_value`] maintains. `aria_value` reaches the
    /// emitted AccessKit tree, which is unobservable at this pin, so the freeze
    /// itself is regression-tested through the computation.
    #[doc(hidden)]
    pub fn composer_aria_value_for_test(&self) -> SharedString {
        self.composer_aria_value.borrow().1.clone()
    }

    // -- The floating composer ---------------------------------------------

    /// The active draft's editor, pinned over the bottom and styled like a post.
    /// Floats (bottom-aligned, overlaying the conversation) when its in-flow slot
    /// is below the fold or when the user has swiped to another branch while
    /// editing; docks to the slot when scrolled to on its own branch. Renders
    /// nothing when no draft is active.
    pub(crate) fn render_active_draft(
        &self,
        roots: &[TreeNode],
        page_width: gpui::Pixels,
        window_h: gpui::Pixels,
        window: &Window,
        cx: &Context<Self>,
    ) -> AnyElement {
        let Some(active) = self.active_draft.clone() else {
            return div().into_any_element();
        };
        let Some(draft) = self.drafts.iter().find(|d| d.id == active) else {
            return div().into_any_element();
        };
        let editor = draft.editor.clone();
        let draft_id = draft.id.clone();
        // Height the pending-reference rail claims from the composer bar — the
        // distance between the two flow marks that bracketed it last frame,
        // not a row-count formula. It is subtracted from the editor's runway
        // floor below so the two share the bar, and folded into the bar's
        // natural height by `record_height`, so the rail is part of what the
        // composer sizes itself to rather than something hanging below the
        // fold.
        let draft_rail_h =
            (self.composer_rail_bottom.get() - self.composer_rail_top.get()).max(0.0);
        let theme = cx.theme();
        let bw = px(body_width(page_width));

        // On its own branch the overlay docks to its placeholder; off it (swiped
        // to a sibling while editing) it always floats.
        let on_path = self.selected_leaf_id(roots, page_width).as_ref() == Some(&active);

        let win = window_h.as_f32();
        // The composer slot's document top — everything on the selected path
        // above it (independent of `page_scroll`). The docked caret-into-view
        // path adds the editor's content-local caret span to this to get the
        // caret's absolute document position and follows it with `page_scroll`.
        let page_slot_doc_top = self.placeholder_doc_top(roots, page_width, window_h);
        let chrome = Self::composer_chrome();
        let content = self.composer_content_h.borrow().as_f32();
        let half_pad = POST_PAD_Y.as_f32() / 2.0;
        // The top chrome's render split (see `composer_scroll_gap`): the thin
        // separator stays outside the scroll clip, the rest rides inside the
        // scroll content. Together they are exactly `chrome`, so all the
        // height/dock math below stays in terms of the total.
        let scroll_gap = composer_scroll_gap();

        // The floating bar height under the window's fraction + sizing mode
        // (the dynamic replacement for the old fixed 50% cap) — the same
        // helper the glide and the off-branch floating pad read, so the three
        // can't disagree on the bar.
        let float_bar_h = self.composer_float_bar_h(window_h);
        let float_top = win - float_bar_h;
        let slot_top = if on_path {
            Some(self.placeholder_doc_top(roots, page_width, window_h) + self.clamped_scroll_y())
        } else {
            None
        };
        let top_y = match slot_top {
            Some(s) => float_top.min(s + half_pad),
            None => float_top,
        };

        let overlayed = top_y >= float_top - 0.5;
        let docked = !overlayed;
        self.composer_overlayed.set(overlayed);

        let full_h = (content + chrome).max(win);
        let bar_h = composer_bar_h(
            float_bar_h,
            full_h,
            float_top,
            top_y,
            self.doc_reserve(),
            docked,
        );
        let body_h = (bar_h - chrome).max(0.0);
        // **The visible quad is not the bar.** `bar_h` above is the composer's
        // *virtual* geometry — the scroll viewport the ramp grows so a scrolled
        // composer's internal offset eases to its top by the time it docks. The
        // painted background quad is a separate, window-clipped rectangle:
        //
        // - at the **top**, when the draft's slot scrolls above the viewport
        //   `top_y` goes negative and a quad hanging above the window top shows
        //   its square mid-section in the corner notches (its own corners are
        //   off-screen, so per-element rounding can't help) — pin the quad's
        //   top at the window edge, hang the inner content at the virtual
        //   offset, and round the quad's top corners when it owns that edge;
        // - at the **bottom**, end the quad exactly at the window edge so its
        //   rounded bottom corners align with the window's (a quad clipped
        //   mid-body by the chrome frame would show square corners there).
        //
        // Clamping `bar_h` itself instead — which is what this used to do —
        // pinned the scroll viewport to the *visible* height, so `scroll_max`
        // never reached zero and a docked composer stayed scrolled off its own
        // top by `doc_reserve + rail`, cut off at the bottom by the same
        // amount. For a floating composer all of this is an identity
        // (`win - top_y == float_bar_h`, `content_shift == 0`).
        let bar_top = top_y.max(0.0);
        let content_shift = top_y - bar_top; // ≤ 0: inner content overhang
        let quad_h = (bar_h + content_shift).min(win - bar_top).max(0.0);
        let covers_top = bar_top <= 0.5;
        // **What the quad clipped, the footer must not lose.** `bar_h` is
        // virtual (see above), and mid-dock the ramp carries the bar's *bottom*
        // past the window edge: the bar's bottom is `top_y + bar_h`, which the
        // ramp drives from exactly `win` (at the float line) to `doc_reserve +
        // full_h` (fully docked) — always ≥ `win`, by up to a `doc_reserve`.
        // The `.min()` above clips the painted quad back to the window, and the
        // footnote rail — laid out at the end of the body's flow, i.e. at the
        // end of the *virtual* runway — went with the clipped part. Hence the
        // rail vanishing on a docked composer and reappearing only once the
        // page reached the very end, where the two bottoms coincide again.
        //
        // The rail is the bar's footer, so it belongs on the bar's *visible*
        // bottom edge. Taking the clipped tail out of the editor's runway floor
        // below moves the rail up by exactly what was cut — no more, so it
        // still lands flush on that edge with its own breath beneath it. When
        // the draft's text is genuinely taller than the runway the floor
        // doesn't bind, and the rail keeps following the text below the fold
        // (where the page scroll reaches it) rather than overlaying it.
        let clipped_tail = (bar_h + content_shift - quad_h).max(0.0);
        // The composer scrolls internally only when floating *and* its content
        // exceeds the visible bar — i.e. it's capped at the window's composer
        // fraction (or pinned under it, with Exact sizing).
        // When it's floating at its natural height (content fits, incl. empty /
        // one line) the editor's `min_height` fills the bar exactly, so there's
        // nothing to scroll; letting the wheel fall through to the page (below)
        // scrolls the conversation underneath instead of trapping it. When docked
        // the page owns the scroll regardless.
        let composer_scrollable = overlayed && (chrome + content > bar_h + 0.5);
        self.composer_scrollable.set(composer_scrollable);
        // The internal fold shadow appears once actual *text* is cut at the
        // fold — the first `scroll_gap` of internal scroll only consumes the
        // in-content spacer, and a shadow over that still-blank strip would
        // announce a cut that hasn't happened.
        let scrolled_down = self.composer_scroll.offset().y.as_f32() < -(scroll_gap + 0.5);

        // Both gutters take the scroll gap as a top margin: the h_flex below
        // pads only the separator, so without it the byline and the action
        // verbs would ride up by the gap — they must stay aligned with the
        // (unscrolled) first text line, exactly where a post's gutter sits.
        let mut byline =
            super::post::byline_gutter("You", theme.info, Some(super::post::DRAFT_BYLINE_OPACITY))
                .mt(px(scroll_gap));
        if overlayed {
            let home_fg = theme.muted_foreground;
            let home_fg_hover = theme.foreground;
            let home_bg_hover = theme.muted;
            byline = byline.child(
                div()
                    .id("space-draft-home")
                    .probe("space/composer/home", gpui::Role::Button, "See in context")
                    .mt_1()
                    .px_1()
                    .rounded_md()
                    .text_sm()
                    .text_color(home_fg)
                    .top_neg_2()
                    .cursor_pointer()
                    .text_align(gpui::TextAlign::Right)
                    .pr_neg_0p5()
                    .hover(move |s| s.text_color(home_fg_hover).bg(home_bg_hover))
                    .child("See in context")
                    .on_click(cx.listener(|this, _, window, cx| this.go_home(window, cx))),
            );
        }

        let mut body = div()
            .id("space-composer-body")
            .w(bw)
            // The body owns the chrome's in-content share: its frame reaches up
            // to the separator (the clip fold), and the first `scroll_gap` of
            // its content is the spacer below. Viewport and content both grow
            // by the same `scroll_gap`, so `scroll_max` — and with it the
            // glide, the dock ramp, and the caret math — is untouched.
            .h(px(body_h + scroll_gap))
            // Scroll tracking stays on **unconditionally**, even when the composer
            // owns no scroll session — the dock transition depends on it. A
            // scrolled floating composer's offset is walked to its top *before*
            // docking by the pre-dock glide (`glide_composer_toward_dock`, run
            // each render while the dock threshold is under the floating bar),
            // and once docked, `bar_h` (and thus `body_h`, the scroll viewport)
            // ramps up toward the full content height so gpui clamps any residual
            // offset to 0 — the backstop for paths the glide can't see. Keeping
            // `track_scroll` off while docked (an earlier version) abandoned the
            // offset the instant the composer docked and snapped the content.
            // A fit-height composer can't scroll regardless (`scroll_max == 0` once
            // the measuring canvas is pinned — see `record_height`), so always-on
            // reintroduces no phantom scroll.
            .overflow_y_scroll()
            .track_scroll(&self.composer_scroll)
            .on_scroll_wheel(cx.listener(|this, ev: &ScrollWheelEvent, window, cx| {
                if matches!(ev.touch_phase, TouchPhase::Started) {
                    this.scroll_owner = None;
                }
                let delta_y = ev.delta.pixel_delta(window.line_height()).y.as_f32();
                if delta_y == 0.0 {
                    return;
                }
                // The composer *owns* the wheel (internal scroll, page locked out)
                // only when floating with real overflow; otherwise the page does —
                // so a floating fit-height composer scrolls the conversation
                // underneath (letting it dock), and a scrolled composer nearing
                // its dock lets the page scroll while the pre-dock glide walks
                // its frozen offset to the top (`glide_composer_toward_dock`).
                let owner = *this
                    .scroll_owner
                    .get_or_insert(if this.composer_scrollable.get() {
                        ScrollOwner::Composer
                    } else {
                        ScrollOwner::Body
                    });
                match owner {
                    ScrollOwner::Composer => cx.stop_propagation(),
                    ScrollOwner::Body => {
                        let off = this.composer_scroll.offset();
                        this.composer_scroll
                            .set_offset(point(off.x, px(this.composer_prev_off_y)));
                    }
                }
            }))
            // The top chrome's in-content share (`composer_scroll_gap`): an
            // in-flow spacer, so unscrolled text sits exactly `chrome` below
            // the bar top while a scrolled composer carries the gap away with
            // the content and clips at the thin separator. A child, not
            // padding on this div — padding joins gpui's scrollable extent
            // and would let a fit-height composer scroll by the gap.
            .child(div().h(px(scroll_gap)))
            // "Notes editor" affordance owned by the editor itself: `min_height`
            // grows it to fill the docked runway, and a click in the blank tail
            // below the text resolves to document end inside the editor (see
            // `MarkdownEditor::min_height` / `offset_for_position`). No overlay
            // listener — a click on the text stays a normal caret placement.
            // `body_h` already carries the half-pad breath (folded into
            // `content` by `record_height` below), so the editor's excess starts
            // right under the last line — no dead strip.
            .child({
                let menu_editor = editor.clone();
                MarkdownEditor::new(&editor)
                    // The footnote rail below shares the bar's height, so the
                    // editor's runway floor is the bar minus the rail — the
                    // rows are single-line by construction (`truncate`), so
                    // the reservation is exact rather than a guess — minus
                    // whatever the window clipped off the bar's virtual bottom
                    // (`clipped_tail`), so the rail lands on the *visible*
                    // edge instead of below it.
                    .style(prose_style(cx))
                    .min_height(px((body_h - draft_rail_h - clipped_tail).max(0.0)))
                    // The editable context menu (Cut / Copy / Paste / Select All).
                    .on_context_menu(cx.listener(
                        move |this, at: &gpui::Point<gpui::Pixels>, _, cx| {
                            this.open_context_menu(
                                *at,
                                menu_editor.clone(),
                                ContextTarget::Editable,
                                cx,
                            );
                        },
                    ))
            })
            // The draft's pending quotes, as footnotes — the same rail a
            // posted exchange carries, right where it will be once posted.
            // The draft's pending quotes, as footnotes, bracketed by the two
            // zero-height flow marks that measure exactly how much room they
            // take (see `references::flow_mark`). With no rail the marks
            // coincide and the reservation is zero — no special case.
            .child(super::references::flow_mark(self.composer_rail_top.clone()))
            .children(self.render_draft_footnotes(&draft_id, true, cx))
            .child(super::references::flow_mark(
                self.composer_rail_bottom.clone(),
            ))
            // Measure the composer body's *natural* content height — the
            // editor's own laid-out text (decoupled from the `min_height`
            // floor, so growing the editor to fill the runway doesn't feed
            // back into the height that sizes it) plus its tail: the rail's
            // measured occupancy, or a bare breath when there is no rail. The
            // rail carries the breath as its own padding, so the two are a
            // `max`, never a sum — the bar reserves exactly what the body
            // draws, in both rail states.
            .child(record_height(
                self.composer_content_h.clone(),
                self.composer_rail_top.clone(),
                self.composer_rail_bottom.clone(),
                editor.downgrade(),
                cx.entity().downgrade(),
                bottom_breath(),
            ))
            // Scroll the caret into view after an edit. Runs in the paint phase
            // (a later sibling than the editor, so this frame's post-edit
            // `last_blocks`/`last_bounds` are fresh — the same ordering
            // `record_height` relies on), gated on the `composer_caret_scroll_pending`
            // flag so it fires once per edit and never fights a manual scroll.
            // Branches on the composer configuration: a floating-with-overflow
            // composer owns `composer_scroll`; a **docked** composer (incl. the
            // blank ⌘N notebook) has no internal scroll, so bringing its caret
            // onto the window is a `page_scroll` concern — `page_slot_doc_top` is
            // the slot's document top (independent of `page_scroll`, so following
            // it converges without oscillation).
            .child(caret_into_view(
                cx.entity().downgrade(),
                editor.downgrade(),
                body_h,
                docked,
                page_slot_doc_top,
                win,
            ));
        body.style().restrict_scroll_to_axis = Some(true);

        // The composer bar is the window's bottom-most opaque surface (the
        // frame cannot clip children — see chrome.rs): round its bottom
        // corners to match the window. Floating, its bottom sits exactly on
        // the window bottom; docked, it either ends at the content bottom
        // (short branches) or extends past the visible edge, where the
        // rounding is simply out of view.
        let mut composer = crate::chrome::round_bottom_client_corners(div(), window)
            .when(covers_top, |d| {
                crate::chrome::round_top_client_corners(d, window)
            })
            .id("space-composer")
            .probe_value(
                "space/composer",
                gpui::Role::TextInput,
                "Message composer",
                self.composer_aria_value(&draft_id, &editor, window, cx),
            )
            .absolute()
            .left_0()
            .right_0()
            .top(px(bar_top))
            .h(px(quad_h))
            .bg(theme.background)
            .on_key_down(cx.listener(|this, ev: &KeyDownEvent, window, cx| {
                if ev.keystroke.key == "escape" {
                    // An open menu absorbs the first Escape; the next
                    // deactivates the draft (deleting it if an empty fork) and
                    // moves focus off the editor to the view root, so a kept
                    // draft reads as exited (no stray cursor) until it's
                    // clicked back into. The context menu is only *yielded* to
                    // here — the view root closes it (see
                    // `context_menu_absorbs_escape`).
                    if this.context_menu_absorbs_escape() {
                        // absorbed
                    } else if this.band_menu.is_some() {
                        this.band_menu = None;
                        cx.notify();
                    } else {
                        this.deactivate_active_draft(cx);
                        window.focus(&this.focus_handle, cx);
                    }
                }
            }))
            .child(
                h_flex()
                    .w(page_width)
                    .mt(px(content_shift))
                    .h(px(bar_h))
                    // Only the separator slice of the top chrome lives outside
                    // the columns; the rest is the body's in-content spacer
                    // (and the gutters' compensating top margins), so the
                    // scroll fold sits a thin, pane-separator band below the
                    // bar's edge while nothing else moves.
                    .pt(px(COMPOSER_SEPARATOR_H))
                    .justify_center()
                    .items_start()
                    .gap(GUTTER_GAP)
                    .child(byline)
                    .child(body)
                    .child(self.render_composer_actions(&editor, cx).mt(px(scroll_gap))),
            );
        if scrolled_down {
            composer = composer.child(
                div()
                    .absolute()
                    .top(px(COMPOSER_SEPARATOR_H))
                    .left_0()
                    .right_0()
                    .h(px(8.))
                    .bg(linear_gradient(
                        180.,
                        linear_color_stop(hsla(0., 0., 0., 0.08), 0.),
                        linear_color_stop(hsla(0., 0., 0., 0.), 1.),
                    )),
            );
        }
        if overlayed {
            // The separator doubles as a **resize handle** while floating: a
            // centered dot marks it, the band carries the vertical-resize
            // cursor, and a drag writes the window's composer fraction
            // directly — switching the sizing to Exact on grab so the bar can
            // be sized in excess of its content (`start_composer_resize`).
            // Docked, the bar is page geometry and offers no handle.
            let handle_fg = theme.muted_foreground.opacity(0.5);
            let grab_bar_h = float_bar_h;
            composer = composer.child(
                div()
                    .id("space-composer-resize")
                    .probe(
                        "space/composer/resize",
                        gpui::Role::Slider,
                        "Resize composer",
                    )
                    .absolute()
                    .top_0()
                    .left_0()
                    .right_0()
                    .h(px(COMPOSER_RESIZE_HIT_H))
                    .cursor(gpui::CursorStyle::ResizeUpDown)
                    .on_mouse_down(
                        gpui::MouseButton::Left,
                        cx.listener(move |this, ev: &gpui::MouseDownEvent, _, cx| {
                            cx.stop_propagation();
                            this.start_composer_resize(ev.position.y.as_f32(), grab_bar_h, win, cx);
                        }),
                    )
                    .child(
                        h_flex()
                            .w_full()
                            .h(px(COMPOSER_SEPARATOR_H))
                            .justify_center()
                            .child(
                                div()
                                    .size(px(COMPOSER_RESIZE_DOT))
                                    .rounded_full()
                                    .bg(handle_fg),
                            ),
                    ),
            );
        }
        // Window-global tracking for an in-flight resize drag (the minimap
        // pattern): `on_mouse_event` listeners are cleared each frame, so they
        // are re-registered every frame; they no-op unless a drag is live, and
        // registering unconditionally avoids a first-move gap. This is what
        // keeps the drag following after the cursor leaves the thin strip —
        // and what keeps it live if the bar docks mid-drag (the handle
        // unmounts; the drag doesn't).
        {
            let weak = cx.entity().downgrade();
            composer = composer.child(
                gpui::canvas(
                    |_, _, _| {},
                    move |_, _, window, _cx| {
                        let move_weak = weak.clone();
                        window.on_mouse_event(
                            move |ev: &gpui::MouseMoveEvent, _phase, _window, cx| {
                                let Some(this) = move_weak.upgrade() else {
                                    return;
                                };
                                this.update(cx, |this, cx| {
                                    if this.composer_resize.is_none() {
                                        return;
                                    }
                                    if !ev.dragging() {
                                        // Button released without a delivered up event.
                                        this.end_composer_resize(cx);
                                        return;
                                    }
                                    cx.stop_propagation();
                                    this.update_composer_resize(ev.position.y.as_f32(), win, cx);
                                });
                            },
                        );
                        let up_weak = weak.clone();
                        window.on_mouse_event(
                            move |_: &gpui::MouseUpEvent, _phase, _window, cx| {
                                let Some(this) = up_weak.upgrade() else {
                                    return;
                                };
                                this.update(cx, |this, cx| this.end_composer_resize(cx));
                            },
                        );
                    },
                )
                .absolute()
                .top_0()
                .left_0()
                .size_full(),
            );
        }
        if overlayed {
            composer = composer.shadow(vec![
                BoxShadow::new(px(0.), SHADOW_OFFSET_Y, theme.foreground.opacity(0.18))
                    .blur_radius(SHADOW_BLUR),
            ]);
        }
        // Float the whole composer as its own stacking layer.
        //
        // gpui assigns each primitive a draw `order` from a `BoundsTree` (order =
        // 1 + max order of already-painted primitives whose *registered* bounds it
        // overlaps), then batches by `(order, primitive_kind)`. A drop shadow's
        // registered bounds are the element rect grown by its *spread* only — the
        // blur reach is **not** included (`Window::paint_drop_shadows`). So in the
        // ~20px window around the dock threshold, where the shadow's blur visually
        // spills over the final separator band but the composer rect hasn't reached
        // it yet, an inline shadow and the band land in disjoint BoundsTree chains
        // and the band (bumped by the high-order posts above it) sorts on top — the
        // shadow renders *behind* the separator, then "jumps" in front once the
        // rects overlap.
        //
        // `layered` paints the composer inside a single `Window::paint_layer` whose
        // bounds are dilated upward to cover the shadow's blur reach. Every composer
        // primitive then shares one layer order, computed from bounds that *do*
        // overlap the band — so the entire composer, shadow included, sits above the
        // page for the whole transition (internal order is preserved: within the
        // shared order, batching still draws Shadow-kind under Quad-kind under text).
        // No `deferred` wrapper is needed: the composer is a later sibling than the
        // scroll subtree, so it already paints after every post/band and its layer
        // order lands above them; the minimap, a still-later sibling, paints above
        // the composer in turn. Staying in the normal paint pass also keeps the
        // composer *below* late overlays such as the gpui dev inspector.
        let reach = px(SHADOW_BLUR.as_f32() - SHADOW_OFFSET_Y.as_f32() + 3.0);
        layered(composer, reach).into_any_element()
    }

    // -- The action gutter ---------------------------------------------------

    /// The active composer's action gutter: **Post** (⌘↩ — save the thought;
    /// the space's participants decide who responds) and, while ⌥ is held,
    /// **Post quietly** (⌘⇧↩ — save without notifying anyone) plus the
    /// keyboard hints. Renders as an empty (reserved) gutter until the draft
    /// has content or ⌥ is held — the affordance appears the moment it's
    /// actionable, and the blank page stays sacred. The composer carries no
    /// model chrome: who answers (and with what model) is Participants
    /// configuration, and explicit asks live on the separators.
    pub(crate) fn render_composer_actions(
        &self,
        editor: &gpui::Entity<MarkdownEditorState>,
        cx: &Context<Self>,
    ) -> gpui::Div {
        let theme = cx.theme();
        let alt = self.window_input.read(cx).alt_held();
        let revealed = !editor.read(cx).is_empty() || alt;

        let mut col = super::post::action_gutter().gap_0p5();
        if !revealed {
            return col;
        }

        let fg = theme.muted_foreground;
        let fg_hover = theme.foreground;
        let bg_hover = theme.muted;
        let hint_fg = theme.muted_foreground.opacity(0.7);

        // Post — save the thought; notify policies drive who responds.
        col = col.child(
            h_flex()
                .id("space-post")
                .probe("space/composer/post", gpui::Role::Button, "Post")
                .px_1()
                .ml_neg_1()
                .rounded_md()
                .cursor_pointer()
                .items_baseline()
                .gap_1p5()
                .text_sm()
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .text_color(fg)
                .hover(move |s| s.text_color(fg_hover).bg(bg_hover))
                .child("Post")
                .when(alt, |d| d.child(kbd_hint("⌘↩", hint_fg)))
                .on_click(cx.listener(|this, _, window, cx| this.submit(&Send, window, cx))),
        );

        // Post quietly — save without notifying anyone. ⌥-revealed: present
        // when reached for, invisible in the common case.
        if alt {
            col = col.child(
                h_flex()
                    .id("space-post-quiet")
                    .probe(
                        "space/composer/post-quiet",
                        gpui::Role::Button,
                        "Post quietly — notify no one",
                    )
                    .px_1()
                    .ml_neg_1()
                    .rounded_md()
                    .cursor_pointer()
                    .items_baseline()
                    .gap_1p5()
                    .text_sm()
                    .text_color(fg)
                    .hover(move |s| s.text_color(fg_hover).bg(bg_hover))
                    .child("Post quietly")
                    .child(kbd_hint("⇧⌘↩", hint_fg))
                    .on_click(
                        cx.listener(|this, _, window, cx| this.post_only(&PostOnly, window, cx)),
                    ),
            );
        }

        col
    }

    /// The active composer's measured geometry: `(reserved, rail, text)` — the
    /// natural content height the bar sizes itself to
    /// ([`composer_content_h`](SpaceView::composer_content_h)), the footnote
    /// rail's measured occupancy (the flow-mark span, `0.0` with no rail), and
    /// the editor's own laid-out text height. `None` with no active draft.
    ///
    /// The invariant tests read off these three: `reserved == text + rail` with
    /// a rail (which carries the bottom breath as its own padding) and
    /// `reserved == text + bottom_breath()` without one — reserved is what the
    /// body draws, and the breath is counted exactly once either way.
    #[doc(hidden)]
    pub fn composer_geometry_for_test(&self, cx: &gpui::App) -> Option<(f32, f32, f32)> {
        let active = self.active_draft.as_ref()?;
        let draft = self.drafts.iter().find(|d| &d.id == active)?;
        let reserved = self.composer_content_h.borrow().as_f32();
        let rail = (self.composer_rail_bottom.get() - self.composer_rail_top.get()).max(0.0);
        let text = draft.editor.read(cx).content_height().as_f32();
        Some((reserved, rail, text))
    }

    /// Dock the active draft at its "home" (the composer's own "See in
    /// context" verb): slot top around 40% of the window, which lands the bar
    /// in the *middle* of the dock ramp — the configuration whose geometry
    /// differs most from both floating and end-of-document docked.
    #[doc(hidden)]
    pub fn dock_active_draft_for_test(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.go_home(window, cx);
    }

    /// Whether the composer floated (vs docked into its slot) on the last
    /// rendered frame — the precondition tests state before asserting anything
    /// about docked geometry.
    #[doc(hidden)]
    pub fn composer_overlayed_for_test(&self) -> bool {
        self.composer_overlayed.get()
    }

    /// The footnote rail's painted bottom edge in **window** coordinates (the
    /// lower flow mark's own position, so it includes the rail's bottom
    /// breath). Tests assert it stays inside the window in every composer
    /// configuration — the rail is the bar's footer, not something that hangs
    /// off the end of the virtual dock runway.
    #[doc(hidden)]
    pub fn composer_rail_bottom_for_test(&self) -> f32 {
        self.composer_rail_bottom.get()
    }
}

/// A small muted keyboard hint beside a verb (shown while ⌥ is held).
fn kbd_hint(text: &'static str, color: gpui::Hsla) -> gpui::Div {
    div().text_xs().text_color(color).child(text)
}

/// The composer bar's **bottom breath** — the mirror of the `half_pad` chrome
/// above the byline, so the last thing in the bar never sits flush against the
/// window edge.
///
/// It is **drawn exactly once**, by whichever element ends the composer body:
/// the footnote rail's own bottom padding when a rail is present (see
/// [`SpaceView::render_draft_footnotes`](super::SpaceView::render_draft_footnotes)),
/// and the editor's runway when it isn't — the editor's `min_height` fills the
/// bar, so with no rail the breath is live, clickable notes space rather than a
/// dead strip. One function, both call sites, so the two can't drift.
pub fn bottom_breath() -> f32 {
    POST_PAD_Y.as_f32() / 2.0
}

/// The breath kept **below the footnote rail** — a post's full bottom pad
/// rather than the bar's half-pad [`bottom_breath`].
///
/// The rail is a ruled, text-bearing footer, not the tail of the writing
/// surface: at the half-pad the last footnote row read as crowded against the
/// window edge, where the same half-pad under a bare runway reads as open notes
/// space. So the rail keeps the page's own vertical rhythm (`POST_PAD_Y`, what
/// a post pads with) beneath it. It rides *inside* the span the two flow marks
/// measure, so `record_height`'s `max(rail, breath)` still counts the bar's
/// bottom breath exactly once.
pub fn rail_breath() -> f32 {
    POST_PAD_Y.as_f32()
}

/// The slice of the composer bar's top chrome that stays **outside** the
/// scroll clip — the thin band between the bar's top edge (where the external
/// drop shadow is cast) and the scroll fold (where scrolled content clips and
/// the internal shadow paints). Deliberately thinner than the full top chrome
/// ([`SpaceView::composer_chrome`], half the inter-post gap): on a scrolled
/// floating composer this band is dead space, and at the full chrome height it
/// read as a blank strip where a familiar pane separator was expected.
pub(crate) const COMPOSER_SEPARATOR_H: f32 = 12.0;

/// The remainder of the bar's top chrome, folded **inside** the scroll content
/// as an in-flow spacer above the editor: `composer_chrome() −`
/// [`COMPOSER_SEPARATOR_H`]. Unscrolled it sits right under the separator, so
/// the text starts exactly `composer_chrome()` below the bar top — the same
/// place as before the split (and the docked editor keeps its post-matching
/// `2·half_pad` slot offset, see `caret_into_view`'s docked arm) — and it
/// scrolls away with the content, so a scrolled composer's dead band is only
/// the separator. The split is a **render concern only**: every height / dock
/// / runway computation keeps using the `composer_chrome()` total, so nothing
/// there can drift. It is a spacer child rather than container padding because
/// gpui adds padding to the scrollable extent (`scroll_max = content + padding
/// − bounds`, `div.rs`), which would mint a `scroll_gap` phantom scroll on a
/// fit-height composer; an in-flow child is covered by the origin-pinned
/// measuring canvas and leaves `scroll_max` exactly as it was.
pub(crate) fn composer_scroll_gap() -> f32 {
    (SpaceView::composer_chrome() - COMPOSER_SEPARATOR_H).max(0.0)
}

/// How the window's floating-composer fraction
/// ([`SpaceView::composer_fraction`], a ratio of the window height) applies to
/// the floating bar.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) enum ComposerSizing {
    /// The fraction is a **cap**: the bar floats at its natural content height
    /// up to `fraction · window`, then scrolls internally — the resting
    /// behavior every activation starts from (and, at the default fraction,
    /// exactly the pre-configurability behavior).
    #[default]
    Max,
    /// The fraction **is** the bar: while floating, the composer is exactly
    /// `fraction · window` tall regardless of its content — short content gets
    /// blank runway below, like the docked notebook — and it enters/leaves the
    /// floating state on that height rather than on what it happens to
    /// contain. Entered by grabbing the separator's resize handle (the whole
    /// point of the drag is sizing the bar in excess of its content); reverted
    /// to [`Self::Max`] whenever the composer deactivates or a different draft
    /// is selected ([`SpaceView::reset_composer_sizing`]).
    Exact,
}

/// An in-flight separator-handle resize drag: the pointer's window-space `y`
/// and the floating bar's height at the moment of grab. Motion applies as a
/// **delta** from these (`bar = start_bar_h + (start_y − pointer_y)`) rather
/// than as absolute window coordinates, so the math is immune to the Linux CSD
/// content insets and can never jump under the grab.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ComposerResizeDrag {
    pub(crate) start_y: f32,
    pub(crate) start_bar_h: f32,
}

/// Bounds for the resize drag's fraction: enough bar left at the bottom for a
/// few lines, enough conversation left visible above it. The *stored* fraction
/// may rest below the minimum (grabbing a short Max bar snapshots its honest
/// current ratio so nothing jumps); the bounds clamp only where the user
/// drags to.
pub(crate) const COMPOSER_FRACTION_MIN: f32 = 0.1;
pub(crate) const COMPOSER_FRACTION_MAX: f32 = 0.85;

/// The resize handle's hit band: the separator plus a few px into the body's
/// blank in-content spacer below it (the chrome.rs resize-band precedent — a
/// thin strip needs a forgiving target, and the overlap covers nothing
/// clickable), and the little dot that marks the separator as a handle.
const COMPOSER_RESIZE_HIT_H: f32 = COMPOSER_SEPARATOR_H + 4.0;
const COMPOSER_RESIZE_DOT: f32 = 4.0;

/// The pure sizing core behind [`SpaceView::composer_float_bar_h`]: the
/// floating bar's height given the natural content height (`chrome +
/// content`), the window fraction, and the sizing mode. Pure and unit-tested.
pub(crate) fn float_bar_height(
    natural: f32,
    fraction: f32,
    win: f32,
    sizing: ComposerSizing,
) -> f32 {
    let cap = fraction * win;
    match sizing {
        ComposerSizing::Max => natural.min(cap),
        ComposerSizing::Exact => cap,
    }
}

use crate::probe::Probe as _;
use std::cell::RefCell;
use std::rc::Rc;

/// Record the composer body's natural content height into `cell`, scheduling a
/// catch-up frame when it changes so the bar resizes as the content settles.
///
/// The height is the sum of what the body actually draws: the editor's own
/// laid-out text (`content_height`, read in the **paint** phase — a later
/// sibling, so the editor's blocks have painted and updated their bounds this
/// frame — and decoupled from the editor's `min_height`, so growing the editor
/// to fill the runway doesn't feed back into the height that sizes the runway),
/// plus the **tail** below that text: the footnote rail's measured height
/// (`rail`, the distance between the two flow marks bracketing it — see
/// [`references::flow_mark`](super::references::flow_mark)), or a bare `breath`
/// when there is no rail.
///
/// **The tail is a `max`, not a sum, because the breath is drawn once.** A
/// populated rail already carries it as its own bottom padding — inside the
/// bracketed span — so adding [`bottom_breath`] on top would reserve it twice
/// and inflate the bar by a pad-height (visible as a gap between the last line
/// of text and the footnote rule, since the editor's `min_height` floor —
/// `body_h − rail` — grows to swallow the surplus). With no rail the marks
/// coincide, `max` degenerates to the breath, and the editor's runway draws it
/// as live notes space. No branch on "is the rail empty", and either way the
/// bar reserves exactly what the body draws. Nothing here is derived from a row
/// count or a styling constant, so the reservation cannot drift from the
/// rendering.
///
/// **Pinned to the origin** (`top_0().left_0()`): it's a zero-visual measuring
/// probe, but as an `absolute` child with no inset taffy places it at its
/// *static* position — after the editor in flow — which folds its own height
/// into the scroll container's `content_size` (`div.rs` unions **all** child
/// bounds, absolute included). That inflated the composer body's scroll range by
/// a full `body_h`, letting a scrolled composer push its editor entirely out of
/// view. Pinning it at `(0,0)` keeps it inside the editor's extent so it adds
/// nothing to the scrollable content.
fn record_height(
    cell: Rc<RefCell<gpui::Pixels>>,
    rail_top: Rc<std::cell::Cell<f32>>,
    rail_bottom: Rc<std::cell::Cell<f32>>,
    editor: gpui::WeakEntity<MarkdownEditorState>,
    view: gpui::WeakEntity<SpaceView>,
    breath: f32,
) -> impl IntoElement {
    gpui::canvas(
        |_, _, _| {},
        move |_, _, window, cx| {
            let Some(editor) = editor.upgrade() else {
                return;
            };
            let rail = (rail_bottom.get() - rail_top.get()).max(0.0);
            let h = editor.read(cx).content_height().as_f32() + rail.max(breath);
            if (cell.borrow().as_f32() - h).abs() > 0.5 {
                *cell.borrow_mut() = gpui::px(h);
                let view = view.clone();
                window.on_next_frame(move |_, cx| {
                    view.update(cx, |_, cx| cx.notify()).ok();
                });
            }
        },
    )
    .absolute()
    .top_0()
    .left_0()
    .size_full()
}

/// The composer bar's **virtual** height: `float_bar_h` while floating, and,
/// once docked, a ramp from there up to `full_h` (the whole content, at least a
/// window) as the slot rises from the float line toward the document's top
/// reserve.
///
/// This is the *dock ramp*. It gives the docked composer its real geometry —
/// `body_h = bar_h − chrome` is the scroll viewport, so as the bar grows
/// `scroll_max = content − body_h` shrinks to zero and the whole content is
/// laid out in flow — and it is the **backstop** for the internal scroll: a
/// scrolled floating composer's offset is normally walked to its top *before*
/// docking (the pre-dock glide, [`SpaceView::glide_composer_toward_dock`]),
/// but any residual the glide couldn't see (a mid-zone resize, a document
/// reflow) is clamped out here as the viewport grows. Reaching `full_h`
/// is what makes `scroll_max` actually reach **zero**; anything that clamps
/// this to the *visible* height (the window bottom, say) leaves a permanent
/// residual offset and the composer docks scrolled off its own top and cut off
/// at the bottom. Window clipping belongs to the painted quad, not here.
///
/// Pure so the ramp is testable without a render.
fn composer_bar_h(
    float_bar_h: f32,
    full_h: f32,
    float_top: f32,
    top_y: f32,
    doc_reserve: f32,
    docked: bool,
) -> f32 {
    if !docked {
        return float_bar_h;
    }
    let denom = (float_top - doc_reserve).max(1.0);
    let progress = ((float_top - top_y) / denom).clamp(0.0, 1.0);
    float_bar_h + progress * (full_h - float_bar_h)
}

/// The dock-approach runway: how much page travel remains before the floating
/// composer docks, saturated at the **approach zone**. `dock_top` is where the
/// bar's top would sit if it docked to its slot this frame (`slot_top +
/// half_pad`, window coordinates), `float_top` the float line (the floating
/// bar's top), `win` the window bottom. The zone is `[float_top, win]` — the
/// dock threshold sitting under the floating bar — so the runway saturates at
/// the bar's height (`win − float_top`): any page position at or below the
/// zone reads as a full runway, which is what gates the glide to the zone
/// (equal saturated runways step nothing in [`approach_glide_offset`]).
/// Pure so the zone gating is unit-testable without a render.
fn dock_runway(dock_top: f32, float_top: f32, win: f32) -> f32 {
    (dock_top - float_top).clamp(0.0, (win - float_top).max(0.0))
}

/// One step of the pre-dock glide: scale the composer's internal scroll `off`
/// (`<= 0`) by the fraction of dock runway remaining — `r_prev` last frame's
/// runway, `r_now` this frame's (both from [`dock_runway`]). Steps only on a
/// *descent* (`r_now < r_prev`); stasis and retreat return `off` unchanged
/// (the glide never reverses — scrolling away from the dock leaves the
/// composer's content where the reader put it). Consecutive descents
/// telescope (`off·(r₁/r₀)·(r₂/r₁) = off·(r₂/r₀)`), so the glide is
/// path-independent over any monotone descent — stop, restart, or jump
/// anywhere in the zone and the offset at a given runway is the same — and a
/// step landing on (or past) the threshold multiplies by zero: exactly zero
/// at dock, by construction rather than by accumulation. Pure and
/// unit-tested.
fn approach_glide_offset(off: f32, r_prev: f32, r_now: f32) -> f32 {
    if off >= 0.0 || r_prev <= 0.0 || r_now >= r_prev {
        return off;
    }
    off * (r_now / r_prev)
}

/// Breathing room kept between the caret and the composer viewport edge when
/// scrolling it into view, so the caret never lands flush against the fold.
const CARET_SCROLL_MARGIN: f32 = 8.0;

/// Given the caret's content-local vertical span (`caret_top`/`caret_bot`,
/// relative to the scroll content's top), the scroll `viewport_h`, the current
/// scroll offset `cur_off` (`<= 0`, more-negative = scrolled down), and the
/// valid scroll depth `scroll_max` (`>= 0`, `content − viewport`), return the
/// new scroll offset that brings the caret inside the viewport with
/// [`CARET_SCROLL_MARGIN`] of breath. A no-op (returns `cur_off`) when the
/// caret is already comfortably visible or when `scroll_max == 0` (a fit-height
/// composer can't scroll — the phantom-scroll invariant). Pure so the geometry
/// is unit-testable without a real render.
fn caret_scroll_offset(
    caret_top: f32,
    caret_bot: f32,
    viewport_h: f32,
    cur_off: f32,
    scroll_max: f32,
    margin: f32,
) -> f32 {
    // Scroll-space top currently shown at the viewport's top edge.
    let view_top = -cur_off;
    let mut new_top = view_top;
    if caret_top - margin < view_top {
        // Caret is above the fold — reveal it near the top.
        new_top = caret_top - margin;
    } else if caret_bot + margin > view_top + viewport_h {
        // Caret is below the fold — reveal it near the bottom.
        new_top = caret_bot + margin - viewport_h;
    }
    new_top = new_top.clamp(0.0, scroll_max.max(0.0));
    -new_top
}

/// A zero-visual paint probe that, when [`SpaceView::composer_caret_scroll_pending`]
/// is armed (an edit just landed), scrolls the caret into view. Reads the
/// editor's caret span + natural height in the **paint** phase — as a later
/// sibling of the editor it sees this frame's post-edit layout — computes the
/// target offset with [`caret_scroll_offset`], and writes it to the scroll
/// handle that actually governs the caret's visibility, chosen by the composer's
/// configuration this frame:
///
/// - **Floating with overflow** (`composer_scrollable`): the composer owns
///   `composer_scroll`, so the caret is scrolled within the composer's own
///   viewport (`body_h`). A fit-height composer has `scroll_max == 0`, so it's
///   inherently a no-op there. `body_h` is the **editor's share** of the
///   viewport — the body's real frame is `body_h + composer_scroll_gap()`,
///   with the gap's spacer above the editor — and the caret span is
///   editor-local, so for the (common) below-the-fold reveal the gap shifts
///   the caret's content position and the viewport bottom by the same amount
///   and cancels exactly; the above-the-fold reveal just gains the gap on top
///   of [`CARET_SCROLL_MARGIN`] (over-reveals, never under).
/// - **Docked** (`!composer_overlayed` — incl. the blank ⌘N notebook, which
///   expands and grows below the window): the composer has no internal scroll
///   and sits at its slot, so bringing the caret onto the *window* is a
///   `page_scroll` concern. The caret's document position is
///   `page_slot_doc_top + caret_content_y` — both terms independent of
///   `page_scroll` — so following it with `page_scroll` (viewport = `window_h`,
///   `scroll_max = -scroll_min_y`) converges immediately (no oscillation as the
///   dock ramp smooths `bar_h`).
/// - **Floating at natural height** (overlayed but fits): the caret is already
///   within the visible bar, which floats at the window bottom rather than at
///   its slot, so neither scroll applies — just consume the flag.
///
/// One-shot per edit: it clears the flag, so a subsequent manual scroll isn't
/// yanked back.
fn caret_into_view(
    view: gpui::WeakEntity<SpaceView>,
    editor: gpui::WeakEntity<MarkdownEditorState>,
    body_h: f32,
    docked: bool,
    page_slot_doc_top: f32,
    window_h: f32,
) -> impl IntoElement {
    gpui::canvas(
        |_, _, _| {},
        move |_, _, window, cx| {
            let (Some(view), Some(editor)) = (view.upgrade(), editor.upgrade()) else {
                return;
            };
            if !view.read(cx).composer_caret_scroll_pending.get() {
                return;
            }
            let ed = editor.read(cx);
            let Some((top, bot)) = ed.caret_content_y() else {
                // No layout yet (or caret off any line) — leave the flag armed
                // so the next paint with a fresh layout can act.
                return;
            };
            let natural = ed.content_height().as_f32();
            let caret_top = top.as_f32();
            let caret_bot = bot.as_f32();
            // Docked vs floating is the **render's own decision this frame**,
            // passed in alongside `body_h` (frame-consistent by construction) —
            // not the `composer_overlayed`/`composer_scrollable` cells, which
            // derive from the frame-behind `composer_content_h` and read
            // inconsistently under a repaint. An earlier version re-derived it
            // here as `body_h > COMPOSER_MAX_FRACTION · window` — a proxy that
            // (a) only held while the float cap was a fixed constant, which the
            // window-local fraction retired, and (b) silently classified the
            // whole lower dock ramp as "floating", scrolling a docked bar's
            // internal offset instead of the page. When docked, the page owns
            // the caret's window visibility; when floating, `natural - body_h`
            // (fresh measures) separates a real overflow (scroll the composer's
            // own bar) from a fit-height bar (caret already visible — no-op;
            // under Exact sizing that includes content shorter than the pinned
            // bar, whose runway shows the caret by construction).
            let composer_scroll_max = (natural - body_h).max(0.0);
            let changed = if !docked && composer_scroll_max > 0.5 {
                // Floating-with-overflow: scroll the composer's own viewport.
                let scroll_max = composer_scroll_max;
                view.update(cx, |this, _cx| {
                    this.composer_caret_scroll_pending.set(false);
                    let cur = this.composer_scroll.offset().y.as_f32();
                    let next = caret_scroll_offset(
                        caret_top,
                        caret_bot,
                        body_h,
                        cur,
                        scroll_max,
                        CARET_SCROLL_MARGIN,
                    );
                    if (next - cur).abs() > 0.5 {
                        let off = this.composer_scroll.offset();
                        this.composer_scroll.set_offset(point(off.x, px(next)));
                        // Keep the wheel handler's frozen-offset bookkeeping in
                        // sync so a following `ScrollOwner::Body` wheel doesn't
                        // snap the composer back to its pre-edit offset.
                        this.composer_prev_off_y = next;
                        true
                    } else {
                        false
                    }
                })
            } else if docked {
                // Docked (incl. blank ⌘N): follow the caret with the page. The
                // caret's document position is the slot's document top, PLUS the
                // editor content's offset within the slot, plus its content-local
                // span; all `page_scroll`-independent, so this converges.
                // `scroll_max = -scroll_min_y` (the page's valid depth,
                // set each frame in `render`). The page wheel handler
                // (`ScrollOwner::Body`) restores no frozen offset, so a following
                // wheel won't fight this programmatic scroll — no bookkeeping.
                //
                // **The editor-top offset (`POST_PAD_Y`).** The composer's editor
                // body does not begin at `page_slot_doc_top` (the slot block top):
                // the docked composer's `top_y` sits `half_pad` below the slot top,
                // and the bar's top chrome — the separator on the inner h_flex
                // plus the body's in-content spacer, together `composer_chrome()
                // = half_pad` (see `composer_scroll_gap`; docked, the internal
                // scroll is zero so the spacer is fully in place) — precedes the
                // editor, so the editor content (where `caret_content_y` measures
                // from y=0) starts `2·half_pad = POST_PAD_Y` below the slot top
                // (exactly aligning with where a post's body sits under its
                // `POST_PAD_Y` top pad). Omitting this term put every docked caret
                // target one pad-height too high — scrolling down never fully
                // revealed the line, scrolling up over-revealed it.
                let editor_top_offset = POST_PAD_Y.as_f32();
                let caret_doc_top = page_slot_doc_top + editor_top_offset + caret_top;
                let caret_doc_bot = page_slot_doc_top + editor_top_offset + caret_bot;
                // The page's scroll depth, computed from the editor's **fresh**
                // content height rather than `scroll_min_y`. `scroll_min_y` is
                // derived (in `render`) from `composer_content_h`, which
                // `record_height` records a frame *behind* — so on the edit frame
                // it can still reflect the pre-edit height and clamp the caret
                // target back to the current offset. The docked runway is
                // `max(window, chrome + content + half_pad)` and the document ends
                // at `page_slot_doc_top + runway` (the slot is the on-path leaf,
                // no trailing band / floating pad), so this reproduces exactly the
                // `scroll_min_y` the *next* frame will settle to — letting the
                // scroll land in one frame, the way the floating branch uses the
                // fresh `natural` height. See `runway_height` / `placeholder_doc_top`.
                let half_pad = POST_PAD_Y.as_f32() / 2.0;
                let runway = window_h.max(SpaceView::composer_chrome() + natural + half_pad);
                let scroll_max = (page_slot_doc_top + runway - window_h).max(0.0);
                view.update(cx, |this, _cx| {
                    this.composer_caret_scroll_pending.set(false);
                    let cur = this.page_scroll.offset().y.as_f32();
                    let next = caret_scroll_offset(
                        caret_doc_top,
                        caret_doc_bot,
                        window_h,
                        cur,
                        scroll_max,
                        CARET_SCROLL_MARGIN,
                    );
                    // Record the slot-relative offset the branch folded into the
                    // caret's document position (`page_slot_doc_top +
                    // editor_top_offset`) so a test can assert the editor-top
                    // offset is included — frame-independent, unlike the final
                    // `next` (gpui-clamped) or a cross-frame content read (lags).
                    this.docked_caret_slot_offset.set(caret_doc_bot - caret_bot);
                    if (next - cur).abs() > 0.5 {
                        let off = this.page_scroll.offset();
                        this.page_scroll.set_offset(point(off.x, px(next)));
                        true
                    } else {
                        false
                    }
                })
            } else {
                // Floating at natural height: the caret is already visible in the
                // bar (which floats at the window bottom, not at its slot), so
                // neither scroll applies — just consume the flag.
                view.update(cx, |this, _cx| {
                    this.composer_caret_scroll_pending.set(false);
                    false
                })
            };
            // The offset is consumed by the scroll container at layout time, so
            // a value written during this frame's paint applies next frame —
            // schedule a repaint (mirrors `record_height`).
            if changed {
                let view = view.downgrade();
                window.on_next_frame(move |_, cx| {
                    view.update(cx, |_, cx| cx.notify()).ok();
                });
            }
        },
    )
    .absolute()
    .top_0()
    .left_0()
    .size_full()
}

/// Paint `child` inside a single `Window::paint_layer` whose bounds are the
/// child's layout bounds grown upward by `dilate_top`.
///
/// A gpui paint layer forces every primitive painted within it to share one
/// draw `order`, derived from the layer bounds' overlap in the `BoundsTree`
/// (rather than each primitive deriving its own order from its own bounds). By
/// growing the layer upward to cover a drop shadow's blur reach — which gpui
/// otherwise omits from the shadow's registered bounds — the whole child sits
/// above the page content the shadow visually overlaps, with no z-order
/// discontinuity as the child scrolls. See the call site for the full rationale.
fn layered(child: impl IntoElement, dilate_top: Pixels) -> Layered {
    Layered {
        child: Some(child.into_any_element()),
        dilate_top,
    }
}

struct Layered {
    child: Option<AnyElement>,
    dilate_top: Pixels,
}

impl Element for Layered {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<gpui::ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static std::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, ()) {
        (self.child.as_mut().unwrap().request_layout(window, cx), ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _state: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) {
        self.child.as_mut().unwrap().prepaint(window, cx);
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _state: &mut Self::RequestLayoutState,
        _prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        let layer = Bounds {
            origin: point(bounds.origin.x, bounds.origin.y - self.dilate_top),
            size: size(bounds.size.width, bounds.size.height + self.dilate_top),
        };
        let child = self.child.as_mut().unwrap();
        window.paint_layer(layer, |window| child.paint(window, cx));
    }
}

impl IntoElement for Layered {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

/// Compact a draft's pending references onto app-core's `1..=N` ordinals at
/// the moment of posting, rewriting the body's `{{ embed N }}` markers to
/// match. Pure — the caller posts the returned pair and writes neither back,
/// so a **rejected** post leaves the draft untouched.
///
/// Why this exists: draft ordinals may gap. Removing a pending reference must
/// not renumber the survivors — their markers already address them, and the
/// wave-1 seam's rule is that ordinals are stable and gaps are fine. But
/// `AppCore::post_with_references` assigns edge ordinals `1..=N` **in supplied
/// order**, so the durable ordinals and the body's markers would disagree
/// after any removal. Compacting here reconciles the two exactly once, at the
/// boundary where the ordinals become durable.
///
/// Only **recognized** markers are rewritten — the editor's own
/// `embed_blocks` scan over the draft's embed map, i.e. exactly the markers
/// the composer is rendering as quote blocks (and, by the crate's lockstep
/// corpus test, exactly the set app-core expands upstream). A marker the
/// author defused inside a fence, or one addressing an ordinal the draft
/// carries no reference for, is left verbatim — it is literal text on every
/// other surface too.
fn compact_draft_references(
    body: &str,
    references: &[super::PendingReference],
) -> (String, Vec<eidola_app_core::ReferenceSpec>) {
    if references.is_empty() {
        return (body.to_string(), Vec::new());
    }
    // `references` arrives in ordinal order; position `i` becomes ordinal
    // `i + 1`.
    let remap: std::collections::HashMap<u64, u64> = references
        .iter()
        .enumerate()
        .map(|(i, r)| (r.ordinal, i as u64 + 1))
        .collect();
    let specs = references.iter().map(|r| r.spec.clone()).collect();

    let embeds = gpui_markdown_editor::EmbedMap::new(
        references
            .iter()
            .map(|r| (r.ordinal, r.snippet.to_string())),
    );
    let mut out = body.to_string();
    // Rewrite back-to-front so earlier spans keep their offsets as later ones
    // change length (`{{ embed 10 }}` → `{{ embed 2 }}`).
    for block in gpui_markdown_editor::embed::embed_blocks(body, &embeds)
        .into_iter()
        .rev()
    {
        let Some(&new) = remap.get(&block.ordinal) else {
            continue; // unmapped marker — literal text everywhere, left alone
        };
        if new == block.ordinal {
            continue;
        }
        out.replace_range(block.range, &gpui_markdown_editor::embed_marker(new));
    }
    (out, specs)
}

#[cfg(test)]
mod tests {
    use super::{
        CARET_SCROLL_MARGIN, approach_glide_offset, caret_scroll_offset, composer_bar_h,
        dock_runway,
    };

    const M: f32 = CARET_SCROLL_MARGIN;

    // -- The pre-dock glide (`glide_composer_toward_dock`'s pure core) -------
    //
    // Scene for all glide tests: an 800-tall window with a 400-tall floating
    // bar (the 50% cap), so the float line is at 400 and the approach zone is
    // the 400px where the slot's dock top sits under the bar.
    const WIN: f32 = 800.0;
    const FLOAT_TOP: f32 = 400.0;

    #[test]
    fn glide_lands_exactly_zero_at_the_threshold() {
        // A descent that reaches the threshold multiplies by zero — exact, not
        // an accumulation that drifts.
        let r0 = dock_runway(700.0, FLOAT_TOP, WIN); // mid-zone
        let r1 = dock_runway(FLOAT_TOP, FLOAT_TOP, WIN); // at the threshold
        assert_eq!(approach_glide_offset(-317.4, r0, r1), 0.0);
        // …including a single step that jumps clean past it (one large wheel
        // delta, a programmatic dock like go_home / scroll-to-tail).
        let below = dock_runway(1500.0, FLOAT_TOP, WIN); // slot below the zone
        let past = dock_runway(250.0, FLOAT_TOP, WIN); // docked, past the line
        assert_eq!(approach_glide_offset(-1234.5, below, past), 0.0);
    }

    #[test]
    fn glide_is_path_independent_over_any_monotone_descent() {
        // Stopping and restarting anywhere must not change where the offset
        // sits at a given runway: chained steps telescope to the direct step.
        let stops = [780.0, 655.0, 610.5, 512.0, 431.0];
        let mut off = -300.0;
        let mut r_prev = dock_runway(stops[0], FLOAT_TOP, WIN);
        for s in &stops[1..] {
            let r = dock_runway(*s, FLOAT_TOP, WIN);
            off = approach_glide_offset(off, r_prev, r);
            r_prev = r;
        }
        let direct = approach_glide_offset(
            -300.0,
            dock_runway(stops[0], FLOAT_TOP, WIN),
            dock_runway(431.0, FLOAT_TOP, WIN),
        );
        assert!(
            (off - direct).abs() < 0.01,
            "chained {off} vs direct {direct}"
        );
    }

    #[test]
    fn glide_ignores_retreat_and_still_lands_zero() {
        // Scrolling away from the dock leaves the offset alone…
        let r_mid = dock_runway(600.0, FLOAT_TOP, WIN);
        let r_back = dock_runway(750.0, FLOAT_TOP, WIN);
        assert_eq!(approach_glide_offset(-200.0, r_mid, r_back), -200.0);
        // …and a later descent from the retreated position still reaches
        // exactly zero at the threshold (scaled over the larger runway).
        let part = approach_glide_offset(-200.0, r_back, r_mid);
        assert!(part > -200.0 && part < 0.0, "part={part}");
        assert_eq!(
            approach_glide_offset(part, r_mid, dock_runway(FLOAT_TOP, FLOAT_TOP, WIN)),
            0.0
        );
    }

    #[test]
    fn glide_is_inert_outside_the_zone() {
        // With the slot anywhere below the zone the runway saturates at the
        // bar height, so consecutive frames read equal runways and a page
        // scroll moves nothing — the composer's content only ever eases while
        // the dock threshold is under the floating bar.
        let deep = dock_runway(2000.0, FLOAT_TOP, WIN);
        let nearer = dock_runway(900.0, FLOAT_TOP, WIN);
        assert_eq!(deep, WIN - FLOAT_TOP);
        assert_eq!(deep, nearer);
        assert_eq!(approach_glide_offset(-500.0, deep, nearer), -500.0);
    }

    #[test]
    fn float_bar_height_caps_under_max_and_pins_under_exact() {
        use super::{ComposerSizing, float_bar_height};
        // Max: the fraction is a cap — natural height wins below it.
        assert_eq!(
            float_bar_height(200.0, 0.5, 800.0, ComposerSizing::Max),
            200.0
        );
        assert_eq!(
            float_bar_height(600.0, 0.5, 800.0, ComposerSizing::Max),
            400.0
        );
        // Exact: the fraction is the bar, whichever side the content is on —
        // this is what lets a drag size the bar in excess of its content, and
        // what keeps a floating bar's height fixed as its content shrinks.
        assert_eq!(
            float_bar_height(200.0, 0.5, 800.0, ComposerSizing::Exact),
            400.0
        );
        assert_eq!(
            float_bar_height(600.0, 0.5, 800.0, ComposerSizing::Exact),
            400.0
        );
        // The resting mode is Max — every activation starts there.
        assert_eq!(ComposerSizing::default(), ComposerSizing::Max);
    }

    #[test]
    fn top_chrome_split_preserves_the_text_position() {
        // The separator (outside the scroll clip) plus the in-content scroll
        // gap must sum to the bar's total top chrome — the invariant that
        // keeps unscrolled text exactly where it sat before the split, keeps
        // the docked editor at its post-matching `2·half_pad` slot offset
        // (`caret_into_view`'s docked arm bakes that as `POST_PAD_Y`), and
        // keeps every height/dock computation honest about using the total.
        use super::{COMPOSER_SEPARATOR_H, SpaceView, composer_scroll_gap};
        assert!(
            COMPOSER_SEPARATOR_H > 0.0 && COMPOSER_SEPARATOR_H < SpaceView::composer_chrome(),
            "the separator is a thin, non-empty slice of the top chrome"
        );
        assert_eq!(
            COMPOSER_SEPARATOR_H + composer_scroll_gap(),
            SpaceView::composer_chrome()
        );
    }

    #[test]
    fn glide_never_deepens_the_offset_and_noops_at_top() {
        // An unscrolled composer stays unscrolled through the whole approach…
        let r0 = dock_runway(790.0, FLOAT_TOP, WIN);
        let r1 = dock_runway(500.0, FLOAT_TOP, WIN);
        assert_eq!(approach_glide_offset(0.0, r0, r1), 0.0);
        // …and a descent step can only move a scrolled offset toward zero.
        let stepped = approach_glide_offset(-120.0, r0, r1);
        assert!((-120.0..=0.0).contains(&stepped), "stepped={stepped}");
    }

    /// The dock ramp must reach the composer's **whole** content height, so
    /// the internal scroll it eases lands at exactly zero. Regression for the
    /// `bar_h.min(win − top_y)` clamp that pinned the scroll viewport to the
    /// visible bar: the ramp then stalled a `doc_reserve`-plus-rail short of
    /// the content, and a docked composer stayed scrolled off its own top by
    /// that much (and clipped at the bottom by the same).
    #[test]
    fn dock_ramp_reaches_the_full_content_height() {
        // A 760-tall window; the composer's content is taller than the window,
        // so `full_h` is the content and the bar must ramp all the way to it.
        let (win, chrome, doc_reserve): (f32, f32, f32) = (760.0, 20.0, 36.0);
        let content: f32 = 1400.0;
        let full_h = (content + chrome).max(win);
        let float_bar_h = (content + chrome).min(0.5 * win);
        let float_top = win - float_bar_h;

        // Floating: the ramp is the identity.
        assert_eq!(
            composer_bar_h(
                float_bar_h,
                full_h,
                float_top,
                float_top,
                doc_reserve,
                false
            ),
            float_bar_h
        );

        // Docked at its resting position — the slot top comes to rest at the
        // document's top reserve, where progress is 1.
        let bar_h = composer_bar_h(
            float_bar_h,
            full_h,
            float_top,
            doc_reserve,
            doc_reserve,
            true,
        );
        assert!((bar_h - full_h).abs() < 0.01, "bar_h={bar_h}");
        // …which leaves the scroll viewport at least as tall as the content:
        // nothing left to scroll, so the eased offset clamps to the top.
        let body_h = bar_h - chrome;
        assert!(body_h >= content, "body_h={body_h} content={content}");
        // The ramp is monotone and starts at the floating height.
        let mid = composer_bar_h(
            float_bar_h,
            full_h,
            float_top,
            (float_top + doc_reserve) / 2.0,
            doc_reserve,
            true,
        );
        assert!(mid > float_bar_h && mid < full_h, "mid={mid}");
    }

    #[test]
    fn caret_already_visible_is_a_noop() {
        // Viewport 100 tall, scrolled to top; caret at y=40..58 sits well
        // inside with margin — offset unchanged.
        assert_eq!(caret_scroll_offset(40.0, 58.0, 100.0, 0.0, 200.0, M), 0.0);
    }

    #[test]
    fn fit_height_composer_never_scrolls() {
        // scroll_max == 0 (content fits the viewport): even a caret past the
        // bottom can't move the offset — the phantom-scroll invariant.
        assert_eq!(caret_scroll_offset(90.0, 108.0, 100.0, 0.0, 0.0, M), 0.0);
    }

    #[test]
    fn caret_below_fold_scrolls_down() {
        // Caret bottom at 300 with a 100-tall viewport at top → reveal it near
        // the bottom: new view_top = 300 + margin − 100, offset = −(that).
        let off = caret_scroll_offset(282.0, 300.0, 100.0, 0.0, 400.0, M);
        assert!((off - -(300.0 + M - 100.0)).abs() < 0.01, "off={off}");
        // And the caret is now inside the viewport.
        let view_top = -off;
        assert!(view_top <= 282.0 && 300.0 <= view_top + 100.0);
    }

    #[test]
    fn caret_above_fold_scrolls_up() {
        // Scrolled down (offset −250, so view_top 250); caret at 120..138 is
        // above the fold → reveal near the top: view_top = 120 − margin.
        let off = caret_scroll_offset(120.0, 138.0, 100.0, -250.0, 400.0, M);
        assert!((off - -(120.0 - M)).abs() < 0.01, "off={off}");
    }

    #[test]
    fn target_is_clamped_into_valid_range() {
        // A caret past content end still can't scroll beyond scroll_max.
        let off = caret_scroll_offset(1000.0, 1018.0, 100.0, 0.0, 150.0, M);
        assert_eq!(off, -150.0);
        // Nor above the top (positive offset is impossible).
        let off = caret_scroll_offset(-50.0, -32.0, 100.0, -80.0, 150.0, M);
        assert_eq!(off, 0.0);
    }
}
