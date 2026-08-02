//! The space window's keyboard model — wave B of the accessibility program
//! (`work/tasks/12`), Mike's decided design.
//!
//! **Two devices, two jobs.** Tab moves *between* regions (the focus model in
//! [`crate::focus`]); the arrow keys are the *within-region* device for the
//! conversation. The conversation is a tree, so it gets its own map rather
//! than a linear tab walk through a hundred posts.
//!
//! **Two levels.** The tree level focuses a post; Enter *enters* it, moving
//! focus into that post's affordance row (the hover-gated Edit / Regenerate /
//! Save / Cancel verbs, which reveal on focus-within exactly as they do on
//! hover). Escape steps back out one level, and never closes the window.
//!
//! ```text
//! tree level     ↑ ↓        the visible render rows, in reading order
//!                ← →        branch moves (see `tree_target`)
//!                Home/End   first / last row of the selected path
//!                Enter      enter the post → affordance level
//!                <char>     jump to the trailing draft and start composing
//! affordance     ← →        cycle the post's verbs
//!                Enter      activate (gpui's own keyboard click)
//!                Escape     back to the tree level
//! tree level     Escape     leave the conversation (focus the view root)
//! ```
//!
//! **The Escape chain** has one owner per rung, and the rungs are ordered by
//! *who is innermost*, because gpui dispatches key **bindings** before key
//! **listeners** and bubbles listeners inner→outer:
//!
//! 1. An open **context menu** wins, always — the view root closes it and
//!    every inner handler yields first via
//!    [`SpaceView::context_menu_absorbs_escape`] (PR #259's split; nothing
//!    here changes it).
//! 2. An inline **edit session** cancels (the post row's own handler).
//! 3. The **composer** deactivates its draft.
//! 4. The **affordance level** steps back to the post.
//! 5. The **post level** releases tree focus to the view root.
//!
//! Rungs 4 and 5 are this module's, and they are deliberately last: they only
//! ever fire when nothing more specific claimed the press, because tree focus
//! and an active draft/edit session are mutually exclusive states.
//!
//! **Branch moves are deliberately dumb.** Mike's call: predictability over
//! cleverness. Left on a post with no siblings does *nothing* — it does not
//! walk up to the nearest fork — and Right on one likewise. See
//! [`tree_target`] for the exact rule.

use gpui::{App, Context, SharedString, Window};

use super::SpaceView;
use super::model::{NodeSrc, PostData, TreeNode};

/// Which rung of the two-level model holds focus.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum FocusLevel {
    /// The post row itself — the arrow-key surface.
    Post,
    /// One of the post's affordance-row verbs, by index into
    /// [`SpaceView::focused_post_verbs`].
    Affordance(usize),
}

/// Where keyboard focus sits inside the conversation, if anywhere.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TreeFocus {
    /// The focused post's tree node id.
    pub node_id: SharedString,
    pub level: FocusLevel,
}

/// A tree-level movement request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TreeMove {
    Up,
    Down,
    Left,
    Right,
    Home,
    End,
}

/// One level of the selected path: the sibling strip and which of them is
/// selected. This is exactly the shape [`SpaceView::selected_levels`] returns,
/// reduced to ids so the decision is pure and testable without a `Window`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Level {
    /// Every sibling in this strip, in render order.
    pub siblings: Vec<SharedString>,
    /// Which of them the path goes through.
    pub active: usize,
}

/// Reduce the view's selected levels to the pure [`Level`] shape, keeping only
/// **post** nodes as focus targets: a draft is the composer (reached by typing
/// or by Tab, never by an arrow) and a streaming leaf has no settled text to
/// read, so neither is a place focus can rest.
///
/// **The filter happens here, not after the move.** It used to be applied to
/// [`tree_target`]'s *answer*, which quietly broke `End`: a space's selected
/// path almost always ends in the tail draft, so `End` resolved to the composer
/// and was then thrown away — a no-op on the one key whose whole job is "take
/// me to the bottom". Filtering the input makes `path.last()` the last *post*,
/// and Left/Right stop counting a draft as a branch sibling for free.
///
/// A level that holds no post at all is dropped rather than left empty, because
/// [`tree_target`] indexes `levels` by position in the flattened path — an
/// empty level would desync the two and hand a branch move the wrong strip.
pub(crate) fn levels_from(levels: &[(Vec<&TreeNode>, usize)]) -> Vec<Level> {
    levels
        .iter()
        .filter_map(|(sibs, active)| {
            let kept: Vec<usize> = (0..sibs.len()).filter(|i| is_post_node(sibs[*i])).collect();
            if kept.is_empty() {
                return None;
            }
            // Keep pointing at the same node when it survived the filter;
            // otherwise fall to the nearest kept sibling before it.
            let active = kept
                .iter()
                .position(|i| i == active)
                .unwrap_or_else(|| kept.partition_point(|i| i < active).saturating_sub(1));
            Some(Level {
                siblings: kept.into_iter().map(|i| sibs[i].id.clone()).collect(),
                active,
            })
        })
        .collect()
}

/// Whether a node is a focusable post (see [`levels_from`]).
pub(crate) fn is_post_node(node: &TreeNode) -> bool {
    matches!(node.src, NodeSrc::Msg(_))
}

/// The pure move decision. `levels` is the selected path root→leaf; `focused`
/// is the id focus currently sits on. Returns the id to focus, or `None` when
/// the move is a deliberate no-op.
///
/// **Up / Down / Home / End** walk the *selected path* — the column of rows
/// the eye actually sees, in render order.
///
/// **Left / Right** are branch moves over the sibling strip the focused post
/// stands in:
///
/// - `Right` → the next sibling; failing that, if the focused post is itself a
///   fork, the branch after the one its strip currently rests on ("Right at a
///   fork enters the next sibling branch").
/// - `Left` → the previous sibling; at the *first* sibling of a fork's strip,
///   the fork's anchor post (the parent), which is the way back toward the
///   spine.
/// - At a post with no siblings and no branches, **both no-op**. This is the
///   decided rule: Left on a spine post does not hunt for the nearest fork.
pub(crate) fn tree_target(levels: &[Level], focused: &str, mv: TreeMove) -> Option<SharedString> {
    let path: Vec<&SharedString> = levels
        .iter()
        .filter_map(|l| l.siblings.get(l.active))
        .collect();
    let pos = path.iter().position(|id| id.as_ref() == focused)?;
    let level = levels.get(pos)?;
    match mv {
        TreeMove::Up => path.get(pos.checked_sub(1)?).map(|id| (*id).clone()),
        TreeMove::Down => path.get(pos + 1).map(|id| (*id).clone()),
        TreeMove::Home => path.first().map(|id| (*id).clone()),
        TreeMove::End => path.last().map(|id| (*id).clone()),
        TreeMove::Right => {
            if let Some(next) = level.siblings.get(level.active + 1) {
                return Some(next.clone());
            }
            // Standing on a fork anchor: descend into the branch after the one
            // the fork currently rests on.
            let below = levels.get(pos + 1)?;
            below.siblings.get(below.active + 1).cloned()
        }
        TreeMove::Left => {
            if level.siblings.len() <= 1 {
                return None;
            }
            if let Some(prev) = level.active.checked_sub(1) {
                return level.siblings.get(prev).cloned();
            }
            // The first sibling of a fork's strip: back to the anchor.
            path.get(pos.checked_sub(1)?).map(|id| (*id).clone())
        }
    }
}

/// The scroll offset that brings `[top, bottom)` into a `viewport_h`-tall
/// window with **minimal motion**: already visible ⇒ unchanged; below the fold
/// ⇒ scroll just far enough that its bottom sits at the fold (less `margin`);
/// above ⇒ just far enough that its top sits at the top. A row taller than the
/// viewport aligns to its top, so you land at the beginning of what you're
/// about to read.
///
/// Offsets are gpui page-scroll offsets: `0` is the document top and scrolling
/// down is *negative*. `min_y` is the most-negative valid offset.
pub(crate) fn scroll_into_view(
    top: f32,
    bottom: f32,
    viewport_h: f32,
    current: f32,
    min_y: f32,
    margin: f32,
) -> f32 {
    let view_top = -current;
    let view_bottom = view_top + viewport_h;
    let target = if top < view_top + margin {
        -(top - margin)
    } else if bottom > view_bottom - margin && bottom - top <= viewport_h - 2.0 * margin {
        -(bottom + margin - viewport_h)
    } else if bottom > view_bottom - margin {
        // Taller than the viewport: align its top rather than its bottom.
        -(top - margin)
    } else {
        return current;
    };
    target.clamp(min_y.min(0.0), 0.0)
}

/// Whether a keystroke is "the user started typing" for task 38 — a printable,
/// non-whitespace character with no command modifier.
///
/// Whitespace is excluded on purpose. Space is gpui's *activation* key on a
/// focused affordance (it fires the click), and neither a leading space nor a
/// leading newline is a draft anybody meant to start.
pub(crate) fn typed_character(keystroke: &gpui::Keystroke) -> Option<String> {
    let m = keystroke.modifiers;
    if m.platform || m.control || m.alt || m.function {
        return None;
    }
    let ch = keystroke.key_char.as_deref()?;
    let mut chars = ch.chars();
    let first = chars.next()?;
    if chars.next().is_some() {
        // A multi-character key_char is an IME/dead-key composition — the
        // editor's own input handler is the only honest place for those.
        return None;
    }
    if first.is_control() || first.is_whitespace() {
        return None;
    }
    Some(ch.to_string())
}

/// How much of the viewport to keep clear above/below a post the keyboard just
/// focused, so a revealed row never sits flush against the fold.
const REVEAL_MARGIN: f32 = 12.0;

impl SpaceView {
    /// The window's key handler for the conversation. Runs as a **listener**
    /// on the view root, so gpui's binding pass has already given every inner
    /// context (the composer's `MarkdownEditor`, an inline edit session) first
    /// refusal — an arrow key inside an editor never reaches here.
    ///
    /// Returns `true` when it consumed the press.
    pub(crate) fn handle_conversation_key(
        &mut self,
        ev: &gpui::KeyDownEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        // A composing session owns the keyboard entirely.
        if self.active_draft.is_some() || self.editing.is_some() {
            return false;
        }
        // Transient overlays speak before the tree does; the context menu's
        // Escape ownership is the view root's and is untouched by this.
        if self.context_menu.is_some() || self.band_menu.is_some() {
            return false;
        }

        let mv = match ev.keystroke.key.as_str() {
            _ if ev.keystroke.modifiers.modified() => None,
            "up" => Some(TreeMove::Up),
            "down" => Some(TreeMove::Down),
            "left" => Some(TreeMove::Left),
            "right" => Some(TreeMove::Right),
            "home" => Some(TreeMove::Home),
            "end" => Some(TreeMove::End),
            _ => None,
        };

        if let Some(mv) = mv {
            return self.move_tree_focus(mv, window, cx);
        }
        match ev.keystroke.key.as_str() {
            "enter" if !ev.keystroke.modifiers.modified() => self.enter_focus_level(window, cx),
            "escape" => self.leave_focus_level(window, cx),
            _ => {
                // Task 38: typing anywhere in the window with nothing composing
                // starts the trailing draft, with the character applied.
                match typed_character(&ev.keystroke) {
                    Some(ch) => self.type_to_compose(&ch, window, cx),
                    None => false,
                }
            }
        }
    }

    /// Move tree focus. With no focus yet, an arrow *enters* the conversation
    /// at the top of the selected path (Up/End land at its end instead), which
    /// is what makes the keyboard model reachable without a pointer.
    fn move_tree_focus(
        &mut self,
        mv: TreeMove,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        // At the affordance level the horizontal pair cycles the post's verbs
        // (Tab does too, since each verb is an ordinary tab stop); the vertical
        // pair steps back out to the post and then moves, so Down never
        // strands focus on a verb of the post you just left.
        if let Some(TreeFocus {
            node_id,
            level: FocusLevel::Affordance(i),
        }) = self.tree_focus.clone()
        {
            let count = self.post_verb_count(&node_id, cx);
            match mv {
                TreeMove::Left | TreeMove::Right if count > 1 => {
                    let next = match mv {
                        TreeMove::Right => (i + 1) % count,
                        _ => (i + count - 1) % count,
                    };
                    self.tree_focus = Some(TreeFocus {
                        node_id,
                        level: FocusLevel::Affordance(next),
                    });
                    cx.notify();
                    return true;
                }
                TreeMove::Left | TreeMove::Right => return true,
                _ => {
                    self.focus_post(node_id, window, cx);
                }
            }
        }

        let viewport = crate::chrome::content_size(window);
        let (page_width, window_h) = (viewport.width, viewport.height);
        let turns = self.stream_overlays(cx);
        let tree = self.effective_tree(page_width, &turns);
        // `levels_from` already keeps only post nodes, so this *is* the
        // navigable path — no post-hoc filter, which is what used to eat `End`.
        let levels = levels_from(&self.selected_levels(&tree, page_width));
        let posts: Vec<SharedString> = levels
            .iter()
            .filter_map(|l| l.siblings.get(l.active))
            .cloned()
            .collect();
        if posts.is_empty() {
            return false;
        }

        let target = match self.tree_focus.as_ref() {
            None => match mv {
                TreeMove::Up | TreeMove::End => posts.last().cloned(),
                _ => posts.first().cloned(),
            },
            Some(focus) => tree_target(&levels, &focus.node_id, mv),
        };
        let Some(target) = target else {
            // A deliberate no-op still counts as handled: the conversation
            // owns the arrows while it holds focus, and letting an unhandled
            // Left fall through to some ancestor is how surprises happen.
            return self.tree_focus.is_some();
        };

        // A branch move changes the selection; a vertical one doesn't, and
        // re-selecting the same path is a no-op the scroll handles absorb.
        self.select_path_to(&tree, &target, page_width);
        self.focus_post(target, window, cx);
        // Recompute against the *new* selection before measuring.
        let tree = self.effective_tree(page_width, &turns);
        self.reveal_focused_post(&tree, page_width, window_h);
        cx.notify();
        true
    }

    /// Put tree focus on `node_id` at the post level and hand the window's
    /// focus to the row's handle, so the ring paints and AccessKit is told.
    #[doc(hidden)]
    pub fn focus_post(
        &mut self,
        node_id: SharedString,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.tree_focus = Some(TreeFocus {
            node_id,
            level: FocusLevel::Post,
        });
        window.focus(&self.post_focus, cx);
    }

    /// Scroll the focused post into view with **minimal motion** — see
    /// [`scroll_into_view`]. A no-op when the post is already on screen.
    fn reveal_focused_post(
        &mut self,
        tree: &[TreeNode],
        page_width: gpui::Pixels,
        window_h: gpui::Pixels,
    ) {
        let Some(focus) = self.tree_focus.as_ref() else {
            return;
        };
        let Some(top) = self.selected_path_doc_top(tree, &focus.node_id, page_width, window_h)
        else {
            return;
        };
        let Some(node) = super::model::node_ref(tree, &focus.node_id) else {
            return;
        };
        let height = self.node_height(node, page_width, window_h);
        let off = self.page_scroll.offset();
        let y = scroll_into_view(
            top,
            top + height,
            window_h.as_f32(),
            off.y.as_f32(),
            self.scroll_min_y.get(),
            REVEAL_MARGIN,
        );
        self.page_scroll.set_offset(gpui::point(off.x, gpui::px(y)));
    }

    /// Enter: descend one level. From the tree level, focus the focused post's
    /// first affordance — the hover-gated verbs, which
    /// [`SpaceView::post_affordances_revealed`] reveals for exactly this. With
    /// no verbs (a post with none, or a space mid-stream) it is a no-op rather
    /// than a level that contains nothing.
    fn enter_focus_level(&mut self, window: &mut Window, cx: &mut Context<Self>) -> bool {
        let Some(focus) = self.tree_focus.as_ref() else {
            return false;
        };
        if !matches!(focus.level, FocusLevel::Post) {
            // At the affordance level Enter is gpui's own keyboard click on the
            // focused verb; nothing for us to do.
            return false;
        }
        let node_id = focus.node_id.clone();
        if self.post_verb_count(&node_id, cx) == 0 {
            return false;
        }
        self.tree_focus = Some(TreeFocus {
            node_id,
            level: FocusLevel::Affordance(0),
        });
        window.focus(&self.affordance_focus, cx);
        cx.notify();
        true
    }

    /// Escape: step back out one level — affordance → post → nothing. It never
    /// closes the window, and it is the *last* rung of the chain (see the
    /// module docs): an open menu, an edit session and an active draft have
    /// all already had the press.
    fn leave_focus_level(&mut self, window: &mut Window, cx: &mut Context<Self>) -> bool {
        let Some(focus) = self.tree_focus.as_ref() else {
            return false;
        };
        match focus.level {
            FocusLevel::Affordance(_) => {
                let node_id = focus.node_id.clone();
                self.focus_post(node_id, window, cx);
            }
            FocusLevel::Post => {
                self.tree_focus = None;
                window.focus(&self.focus_handle, cx);
            }
        }
        cx.notify();
        true
    }

    /// Task 38 — **type to compose.** A printable character with nothing
    /// composing jumps to the trailing speculative draft at the end of the
    /// current branch, caret after any text it already holds, and applies the
    /// character.
    ///
    /// **The page does not move.** That is the whole point of the shortcut: it
    /// is a way to *start writing* from wherever the reader is, not a
    /// navigation gesture. So the activation deliberately skips every settling
    /// step a click on the composer would run — no `dock_active_draft`, no
    /// `scroll_to_tail`, no branch re-selection. (An off-screen composer
    /// *floats* at the window bottom, so the caret reveal a buffer change arms
    /// moves the composer's own internal scroll, never the page.)
    fn type_to_compose(&mut self, ch: &str, window: &mut Window, cx: &mut Context<Self>) -> bool {
        let viewport = crate::chrome::content_size(window);
        let page_width = viewport.width;
        let turns = self.stream_overlays(cx);
        let tree = self.effective_tree(page_width, &turns);
        // The trailing draft of the branch the reader is on: the tail draft
        // hanging off the selected leaf, else the blank space's root draft.
        // A tail draft attaches as a leaf of the post it follows, so the
        // selected leaf is usually the draft itself; failing that, the draft
        // hanging off the selected leaf, and failing that a blank space's root
        // draft.
        let leaf = self.selected_leaf_id(&tree, page_width);
        let draft_id = leaf
            .as_ref()
            .and_then(|l| self.drafts.iter().find(|d| d.id == *l))
            .or_else(|| {
                leaf.as_ref().and_then(|l| {
                    self.drafts
                        .iter()
                        .find(|d| d.parent.as_deref() == Some(l.as_ref()))
                })
            })
            .or_else(|| self.drafts.iter().find(|d| d.parent.is_none()))
            .map(|d| d.id.clone());
        let Some(draft_id) = draft_id else {
            return false;
        };

        self.tree_focus = None;
        self.activate_draft(draft_id.clone(), cx);
        if let Some(draft) = self.drafts.iter().find(|d| d.id == draft_id) {
            let editor = draft.editor.clone();
            let handle = editor.read(cx).focus_handle.clone();
            editor.update(cx, |e, cx| e.append_at_end(ch, cx));
            handle.focus(window, cx);
            // `activate_draft` seeds the accessible value from the draft *as it
            // stood* — which here is before the character that started the
            // session. Re-seed after the append, or the composer reports its
            // pre-keystroke text until focus leaves it (the §4 freeze rule
            // means nothing else refreshes a focused composer's value, which is
            // exactly what makes a stale seed stick).
            self.seed_composer_aria_value(&draft_id, cx);
        }
        cx.notify();
        true
    }

    /// How many affordance-row verbs the post at `node_id` currently offers —
    /// the affordance level's cycle length. Mirrors
    /// [`SpaceView::render_post_actions`]'s own gating so the two cannot
    /// disagree about what is there to focus.
    pub(crate) fn post_verb_count(&self, node_id: &str, cx: &Context<Self>) -> usize {
        if self.editing.as_ref().map(|e| &e.node_id) == Some(&SharedString::from(node_id)) {
            return 2; // Save, Cancel
        }
        if self.space.read(cx).is_streaming() || self.editing.is_some() {
            return 0;
        }
        let Some(idx) = self.post_index(node_id) else {
            return 0;
        };
        let post = &self.posts[idx];
        if post.action_id.is_none() {
            return 0;
        }
        match post.role.as_ref() {
            "user" | "assistant" => 1,
            _ => 0,
        }
    }

    /// The transcript index of the post rendering as tree node `node_id`.
    fn post_index(&self, node_id: &str) -> Option<usize> {
        (0..self.posts.len()).find(|&i| super::model::node_id(&self.posts, i) == node_id)
    }

    /// Whether the post row itself should track the view's post focus handle
    /// this frame — i.e. tree focus is on it and has **not** descended into its
    /// affordance row (where the verb tracks its own handle instead).
    pub(crate) fn post_row_holds_focus(&self, node_id: &str) -> bool {
        matches!(
            self.tree_focus.as_ref(),
            Some(TreeFocus { node_id: id, level: FocusLevel::Post }) if id == node_id
        )
    }

    /// Whether the post at `node_id` should show its affordance row: hovered,
    /// **or** carrying keyboard focus. The focus half is wave B's S7 fix —
    /// gpui suppresses hover entirely while the input modality is keyboard, so
    /// without it a keyboard user could focus a post and find its verbs gone.
    #[doc(hidden)]
    pub fn post_affordances_revealed(&self, node_id: &str) -> bool {
        self.hovered_post.as_deref() == Some(node_id)
            || self
                .tree_focus
                .as_ref()
                .is_some_and(|f| f.node_id == node_id)
    }

    /// Which affordance index, if any, the keyboard has entered on this post.
    pub(crate) fn focused_affordance(&self, node_id: &str) -> Option<usize> {
        match self.tree_focus.as_ref() {
            Some(TreeFocus {
                node_id: id,
                level: FocusLevel::Affordance(i),
            }) if id == node_id => Some(*i),
            _ => None,
        }
    }

    /// Release tree focus once the conversation no longer holds the window's
    /// focus. Called at the head of [`SpaceView::render`], which is the one
    /// place with both `&mut self` and a `&Window` to ask.
    ///
    /// [`SpaceView::tree_focus`] is bookkeeping — where the *keyboard model*
    /// thinks it is — while the window's focus is the truth, and Tab, a click
    /// on the composer, or an inline edit session all move the latter without
    /// telling the former. The post ring is drawn manually from the
    /// bookkeeping (`post_row_holds_focus`), so a stale value paints a second
    /// apparent focus target beside the real one, and the post's hover-gated
    /// verbs stay revealed on a post nobody is on. Deriving from the real
    /// state costs one comparison per frame and needs no clear-call at any of
    /// the exits — which is the point: there is no exit to forget.
    ///
    /// **Each level is observed through the handle that can actually answer
    /// for it.** The post level asks `post_focus.is_focused` — one element, one
    /// handle, exact. The affordance level cannot ask
    /// `affordance_focus.is_focused`, because Tab walks between the post's
    /// sibling verbs (the keyboard map promises it) and only the *named* verb
    /// tracks that handle; the others ride gpui's implicit handles, which this
    /// crate never receives, so an exact test would drop the level on the very
    /// Tab it advertises. It asks the row **container** instead
    /// ([`SpaceView::affordance_row_focus`]): `contains_focused` resolves
    /// through the dispatch tree, where an implicit descendant is an ordinary
    /// node — so a Tab *within* the row stays inside the container and a Tab
    /// *out* of it clears, which is exactly the distinction the level means.
    ///
    /// The named verb's own handle is checked **as well**, and that is not
    /// belt-and-braces: containment is answered from
    /// `window.rendered_frame.dispatch_tree`, i.e. the **last painted** frame,
    /// but the gutter only starts tracking the row handle on the frame that
    /// renders *after* the level is entered. Without the exact half, the entry
    /// frame would read "nothing inside is focused", clear the level, and the
    /// gutter would never track it again — a one-frame lag that latches.
    ///
    /// **A transient overlay borrows the keyboard; it does not end the level.**
    /// A context menu, a band menu or the highlight picker takes the window's
    /// focus while it is open — and, at this pin, does *not* hand it back when
    /// it closes. Reading that as "the conversation lost focus" would silently
    /// consume a rung of the decided Escape chain (the menu answers the first
    /// press, the focus levels the ones after). So the observation parks while
    /// an overlay is open — the same set
    /// [`SpaceView::handle_conversation_key`] yields to, so the key handler and
    /// the observation cannot disagree about who owns the keyboard — and, on
    /// the falling edge, **restores** focus to the level's own handle rather
    /// than clearing it. That is the honest reading of a borrow, and it is one
    /// place rather than a restore-call on each of the overlay-close paths
    /// (there are eight).
    pub(crate) fn sync_tree_focus(&mut self, window: &mut Window, cx: &mut App) {
        let overlay = self.context_menu.is_some()
            || self.band_menu.is_some()
            || self.highlight_picker.is_some();
        let borrowed = std::mem::replace(&mut self.overlay_held_focus, overlay);
        if overlay {
            return;
        }
        let Some(focus) = self.tree_focus.as_ref() else {
            return;
        };
        let live = match focus.level {
            FocusLevel::Post => self.post_focus.is_focused(window),
            FocusLevel::Affordance(_) => {
                self.affordance_focus.is_focused(window)
                    || self.affordance_row_focus.contains_focused(window, cx)
            }
        };
        if live {
            return;
        }
        if borrowed {
            let handle = match focus.level {
                FocusLevel::Post => self.post_focus.clone(),
                FocusLevel::Affordance(_) => self.affordance_focus.clone(),
            };
            window.focus(&handle, cx);
            return;
        }
        self.tree_focus = None;
    }

    /// Carry keyboard tree focus across a generation change of the post it
    /// sits on — the [`SpaceView::rethread_drafts`] rule applied to the other
    /// window-local reference into the transcript.
    ///
    /// [`TreeFocus::node_id`] is a tree node id, which for a post is its
    /// **action** id. An edit or a regeneration appends a new generation of the
    /// same *item*, so the reloaded transcript carries only the new tip and the
    /// focused id names a post that is no longer there. Nothing then recovers:
    /// [`tree_target`] can't find the id in the path and returns `None`, and
    /// `move_tree_focus`'s deliberate-no-op arm reports the press as *handled*
    /// — so every arrow reads as inert (and `post_verb_count` finds no post, so
    /// Enter is dead too) until the reader escapes out of the conversation
    /// entirely. Focus a post, edit it, and the keyboard model is simply gone.
    ///
    /// Reply threading already follows item identity (workspace `AGENTS.md`:
    /// an action id is causality, an item id is the intended logical flow), so
    /// the cure is the same one: resolve the vanished action id through the
    /// *outgoing* snapshot to its item, then forward focus to that item's
    /// current tip in the incoming one. The level (post vs. affordance) is
    /// preserved — an edit committed from the affordance row leaves you on the
    /// affordance row of the post you were editing. Focus is cleared **only**
    /// when the item genuinely left the snapshot, which is the honest outcome
    /// for a post that no longer exists.
    pub(crate) fn retarget_tree_focus(&mut self, next: &[PostData]) {
        let Some(focus) = self.tree_focus.as_ref() else {
            return;
        };
        let stale = focus.node_id.clone();
        // Node ids, not action ids: an optimistic row with no action id yet
        // renders under a positional fallback, and focus can sit on it.
        if (0..next.len()).any(|i| super::model::node_id(next, i) == stale) {
            return;
        }
        let forwarded = self
            .posts
            .iter()
            .find(|p| p.action_id.as_deref() == Some(stale.as_ref()))
            .and_then(|p| p.item_id.clone())
            .and_then(|item| {
                next.iter()
                    .find(|p| p.item_id.as_deref() == Some(item.as_ref()))
                    .and_then(|p| p.action_id.clone())
            });
        match forwarded {
            Some(tip) => {
                let level = focus.level.clone();
                self.tree_focus = Some(TreeFocus {
                    node_id: tip,
                    level,
                });
            }
            None => self.tree_focus = None,
        }
    }

    /// Test seam: where tree focus sits, as `(node id, level index)` — `None`
    /// at the post level, `Some(i)` at affordance `i`.
    #[doc(hidden)]
    pub fn tree_focus_for_test(&self) -> Option<(String, Option<usize>)> {
        self.tree_focus.as_ref().map(|f| {
            (
                f.node_id.to_string(),
                match f.level {
                    FocusLevel::Post => None,
                    FocusLevel::Affordance(i) => Some(i),
                },
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lv(sibs: &[&str], active: usize) -> Level {
        Level {
            siblings: sibs.iter().map(|s| SharedString::from(*s)).collect(),
            active,
        }
    }

    /// A linear conversation: a → b → c, no forks anywhere.
    fn linear() -> Vec<Level> {
        vec![lv(&["a"], 0), lv(&["b"], 0), lv(&["c"], 0)]
    }

    /// A fork at `b`: two branches, `b1` (the spine) and `b2`, and the strip
    /// currently rests on `b1`, which continues to `b1a`.
    fn forked() -> Vec<Level> {
        vec![
            lv(&["a"], 0),
            lv(&["b"], 0),
            lv(&["b1", "b2"], 0),
            lv(&["b1a"], 0),
        ]
    }

    #[test]
    fn up_and_down_walk_the_visible_rows() {
        let l = linear();
        assert_eq!(tree_target(&l, "a", TreeMove::Down).unwrap(), "b");
        assert_eq!(tree_target(&l, "b", TreeMove::Down).unwrap(), "c");
        assert_eq!(tree_target(&l, "c", TreeMove::Down), None);
        assert_eq!(tree_target(&l, "c", TreeMove::Up).unwrap(), "b");
        assert_eq!(tree_target(&l, "a", TreeMove::Up), None);
    }

    #[test]
    fn home_and_end_reach_the_ends_of_the_selected_path() {
        let l = forked();
        assert_eq!(tree_target(&l, "b1", TreeMove::Home).unwrap(), "a");
        assert_eq!(tree_target(&l, "a", TreeMove::End).unwrap(), "b1a");
    }

    #[test]
    fn left_and_right_no_op_at_non_forking_posts() {
        // The decided rule (Mike, 2026-08-01): predictability over cleverness.
        // Left on a spine post does *not* walk up to the nearest fork.
        let l = linear();
        for id in ["a", "b", "c"] {
            assert_eq!(tree_target(&l, id, TreeMove::Left), None, "left at {id}");
            assert_eq!(tree_target(&l, id, TreeMove::Right), None, "right at {id}");
        }
    }

    #[test]
    fn right_at_a_fork_enters_the_next_sibling_branch() {
        let l = forked();
        // From the anchor: into the branch after the selected one.
        assert_eq!(tree_target(&l, "b", TreeMove::Right).unwrap(), "b2");
        // Standing in the strip: the next sibling.
        assert_eq!(tree_target(&l, "b1", TreeMove::Right).unwrap(), "b2");
        // …and no wrap at the end.
        let rested = vec![lv(&["a"], 0), lv(&["b"], 0), lv(&["b1", "b2"], 1)];
        assert_eq!(tree_target(&rested, "b2", TreeMove::Right), None);
    }

    #[test]
    fn left_returns_toward_the_spine_then_to_the_anchor() {
        let rested = vec![lv(&["a"], 0), lv(&["b"], 0), lv(&["b1", "b2"], 1)];
        // From the second branch, back to the first.
        assert_eq!(tree_target(&rested, "b2", TreeMove::Left).unwrap(), "b1");
        // From the first branch, out to the fork's anchor post.
        let l = forked();
        assert_eq!(tree_target(&l, "b1", TreeMove::Left).unwrap(), "b");
        // The anchor itself has no siblings here — no-op, per the rule above.
        assert_eq!(tree_target(&l, "b", TreeMove::Left), None);
    }

    #[test]
    fn a_post_off_the_selected_path_moves_nowhere() {
        // Focus always follows selection, so this cannot normally happen; the
        // guard keeps a stale id from teleporting the reader.
        let l = forked();
        assert_eq!(tree_target(&l, "b2", TreeMove::Down), None);
    }

    #[test]
    fn scroll_into_view_never_moves_a_visible_row() {
        // Row 100..200 inside a 0..600 viewport.
        assert_eq!(scroll_into_view(100., 200., 600., 0., -1000., 8.), 0.);
    }

    #[test]
    fn scroll_into_view_reveals_below_and_above_with_minimal_motion() {
        // Below the fold: bottom lands at the fold less the margin.
        let off = scroll_into_view(900., 1000., 600., 0., -2000., 8.);
        assert!((off - -(1000. + 8. - 600.)).abs() < 1e-3, "{off}");
        // Above: top lands at the top plus the margin.
        let off = scroll_into_view(100., 200., 600., -500., -2000., 8.);
        assert!((off - -92.).abs() < 1e-3, "{off}");
    }

    #[test]
    fn scroll_into_view_aligns_the_top_of_an_oversized_row() {
        // A post taller than the window: land at its beginning, not its end.
        let off = scroll_into_view(1000., 2000., 600., 0., -3000., 8.);
        assert!((off - -992.).abs() < 1e-3, "{off}");
    }

    #[test]
    fn scroll_into_view_clamps_to_the_document() {
        let off = scroll_into_view(900., 1000., 600., 0., -100., 8.);
        assert_eq!(off, -100.);
    }

    fn stroke(key: &str, key_char: Option<&str>, modifiers: gpui::Modifiers) -> gpui::Keystroke {
        gpui::Keystroke {
            key: key.into(),
            key_char: key_char.map(|s| s.to_string()),
            modifiers,
        }
    }

    #[test]
    fn typed_character_accepts_printable_input_only() {
        let plain = gpui::Modifiers::default();
        assert_eq!(
            typed_character(&stroke("a", Some("a"), plain)).as_deref(),
            Some("a")
        );
        assert_eq!(
            typed_character(&stroke("é", Some("é"), plain)).as_deref(),
            Some("é")
        );
        // Whitespace is the activation key / a pointless leading blank.
        assert_eq!(typed_character(&stroke("space", Some(" "), plain)), None);
        assert_eq!(typed_character(&stroke("enter", Some("\n"), plain)), None);
        // Navigation keys carry no character at all.
        assert_eq!(typed_character(&stroke("down", None, plain)), None);
        // A command chord belongs to the keymap.
        assert_eq!(
            typed_character(&stroke(
                "a",
                Some("a"),
                gpui::Modifiers {
                    platform: true,
                    ..Default::default()
                }
            )),
            None
        );
        // Shift is fine — that is how capitals arrive.
        assert_eq!(
            typed_character(&stroke(
                "A",
                Some("A"),
                gpui::Modifiers {
                    shift: true,
                    ..Default::default()
                }
            ))
            .as_deref(),
            Some("A")
        );
    }
}
