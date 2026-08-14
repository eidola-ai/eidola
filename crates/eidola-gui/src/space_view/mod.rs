//! The Space view — a conversation rendered as a tree of recursively-nested
//! scrollers (vertical scroll walks the selected root→leaf path; horizontal
//! scroll flick-snaps between sibling branches, each page carrying its whole
//! subtree), with a right-edge topology minimap and a floating/docking
//! composer.
//!
//! This is the window-root **lens** over the shared [`crate::space::Space`]
//! entity (the `ChatView` analogue), decomposed into focused submodules:
//!
//! - [`model`] — the UI tree + pure structural helpers.
//! - [`layout`] — the cached layout/virtualization model (per-post heights,
//!   doc-space positions, selection-aware heights).
//! - [`nav`] — snap physics + scroll-gesture bookkeeping.
//! - [`post`] — render one post + its separator band (branch dots, the
//!   Reply-or-Ask menu).
//! - [`composer`] — the floating/docking draft composer, the Post routing,
//!   and the action gutter (Post / ⌥ Post quietly).
//! - [`context_menu`] — the right-click menu over any of the space's editors.
//! - [`inspector`] — the per-space settings panel that splits the window.
//! - [`inspector_participants`] — that panel's Participants section (the roster
//!   the standalone Participants window used to hold).
//! - [`keyboard`] — the two-level keyboard model over the tree (wave B).
//! - [`minimap`] — the topology minimap.
//! - [`traces`] — the per-post trace disclosure (what a turn actually did).
//!
//! Performance: only posts intersecting the viewport render the real
//! `MarkdownEditor`; off-screen posts render as sized placeholders sized from
//! the cached layout, so per-frame text shaping is bounded to visible posts.

pub mod composer;
pub mod context_menu;
pub mod inspector;
pub mod inspector_participants;
pub mod keyboard;
pub mod layout;
pub mod minimap;
pub mod model;
pub mod nav;
pub mod post;
pub mod references;
pub mod traces;

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use eidola_app_core::error::AppError;
use gpui::{
    AnyElement, AppContext, Bounds, ClipboardItem, Context, Entity, FocusHandle, Focusable,
    FontWeight, InteractiveElement, IntoElement, IsZero, Overflow, ParentElement, Pixels, Render,
    Role, ScrollHandle, ScrollWheelEvent, SharedString, StatefulInteractiveElement, Styled,
    Subscription, Task, TouchPhase, Window, div, prelude::FluentBuilder as _, px, rems,
};
use gpui_component::{ActiveTheme, h_flex, v_flex};
use gpui_markdown_editor::{MarkdownEditorState, MarkdownStyle};

use crate::actions::CloseWindow;
use crate::focus::TabRegion as _;
use crate::overlay::{Contain as _, Overlay};
use crate::probe::Probe as _;
use crate::space::{ChatMessageView, Space, SpaceEvent};
use crate::stores::Stores;
use crate::theme;
use crate::window_input::WindowInput;

use layout::{ComposerGutterHeights, Layout};
use model::{NodeSrc, PostData, TreeNode};
use nav::{PageGlide, ScrollAxis, ScrollOwner, SnapAnim};

// Re-export the composer actions (⌘↩ Post / ⌘⇧↩ Post quietly, routed via
// the editor's `PressEnter` event).
pub use crate::actions::{PostOnly, Send};

// ---------------------------------------------------------------------------
// Layout constants — the book typography + the tree-navigation geometry.
// ---------------------------------------------------------------------------

/// Height reserved at the top of the window for the (transparent) titlebar —
/// the macOS traffic-light band / the Linux CSD controls + drag strip.
pub(crate) const TITLE_BAR_RESERVE: Pixels = crate::titlebar::DRAG_BAND_HEIGHT;

/// Prose typography for narrative content — Newsreader at a book size/leading,
/// distinct from the system UI font the theme uses for chrome. Matches the
/// chat composer's typography so a post shapes pixel-for-pixel like what the
/// user types.
pub(crate) const PROSE_FONT_SIZE: Pixels = px(17.);
pub(crate) const PROSE_LINE_HEIGHT: f32 = 1.65;
/// Inter-block spacing as a multiple of the font size (the editor splits it
/// half above / half below each block). Kept here as the single source of truth
/// for both [`prose_style`] and the height estimate, so they can't drift.
pub(crate) const PROSE_PARAGRAPH_GAP: f32 = 1.5;

/// The byline gutter (right-aligned author + time) and the centered reading
/// column it sits beside.
pub(crate) const GUTTER_WIDTH: Pixels = px(120.);
pub(crate) const GUTTER_GAP: Pixels = px(28.);
pub(crate) const BODY_MAX_WIDTH: Pixels = px(600.);

/// Vertical breathing room around each post, plus the faint full-bleed band
/// that separates one depth level of the tree from the next.
pub(crate) const POST_PAD_Y: Pixels = px(40.);
pub(crate) const BAND_HEIGHT: Pixels = px(48.);

/// The **default** floating-composer fraction of the window height — every
/// window opens with it, so the out-of-the-box behavior is unchanged. The live
/// value is per-window state ([`SpaceView::composer_fraction`], a ratio so it
/// survives window resizes), adjusted by dragging the floating bar's separator
/// handle and applied per [`composer::ComposerSizing`]: as a *max* (the bar
/// grows with content up to the fraction, then scrolls internally — the
/// resting behavior) or *exact* (the bar is pinned to the fraction while
/// floating — entered by the resize drag, reverted on deactivation). Near the
/// bottom of a branch the composer *docks* and grows into the page either way.
pub(crate) const COMPOSER_MAX_FRACTION: f32 = 0.5;

/// Width of the topology minimap pinned to the right edge, and the gap between
/// sibling columns within a minimap row.
pub(crate) const MINIMAP_WIDTH: Pixels = px(36.);
pub(crate) const MINIMAP_COL_GAP: Pixels = px(4.);
/// How long the minimap lingers after scrolling stops before it fades, and the
/// fade-out duration once hiding begins (mirrors macOS overlay scrollbars).
pub(crate) const MINIMAP_HIDE_DELAY: std::time::Duration = std::time::Duration::from_millis(400);
pub(crate) const MINIMAP_FADE: std::time::Duration = std::time::Duration::from_millis(200);

/// How far beyond the viewport a post may sit and still render the real editor
/// (so a post is fully shaped just before it scrolls into view). Off-screen
/// past this margin renders as a sized placeholder.
pub(crate) const VIRT_MARGIN: f32 = 600.0;

/// How close to the end of the document the page must sit for a streaming turn
/// to keep it pinned there ("tail-following"). A couple of pixels of slack
/// absorbs sub-pixel layout rounding without ever mistaking a deliberate scroll
/// away from the tail for "still at the end".
pub(crate) const TAIL_FOLLOW_EPSILON: f32 = 2.0;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum TailPin {
    #[default]
    Inactive,
    Converging,
    Observing,
}

impl TailPin {
    fn active(self) -> bool {
        self != Self::Inactive
    }

    fn forced(self) -> bool {
        self == Self::Converging
    }
}

impl SpaceView {
    /// A reader-driven scroll or navigation takes the viewport from the
    /// post-submit pin: a still-forcing `Converging` demotes to `Observing`,
    /// so the exchange keeps following only while the ordinary at-tail
    /// observation holds — the pin never snaps a reader back who moved (or
    /// asked to be taken) somewhere else before convergence finished. The
    /// callers are exactly the seams that cancel a navigation glide, plus the
    /// glide door itself: the reader's own motions, and the navigations they
    /// asked for.
    pub(crate) fn demote_tail_pin_for_reader(&mut self) {
        if self.tail_pin.forced() {
            self.tail_pin = TailPin::Observing;
        }
    }
}

/// Vertical band (px) at the top and bottom of the viewport within which an
/// active readonly-post selection drag autoscrolls the page.
pub(crate) const SELECTION_AUTOSCROLL_MARGIN: f32 = 56.0;
/// Peak autoscroll speed (px per frame) at the very edge of the viewport,
/// ramping down to zero at the inner edge of [`SELECTION_AUTOSCROLL_MARGIN`].
pub(crate) const SELECTION_AUTOSCROLL_MAX_SPEED: f32 = 32.0;

/// `MarkdownStyle` for prose bodies and the composer: Newsreader at a book
/// size/leading with a size-led heading ramp (h1 2.5× … h4 1.125×) at a
/// uniform Medium (500) weight — size carries the hierarchy — and Courier-New
/// inline code (its x-height matches Newsreader's, where Menlo reads too
/// large). Mirrors the website's prose ramp (`www/static/site.css`).
/// `from_theme` seeds the system font + theme colors, so we override the
/// family back to Newsreader for narrative content.
pub(crate) fn prose_style(cx: &gpui::App) -> MarkdownStyle {
    // The prose ramp is the one place the type ramp is spelled in absolute
    // pixels rather than `rems()`, so it doesn't ride the scaled `rem_size`
    // automatically — multiply the base size by the type-scale factor here. The
    // `rems()`-relative leading and paragraph gap below *do* ride `rem_size` (=
    // the theme UI font size, itself scaled), so they stay proportional to the
    // scaled prose size for free.
    let base = PROSE_FONT_SIZE * theme::font_scale(cx);
    let mut style = MarkdownStyle::from_theme(cx)
        .font_size(base)
        .line_height(rems(PROSE_LINE_HEIGHT))
        .paragraph_gap(rems(PROSE_PARAGRAPH_GAP))
        .heading_base_font_size(base)
        .heading_font_size(|level, base| match level {
            1 => base * 2.5,
            2 => base * 1.75,
            3 => base * 1.25,
            _ => base * 1.125,
        })
        .heading_weight(|_| FontWeight::MEDIUM)
        .highlight_color(quoted_passage_wash(cx))
        .inline_code_font_family("Courier New");
    style.font_family = theme::FONT_FAMILY.into();
    style
}

/// The wash behind a passage someone has quoted (the editor's opaque
/// highlight plugin, driven by `references::highlight_ranges`).
///
/// It reads as a **reader's pencil mark, not a UI selection**: the brand's own
/// warm amber at a very low alpha, so the text on top is untouched and the
/// mark is something you notice rather than something that interrupts. It is
/// deliberately fainter than `theme.selection` in both palettes, so selecting
/// across a quoted passage still reads as a selection; day carries a touch
/// less alpha than night because the near-white paper shows a wash more
/// readily than the blue-grey dark ground does.
pub(crate) fn quoted_passage_wash(cx: &gpui::App) -> gpui::Hsla {
    if cx.theme().mode.is_dark() {
        gpui::hsla(0.09, 0.50, 0.60, 0.16)
    } else {
        gpui::hsla(0.09, 0.70, 0.52, 0.13)
    }
}

/// Format a unix-millis timestamp to a clock time ("10:03 AM") for the gutter
/// byline. Empty for a missing/zero timestamp (synthetic rows). Uses the UTC
/// time-of-day (no timezone dependency); the value only needs to read as a time.
pub(crate) fn fmt_clock(ms: i64) -> String {
    if ms <= 0 {
        return String::new();
    }
    let tod = ms.div_euclid(1000).rem_euclid(86_400);
    let (mut h, m) = (tod / 3600, (tod % 3600) / 60);
    let am = h < 12;
    if h == 0 {
        h = 12;
    } else if h > 12 {
        h -= 12;
    }
    format!("{h}:{m:02} {}", if am { "AM" } else { "PM" })
}

/// A quoted reference the user has attached to a draft but not yet posted —
/// the window-local, pre-durable twin of `eidola_app_core::PostReference`.
///
/// **The ordinal is the shared key** (see the workspace `AGENTS.md`, "Quoted
/// references — the ordinal seam"): it addresses this reference in the draft
/// body's `{{ embed N }}` marker, in the editor's embed map (so the marker
/// renders as a real quote block while composing), in the footnote rail, and —
/// once posted — in the `action_antecedent` row app-core writes. Ordinals run
/// `1..` in add order and **never renumber**: removing one leaves a gap, which
/// is correct, because the embed map is a map and the surviving markers must
/// keep addressing the same references.
#[derive(Clone, Debug)]
pub(crate) struct PendingReference {
    /// This reference's ordinal within the draft (`1..`; 0 is app-core's
    /// reserved `reply` edge and is never minted here).
    pub(crate) ordinal: u64,
    /// The write-side spec handed to app-core at post time. Names the
    /// **concrete generation** quoted — references never remap to an item's
    /// current tip.
    pub(crate) spec: eidola_app_core::ReferenceSpec,
    /// The quoted post's byline ("You", an agent's label) — the footnote row's
    /// attribution.
    pub(crate) byline: SharedString,
    /// The quoted markdown itself — the embed map's value (what the marker
    /// renders as) and the footnote row's snippet.
    pub(crate) snippet: SharedString,
}

/// An unsent draft reply — a local UI node with its own editor. It renders
/// inline like any post (a "Draft" byline + an editable body) when inactive,
/// and floats as the composer when it is the [`SpaceView::active_draft`].
/// Persists when deselected; a blank one is deleted on retire.
pub(crate) struct Draft {
    /// Unique, stable across frames (`draft-{seq}`) — the node id, editor key,
    /// and horizontal-scroll key.
    pub(crate) id: SharedString,
    /// The persisted post this draft replies to (its `action_id`), or `None`
    /// for a blank space's root draft. Becomes the new post's reply antecedent.
    pub(crate) parent: Option<SharedString>,
    /// This draft's editable editor (retains its content while deselected).
    pub(crate) editor: Entity<MarkdownEditorState>,
    /// The draft's pending quoted references, in ordinal order — the footnote
    /// rail's rows and the editor's embed map while composing. Handed to
    /// app-core as `Vec<ReferenceSpec>` when the draft is posted; a **rejected**
    /// post leaves them (and the draft) exactly as they were.
    pub(crate) references: Vec<PendingReference>,
    /// Focus → activate; `PressEnter` → submit/post-only. Held so it outlives
    /// the closure and is dropped with the draft.
    pub(crate) _sub: Subscription,
}

impl Draft {
    /// The next free ordinal: one past the highest in use, so removing a
    /// reference leaves a gap rather than renumbering the survivors (whose
    /// markers already address them).
    pub(crate) fn next_ordinal(&self) -> u64 {
        self.references.iter().map(|r| r.ordinal).max().unwrap_or(0) + 1
    }

    /// The draft's embed map — ordinal → quoted markdown — fed straight to
    /// `MarkdownEditorState::set_embeds` so each `{{ embed N }}` marker
    /// materializes as a quote block while composing. The composing twin of
    /// `PostNode::embed_map()`.
    pub(crate) fn embed_map(&self) -> Vec<(u64, String)> {
        self.references
            .iter()
            .map(|r| (r.ordinal, r.snippet.to_string()))
            .collect()
    }
}

/// The window-local cascade-paused notice: a submit's (or driven turn's)
/// notification plan hit the space's cascade limit at `target_action_id`.
/// Quiet and dismissible; its action is an explicit ask (which bypasses the
/// guard), so the conversation is resumable in one click.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CascadeNotice {
    pub(crate) depth: i64,
    pub(crate) limit: i64,
    pub(crate) target_action_id: String,
}

/// A branch selection deferred to the next render, where the effective tree
/// carrying the node (and the scroll handles the selection writes) both exist.
/// Three sources: a freshly-created fork draft, a freshly-started turn whose
/// streaming leaf is a *new sibling* under its target, and a completed turn
/// whose selection has to move from that leaf onto the post it wrote
/// ([`SpaceView::follow_completed_turn`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PendingSelect {
    /// The node to bring onto the selected path.
    pub(crate) node: SharedString,
    /// Where the page rests once the branch is selected.
    pub(crate) settle: PendingSettle,
}

/// The resting position a [`PendingSelect`] scrolls to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PendingSettle {
    /// Dock the active draft at its home (a new composer: you are writing).
    DockDraft,
    /// Park at the end of the selected branch (a new turn: the answer will
    /// grow there).
    BranchEnd,
    /// Leave the page where it is — the selection is *following* something the
    /// reader is already positioned on (a turn's leaf becoming its post), not
    /// taking them somewhere new.
    Stay,
}

/// The open source-highlight picker: a quoted passage that **several** posts
/// reference, so a click can't disambiguate on its own. Anchored to the post
/// whose text was clicked, listing each referencing post (byline + snippet) as
/// a choice — the band-menu pattern, one open at a time.
#[derive(Clone, Debug)]
pub(crate) struct HighlightPicker {
    /// The candidates: `(referencing action id, its space, the row's label)`.
    pub(crate) choices: Vec<(String, String, SharedString)>,
}

/// An in-progress inline edit of a persisted post: the post's own body editor
/// is enabled in place, `⌘↩` commits a new generation via [`Space::edit`],
/// Escape restores the original text. Window-local, like a draft.
pub(crate) struct EditingPost {
    /// The persisted action being edited (the commit target).
    pub(crate) action_id: SharedString,
    /// The tree node id keying [`SpaceView::bodies`] (the action id for
    /// persisted posts).
    pub(crate) node_id: SharedString,
    /// The pre-edit markdown, restored on Escape.
    pub(crate) original: String,
    /// Reference **ordinals** the user has marked for removal in this session
    /// (the footnote rail's chips). Handed to `edit_post_with_removals` on
    /// commit: the new generation drops them, the prior generation keeps them
    /// (history is append-only). Ordinal 0 — the structural `reply` edge — is
    /// never removable and can never enter this list (the rail renders only
    /// `reference` edges, and app-core refuses 0 besides).
    pub(crate) removed_references: Vec<i64>,
    /// `PressEnter` (⌘↩) on the post's editor commits the edit. Held so it
    /// dies with the edit session.
    pub(crate) _sub: Subscription,
}

pub struct SpaceView {
    /// The store bundle — held whole for model resolution (config default) and
    /// to open spaces through the `SpacesStore` registry.
    pub(crate) stores: Stores,
    /// The shared per-conversation entity. The view is a window-local lens over
    /// it; two windows on one space share this entity.
    pub(crate) space: Entity<Space>,
    /// Per-window modifier state — the root registers the window's single
    /// `on_modifiers_changed` listener and mirrors events here; the composer's
    /// action gutter reads it for the ⌥ reveal (Post + keyboard hints).
    pub(crate) window_input: Entity<WindowInput>,
    /// The view's focus handle — `track_focus`ed on the root so behavior tests
    /// dispatch actions through it; production focus lives on the composer.
    pub(crate) focus_handle: FocusHandle,
    pub(crate) _subs: Vec<Subscription>,

    /// Per-row render snapshot, rebuilt when the transcript changes — cheap
    /// `SharedString` data the tree/render path work over without re-cloning
    /// message content each frame.
    pub(crate) posts: Vec<PostData>,

    /// One read-only markdown-editor state per persisted post, keyed by node id.
    /// Posts render `disabled` so they're pixel-identical to the composer.
    pub(crate) bodies: HashMap<SharedString, Entity<MarkdownEditorState>>,
    /// The post content each body editor was last seeded with, keyed like
    /// [`Self::bodies`]. `sync_bodies` re-seeds an editor only when the
    /// *post's* content changes (edit/regenerate replacing it in place) —
    /// never because the editor's live buffer differs from the post. Comparing
    /// against the buffer instead (the previous behavior) turned any
    /// buffer-vs-source divergence into a per-frame `set_value` reset loop
    /// that made selection impossible on the affected post. The editor no
    /// longer rewrites read-only buffers at all (`update_readonly`), so this
    /// is defense in depth: if a divergence ever reappears, it must not
    /// escalate into a frame loop.
    pub(crate) body_seeds: HashMap<SharedString, SharedString>,
    /// One subscription per body editor, keyed like [`Self::bodies`]. It
    /// carries the read-only editors' `SelectionChanged` into
    /// [`Self::note_body_selection`] — the only way this view learns that a
    /// passage is quotable, and therefore what gates the Edit menu's Quote
    /// items. Dropped with the editor when the post leaves the transcript.
    pub(crate) body_subs: HashMap<SharedString, Subscription>,
    /// The live quotable selection inside some post's read-only body, if any
    /// (the most recent one made — editors don't clear each other's
    /// selections, so "the last passage you selected" is the honest reading).
    /// `None` means the Quote actions are unregistered and their menu items
    /// grey out. Window-local by nature: a selection is a cursor, and two
    /// windows on one space are two cursors (STATE.md's draft rule).
    pub(crate) post_selection: Option<references::PostSelection>,
    /// The open source-highlight picker, if any — a passage several posts
    /// quoted, awaiting a choice of which referencing post to visit. Window-
    /// local picker state, like the band menu.
    pub(crate) highlight_picker: Option<HighlightPicker>,
    /// The open **quote-into-another-conversation** picker, if any (task 37's
    /// creation UI): the passage, and — once a destination is chosen — the
    /// destination the visibility statement names. Window-local transient
    /// state, like the highlight picker it sits beside.
    pub(crate) quote_destination: Option<references::QuoteDestination>,
    /// Scroll position of the destination picker's bounded, **virtualized**
    /// list — a stored `UniformListScrollHandle`, per the virtualized-list
    /// idiom (it survives re-renders and is what the floating indicator binds
    /// to). Reset to the top each time the picker opens.
    pub(crate) quote_destination_scroll: gpui::UniformListScrollHandle,
    /// One focus handle per **bottom band** — the failure notice, the
    /// denied-follow notice, the cascade notice.
    ///
    /// Each band's Dismiss (and the failure band's Retry/Copy) is a real tab
    /// stop, and dismissing unmounts the band around it, so a keyboard reader
    /// who pressed one was left holding a handle to something nobody paints:
    /// the dead-handle class again, on the surfaces this PR added and the two
    /// beside them (Codex review, PR #280). Tracked on each band's container
    /// while it paints, so the dismiss can ask *containment* before it clears
    /// and hand the keyboard back only from a band that was holding it.
    pub(crate) band_focus: [FocusHandle; 3],
    /// The open right-click menu over one of the space's editors, if any —
    /// window-local transient state, like the band menu and the picker (one
    /// open at a time; see [`context_menu`]).
    pub(crate) context_menu: Option<context_menu::PostContextMenu>,
    /// Supersede slot for a cross-space navigation's home-space resolve. The
    /// work is a pure read whose only effect is opening a window, so a window
    /// closing mid-resolve strands nothing (STATE.md — owner = blast radius).
    pub(crate) navigate_task: Option<Task<()>>,
    /// Posts that rendered for real this frame and therefore want their
    /// incoming-reference index (the source-highlight data). Filled during
    /// render (where visibility is known but `self` is shared) and drained by
    /// `sync_references` at the head of the *next* render, where the `&mut`
    /// borrow exists — the same defer-one-frame idiom as `record_height` and
    /// `slot_bounds`. Bounding the fetch to visible posts is what keeps a long
    /// transcript from spawning one query per row on open; a frame of latency
    /// on a decoration is invisible.
    pub(crate) wants_incoming_refs: RefCell<HashSet<SharedString>>,
    /// The last raw exchange this window asked the Record to open (a trace
    /// row's Record link). Recorded before the `AppGlobal` guard so behavior
    /// tests can assert the deep link without a real Record window — the
    /// `Space::last_submitted_model` seam, applied to a cross-window verb.
    pub(crate) last_record_request: Option<String>,
    /// One read-only editor per in-flight turn, synced to that turn's live
    /// streaming partial each frame and pruned when the turn ends. Keyed by
    /// [`StreamingTurn::seq`] so concurrent turns render side by side.
    pub(crate) streaming_bodies: HashMap<u64, Entity<MarkdownEditorState>>,
    /// Last value pushed into each turn's editor, to skip redundant re-parses.
    pub(crate) streaming_synced: HashMap<u64, String>,

    /// The unsent **drafts** — local UI nodes (never in `posts` until sent),
    /// each its own editor that persists when deselected. A draft attaches to
    /// the persisted tree as a leaf of its `parent` (the post it replies to);
    /// `None` parent is a blank space's root draft. The active one floats as the
    /// composer; the rest render inline (see [`composer`](self)).
    pub(crate) drafts: Vec<Draft>,
    /// The focused draft, if any — the one that adopts the floating composer
    /// behavior. `None` means no composer is open (the resting state of a space
    /// with existing posts; you click a band's "+" to start one).
    pub(crate) active_draft: Option<SharedString>,
    /// Monotonic counter for minting unique draft ids.
    pub(crate) next_draft_seq: usize,
    /// A node to bring onto the selected path on the next render — where the
    /// real effective tree (and its scroll handles) exist. A freshly-created
    /// draft and a freshly-started turn both name a node the current frame's
    /// tree does not carry yet; [`PendingSelect::settle`] says where the page
    /// should come to rest once the branch is selected.
    pub(crate) pending_select: Option<PendingSelect>,
    /// Internal scroll of the floating composer overlay.
    pub(crate) composer_scroll: ScrollHandle,
    pub(crate) composer_prev_off_y: f32,
    /// The floating composer's height as a **fraction of the window** —
    /// window-local state (never persisted to the space, config, or shared
    /// across windows), seeded with [`COMPOSER_MAX_FRACTION`] so every window
    /// opens with the default. Written only by the separator-handle resize
    /// drag; a ratio rather than pixels, so a window resize scales the bar
    /// with it. Survives deactivation — only the *sizing mode* below reverts.
    pub(crate) composer_fraction: f32,
    /// How [`Self::composer_fraction`] applies to the floating bar: `Max`
    /// (grow with content up to it — the resting behavior) or `Exact` (pinned
    /// to it regardless of content — entered by the resize drag, so the bar
    /// can exceed its content). Reverts to `Max` whenever the composer
    /// deactivates or a different draft is selected
    /// ([`Self::reset_composer_sizing`]).
    pub(crate) composer_sizing: composer::ComposerSizing,
    /// An in-flight separator-handle resize drag, if any (see
    /// [`composer::ComposerResizeDrag`]). Tracked window-globally the way a
    /// minimap drag is, so the drag keeps following after the cursor leaves
    /// the thin strip.
    pub(crate) composer_resize: Option<composer::ComposerResizeDrag>,
    /// The previous frame's dock-approach runway — how far the active draft's
    /// would-be dock top still had to travel to reach the float line, saturated
    /// at the approach zone (see [`composer::dock_runway`]). `None` when no
    /// active on-path draft rendered last frame, so a stale runway can never
    /// seed a bogus first step. Consumed once per render by
    /// [`Self::glide_composer_toward_dock`], which scales the composer's
    /// internal scroll by the runway each frame consumes so a scrolled floating
    /// composer's content reaches its own top exactly as the composer docks.
    pub(crate) composer_dock_runway: Option<f32>,
    /// Whether the composer overlay is floating (vs docked), cached from the
    /// last render so the scroll handler can decide session ownership.
    pub(crate) composer_overlayed: Cell<bool>,
    /// Whether the floating composer's content actually overflows its visible
    /// bar (i.e. it's capped at [`COMPOSER_MAX_FRACTION`]), cached from the last
    /// render. A floating composer only *owns* the scroll — and is only itself
    /// scrollable — when this is true; when it's showing at its natural height
    /// (content fits, incl. empty / one line) a wheel over it scrolls the page
    /// underneath, so it can dock, instead of being trapped scrolling nothing.
    pub(crate) composer_scrollable: Cell<bool>,
    /// The composer's natural (unclipped) content height, recorded each frame:
    /// the editor's own laid-out text height **plus** the footnote rail's
    /// measured height (see [`composer_rail_h`](Self::composer_rail_h)) plus
    /// the bottom breath. Everything that sizes the composer — the floating
    /// bar, the docked runway, the minimap — reads this one value, so what the
    /// bar reserves is what the body actually renders.
    pub(crate) composer_content_h: Rc<RefCell<Pixels>>,
    /// Compact metadata and action rows around the active editor. This is
    /// synchronized before document geometry is derived so every composer
    /// consumer includes the same natural height.
    pub(crate) composer_gutters: Cell<ComposerGutterHeights>,
    /// Flow positions bracketing the active composer's footnote rail, written
    /// by the two zero-height probes the composer body places around it (see
    /// [`references::flow_mark`]). Their difference is the rail's **measured**
    /// occupancy — margin, rule, padding and all — which is what the bar
    /// reserves for it, instead of a row-count formula that drifts the moment
    /// the rail's styling changes. With no rail rendered the two coincide, so
    /// the reservation is zero without a special case. Both paint before
    /// `record_height`'s probe, so the sum above is consistent within a frame.
    pub(crate) composer_rail_top: Rc<Cell<f32>>,
    pub(crate) composer_rail_bottom: Rc<Cell<f32>>,
    /// Painted bounds of the composer's in-flow placeholder slot, keyed by the
    /// draft sentinel id — positions the dock and feeds the minimap.
    pub(crate) slot_bounds: Rc<RefCell<HashMap<SharedString, Bounds<Pixels>>>>,
    /// Owner of the current vertical scroll session.
    pub(crate) scroll_owner: Option<ScrollOwner>,
    /// The separator band whose Reply-or-Ask menu is open (by node id), if
    /// any. Window-local; dismissed by click-out, a choice, or Escape.
    pub(crate) band_menu: Option<SharedString>,
    /// The window-local cascade-paused notice, if showing.
    pub(crate) cascade_notice: Option<CascadeNotice>,
    /// The post whose action gutter currently reveals its hover affordances
    /// (Edit / Regenerate), by node id.
    pub(crate) hovered_post: Option<SharedString>,
    /// The post currently being edited in place, if any.
    pub(crate) editing: Option<EditingPost>,
    /// Where the **keyboard** sits inside the conversation — which post, and
    /// whether focus has entered that post's affordance row. `None` means the
    /// conversation holds no keyboard focus (the resting state, and where
    /// Escape at the post level returns to). See [`keyboard`].
    pub(crate) tree_focus: Option<keyboard::TreeFocus>,
    /// The focus handle the **focused post row** tracks. One handle moved from
    /// row to row rather than one per post: gpui reports focus into the
    /// AccessKit tree from whichever node tracks the focused handle, and the
    /// row's `focus_visible` ring reads the same handle — so one handle gives
    /// both, and nothing has to be minted or reaped as the transcript changes.
    pub(crate) post_focus: FocusHandle,
    /// One focus handle **per affordance slot**, tracked by the verbs of
    /// whichever post currently holds the affordance level. A handle per slot
    /// rather than one for "the focused verb", because Tab walks between the
    /// verbs and only a handle *we* own can answer which one it landed on: a
    /// single handle left [`keyboard::TreeFocus`]'s index frozen at the verb
    /// Enter had entered, so the next `Right` cycled from a stale position. It
    /// is a pool, grown to the largest verb row seen, and only the level's own
    /// post tracks from it — two elements claiming one handle report focus
    /// twice in a frame.
    ///
    /// Each handle carries `tab_index(0).tab_stop(true)`: gpui reads tab order
    /// off the *tracked* handle once one exists, and these verbs must stay
    /// ordinary Tab destinations.
    pub(crate) affordance_slots: Vec<FocusHandle>,
    /// **Who** an open transient overlay left holding the keyboard, recorded
    /// each frame one is up — see [`keyboard::sync_tree_focus`], which uses the
    /// falling edge to hand focus *back* to the conversation instead of reading
    /// the borrow as a loss. The handle, not a flag: on the falling edge it is
    /// the difference between "the overlay never gave the keyboard back" (still
    /// focused, element gone — restore) and "something else claimed it in the
    /// meantime" (a menu item that opened a draft and focused its editor —
    /// leave it alone).
    pub(crate) overlay_borrowed_focus: Option<FocusHandle>,
    /// Set when the active draft's editor emits a buffer [`Change`], consumed
    /// by the composer body's `caret_into_view` canvas on the next paint to
    /// scroll the new caret position into the composer's visible viewport
    /// (`composer.rs`). A `Cell` so the paint-phase canvas can clear it without
    /// an entity update; only ever set for the active draft, so an off-screen /
    /// inline edit never triggers a composer scroll.
    pub(crate) composer_caret_scroll_pending: Cell<bool>,
    /// The composer's accessible **value** — `(draft id, text)`, the draft as
    /// assistive technology last read it.
    ///
    /// It is refreshed at two settled moments and at no other: when a
    /// *different* draft becomes the active one (so re-opening a saved draft
    /// reads its real text), and on any frame where the composer does **not**
    /// hold keyboard focus. It therefore never tracks keystrokes: AT re-reads a
    /// focused control's whole value on every change, which would turn typing
    /// into a stutter of the entire draft (audit §4; Zed's own text field
    /// freezes for the same reason).
    pub(crate) composer_aria_value: RefCell<(SharedString, SharedString)>,

    /// Vertical scroll of the whole page.
    pub(crate) page_scroll: ScrollHandle,
    /// The most-negative valid page scroll `y` for the current frame (the
    /// content hard-stops here). Set once per `render` from the real document
    /// height; everything that *positions* content from the scroll offset reads
    /// it via `clamped_scroll_y`, so transient momentum overshoot past the ends
    /// never moves the docked composer / posts / minimap (the flicker fix,
    /// generalized).
    pub(crate) scroll_min_y: Cell<f32>,
    /// The previous frame's **end of written content** — the page scroll `y` at
    /// which the selected branch's last post/streaming leaf sat against the
    /// window bottom, with any trailing draft's speculative runway excluded.
    /// [`Self::follow_streaming_tail`] both tests against it ("is the reader
    /// still at the end?") and scrolls to this frame's value, so the two can
    /// never disagree about where "the end" is.
    pub(crate) follow_anchor: Cell<f32>,
    /// The turn whose streaming leaf the **last rendered frame** had on the
    /// selected path (`layout::selected_turn_seq`). It is a record rather than
    /// a live query because the question is asked after the fact: a completed
    /// turn's leaf is already gone from the tree, so only the last frame can
    /// say whether the reader was parked on it — see
    /// [`Self::follow_completed_turn`].
    pub(crate) selected_turn: Cell<Option<u64>>,
    /// The reader just posted: extend tail-following to the growth that
    /// follows a submit, until the exchange settles. Armed by
    /// [`Self::settle_on_new_post`], cleared in `render` the moment the space
    /// is neither producing nor busy — and by [`Self::activate_draft`], since a
    /// reader who has started composing again owns the viewport. See
    /// [`Self::follow_streaming_tail`] for why the pin is needed at all.
    pub(crate) tail_pin: TailPin,
    /// Test-only record of the slot-relative offset the **docked**
    /// `caret_into_view` branch folded into the caret's document position —
    /// `page_slot_doc_top + editor_top_offset` (`caret_doc_bot - caret_bot`).
    /// The final page-scroll target is gpui-clamped against a frame-lagged
    /// content size and can lag under parallel test load, so tests assert this
    /// frame-independent difference instead, to guard that the docked reveal
    /// accounts for the editor's `POST_PAD_Y` content-top offset within the slot.
    /// Written in `composer::caret_into_view`'s docked arm; read via
    /// `docked_caret_slot_offset_for_test`.
    pub(crate) docked_caret_slot_offset: Cell<f32>,
    /// One horizontal scroller per node that has children, keyed by node id.
    pub(crate) scrolls: HashMap<SharedString, ScrollHandle>,
    /// Branch count per scroller id, refreshed each frame in [`Self::sync_scrolls`]
    /// so the snap-on-release can size the target for whichever scroller *owns*
    /// the gesture — not just the strip the cursor happens to be over at lift.
    pub(crate) scroller_counts: HashMap<SharedString, usize>,
    /// The branch scroller that owns the current *horizontal* gesture — decided
    /// by the first strip to handle a horizontal step and held until the gesture
    /// ends, mirroring the vertical [`ScrollOwner`]. Sibling branches differ in
    /// height, so mid-slide the cursor can drift over a *nested* strip on the
    /// incoming branch; without this lock that nested strip would steal the
    /// gesture (halting the slide or scrolling a sub-branch). Instead the drifted
    /// strip forwards its delta here.
    pub(crate) h_scroll_owner: Option<SharedString>,
    /// The axis the current gesture is locked to (`None` between gestures).
    pub(crate) scroll_axis: Option<ScrollAxis>,
    /// The most recent non-zero horizontal step, used as the release velocity
    /// for the snap's flick decision.
    pub(crate) last_h_delta: Pixels,
    /// The branch scroller currently gliding to a snap point, if any.
    pub(crate) snap: Option<SnapAnim>,
    /// A settled branch + resting x, pinned until the next gesture so trailing
    /// momentum can't drift the page off-branch.
    pub(crate) snap_pin: Option<(SharedString, f32)>,
    /// The page (vertical) glide carrying the reader to a place they asked to
    /// be taken to — a reference, a footnote's source, "See in context". See
    /// [`PageGlide`]; cancelled by any scroll the reader or the tail drives.
    pub(crate) page_glide: Cell<Option<PageGlide>>,
    /// The previous frame's page width, to remap branch offsets on resize.
    pub(crate) last_page_width: Option<Pixels>,

    /// The cached post-height layout (the virtualization core).
    pub(crate) layout: Layout,
    /// Frames remaining to force every on-path post to render real (bypassing
    /// viewport virtualization) so it measures into the layout cache. Armed
    /// whenever an on-path post is unmeasured (cold open, width change that
    /// emptied the cache, freshly added post). While warming, the document's
    /// measured height is established up front, so an off-screen post later
    /// measuring estimate→real can't shift the whole document below it — which
    /// otherwise jumps the dock/float threshold (and the composer's drop shadow
    /// gated on it) and resizes the minimap columns as you scroll toward them.
    /// `Cell` so the `&self` render path can read it.
    pub(crate) warm_remaining: Cell<u8>,
    /// Signature of the minimap's layout inputs, so a reflow/scroll schedules
    /// exactly one catch-up frame.
    pub(crate) minimap_sig: f32,
    pub(crate) minimap_visible: bool,
    pub(crate) minimap_gesturing: bool,
    pub(crate) minimap_hovered: bool,
    pub(crate) minimap_fade_gen: usize,
    pub(crate) minimap_hide_task: Option<Task<()>>,
    /// The minimap container's absolute painted bounds, recorded each frame so a
    /// mousedown/drag can convert a window-space y into a minimap-local y (the
    /// container spans the full viewport height on the right edge).
    pub(crate) minimap_bounds: Rc<Cell<Option<Bounds<Pixels>>>>,
    /// An in-flight scrollbar-style minimap drag, if any. Set on a same-branch
    /// (or handle) mousedown and cleared on mouse-up; while set, window-global
    /// move/up listeners drag `page_scroll` (see [`minimap`]). Locked to the
    /// selected branch for the drag's duration — the snapshotted `scale`/floor
    /// stay valid because the branch (and thus the page height) is fixed.
    pub(crate) minimap_drag: Option<minimap::MinimapDrag>,

    /// A minimal honest error band (e.g. a submit failing before onboarding
    /// exists). Full onboarding is a later, separate window.
    pub(crate) error: Option<String>,

    /// A quiet notice about a **reference** that could not be followed (task
    /// 37): the reader clicked a quote from a conversation they take no part
    /// in. Its own field rather than `error`'s, because a refusal is not a
    /// failure — nothing broke, there is nothing to retry, and the band it
    /// renders in is the muted one.
    pub(crate) reference_notice: Option<SharedString>,

    /// Whether this window's **inspector** (the per-space settings panel) is
    /// open. Per-window by design — two windows on one space are two vantage
    /// points, and the panel is a way of looking, not a property of the space
    /// (STATE.md's scoping table: picker/disclosure state is view state). The
    /// only doors are `Space ▸ Show/Hide Inspector` and ⌥⌘I.
    pub(crate) inspector_open: bool,
    /// The inspector body's own scroll.
    pub(crate) inspector_scroll: ScrollHandle,
    /// The inspector's title field + its `PressEnter`/`Blur` subscription,
    /// minted on first render and re-seeded from the Library index whenever the
    /// stored title moves while the field is unfocused ([`inspector`]).
    pub(crate) inspector_title: Option<(Entity<gpui_component::input::InputState>, Subscription)>,
    /// The title the field was last seeded with, so a re-seed happens exactly
    /// when the space's real title moves.
    pub(crate) inspector_title_seed: Option<SharedString>,
    /// Whether the inspector's router-model dropdown is open.
    pub(crate) inspector_router_picker: bool,
    /// That dropdown's own scroll (reset to the top on each open).
    pub(crate) inspector_picker_scroll: ScrollHandle,
    /// The Participants section's open disclosure — one at a time, because it
    /// *is* the editor (live inputs plus an explicit Save). See
    /// [`inspector_participants`].
    pub(crate) inspector_participant_edit: Option<inspector_participants::ParticipantEdit>,
    /// The open add-a-participant form, if any.
    pub(crate) inspector_participant_add: Option<inspector_participants::ParticipantAdd>,
    /// The open "save these participants as a template" form, if any.
    pub(crate) inspector_template_form: Option<inspector_participants::TemplateForm>,
    /// The open "Invite an agent…" form, if any — task 37's grant, where a
    /// space gives an agent (shared, or one that has to be shared first)
    /// membership as an observer.
    pub(crate) inspector_invite: Option<inspector_participants::InviteForm>,
    /// The invite form's own candidate read. View-owned: the list is only ever
    /// looked at while the form is open, so a window closing mid-read strands
    /// nothing (STATE.md — owner = blast radius).
    pub(crate) inspector_invite_task: Option<Task<()>>,
    /// Which participant model dropdown is open (at most one).
    pub(crate) inspector_participant_picker: Option<inspector_participants::ParticipantPicker>,
    /// That dropdown's own scroll (reset to the top on each open).
    pub(crate) inspector_participant_picker_scroll: ScrollHandle,
    /// Scroll position of the invite form's **virtualized** candidate list —
    /// a stored `UniformListScrollHandle` per the virtualized-list idiom.
    pub(crate) inspector_invite_scroll: gpui::UniformListScrollHandle,

    /// The window title last pushed to the platform (and to the a11y root
    /// node), so an unchanged title never re-enters AppKit every frame. The
    /// title tracks the space's Library title, which arrives after the first
    /// exchange auto-titles it and changes again on rename — hence a render
    /// concern rather than a one-shot at window open.
    pub(crate) window_title: Option<SharedString>,
}

impl SpaceView {
    pub fn new(
        stores: Stores,
        space_id: Option<String>,
        window_input: Entity<WindowInput>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let focus_handle = cx.focus_handle();

        // A brand-new (blank ⌘N) space opens with the composer ready — the
        // cursor at the top of an empty notebook. A reopened space with history
        // opens **without** a composer: you start one by clicking a band's "+".
        let is_blank = space_id.is_none();

        // Get-or-create the shared `Space` through the registry (join-existing).
        let spaces = stores.spaces.clone();
        let space = match space_id {
            Some(id) => spaces.update(cx, |s, cx| s.open(id, cx)),
            None => spaces.update(cx, |s, cx| s.blank(cx)),
        };
        // Tell the entity this window is drawing it. A space is not a singleton
        // on screen, so a cross-window handoff (task 37's quote) has to be able
        // to ask whether this conversation is *already* open, and to address
        // itself to one of the windows that has it. Registered on the entity
        // rather than by space id because a blank ⌘N space is adopted into the
        // registry only once it earns an id — the entity is what stays put.
        let handle = window.window_handle();
        space.update(cx, |space, _| space.attach_window(handle));

        let _subs = vec![
            // Any space change re-derives the render snapshot and re-renders.
            cx.observe(&space, |this: &mut Self, _, cx| {
                this.rebuild(cx);
                cx.notify();
            }),
            cx.subscribe_in(&space, window, Self::on_space_event),
            cx.observe(&window_input, |_, _, cx| cx.notify()),
            cx.observe(&stores.config, |_, _, cx| cx.notify()),
            // The space index carries the title — the window's name (see
            // `sync_window_title`), which arrives only once the first
            // exchange auto-titles the space and changes again on rename.
            // `observe_in` (not plain `observe`) because the title has to be
            // written *before* the frame this notify schedules — see
            // `sync_window_title`.
            cx.observe_in(&stores.spaces, window, |this, _, window, cx| {
                this.sync_window_title(window, cx);
                cx.notify();
            }),
            // The space's participants feed the separator Ask menus, the
            // streaming bylines, and the cascade notice's ask affordances.
            cx.observe(&stores.participants, |_, _, cx| cx.notify()),
            // This space's own settings are the inspector's rows. Every one of
            // this store's announcements is asynchronous — the panel's opening
            // `ensure` load completing or failing, each write's re-read, a bus
            // `Change::Space` refresh — and none of them is accompanied by
            // anything else that would repaint this window, so without this the
            // panel could sit on "Loading…" until an unrelated event redrew it.
            cx.observe(&stores.space_settings, |_, _, cx| cx.notify()),
            // Local models feed the request panel (a load/unload while it's
            // open must re-render — offline, the models store is quiet, so
            // this is the *only* signal that would refresh it) *and* the
            // byline display names, which live in the rebuilt snapshot.
            cx.observe(&stores.local_models, |this: &mut Self, _, cx| {
                this.rebuild(cx);
                cx.notify();
            }),
            // Backend enable/disable flips which groups the panel shows;
            // display names feed the byline snapshot too.
            cx.observe(&stores.backends, |this: &mut Self, _, cx| {
                this.rebuild(cx);
                cx.notify();
            }),
            // The third of the model-picker read set (`model_groups` reads all
            // three): a remote catalog's fetch lands long after the window has
            // drawn, and an open router picker has to gain those options when
            // it does. No `rebuild` — catalogs feed the picker's list, not the
            // transcript snapshot. The other `router_field` consumer (the
            // Space Templates pane) observes the same three.
            cx.observe(&stores.models, |_, _, cx| cx.notify()),
        ];

        let mut this = Self {
            stores,
            space,
            window_input,
            focus_handle,
            _subs,
            posts: Vec::new(),
            bodies: HashMap::new(),
            body_seeds: HashMap::new(),
            body_subs: HashMap::new(),
            post_selection: None,
            highlight_picker: None,
            quote_destination: None,
            quote_destination_scroll: gpui::UniformListScrollHandle::new(),
            band_focus: [cx.focus_handle(), cx.focus_handle(), cx.focus_handle()],
            context_menu: None,
            navigate_task: None,
            wants_incoming_refs: RefCell::new(HashSet::new()),
            last_record_request: None,
            streaming_bodies: HashMap::new(),
            streaming_synced: HashMap::new(),
            drafts: Vec::new(),
            active_draft: None,
            next_draft_seq: 0,
            pending_select: None,
            composer_scroll: ScrollHandle::new(),
            composer_prev_off_y: 0.0,
            composer_fraction: COMPOSER_MAX_FRACTION,
            composer_sizing: composer::ComposerSizing::Max,
            composer_resize: None,
            composer_dock_runway: None,
            composer_overlayed: Cell::new(false),
            composer_scrollable: Cell::new(false),
            composer_content_h: Rc::new(RefCell::new(px(0.))),
            composer_gutters: Cell::new(ComposerGutterHeights::default()),
            composer_rail_top: Rc::new(Cell::new(0.0)),
            composer_rail_bottom: Rc::new(Cell::new(0.0)),
            slot_bounds: Rc::new(RefCell::new(HashMap::new())),
            scroll_owner: None,
            band_menu: None,
            cascade_notice: None,
            hovered_post: None,
            editing: None,
            tree_focus: None,
            post_focus: cx.focus_handle(),
            affordance_slots: Vec::new(),
            overlay_borrowed_focus: None,
            composer_caret_scroll_pending: Cell::new(false),
            composer_aria_value: RefCell::new((SharedString::default(), SharedString::default())),
            page_scroll: ScrollHandle::new(),
            scroll_min_y: Cell::new(0.0),
            follow_anchor: Cell::new(0.0),
            selected_turn: Cell::new(None),
            tail_pin: TailPin::Inactive,
            docked_caret_slot_offset: Cell::new(0.0),
            scrolls: HashMap::new(),
            scroller_counts: HashMap::new(),
            h_scroll_owner: None,
            scroll_axis: None,
            last_h_delta: px(0.),
            snap: None,
            snap_pin: None,
            page_glide: Cell::new(None),
            last_page_width: None,
            layout: Layout::new(),
            warm_remaining: Cell::new(0),
            minimap_sig: f32::NAN,
            minimap_visible: false,
            minimap_gesturing: false,
            minimap_hovered: false,
            minimap_fade_gen: 0,
            minimap_hide_task: None,
            minimap_bounds: Rc::new(Cell::new(None)),
            minimap_drag: None,
            error: None,
            reference_notice: None,
            inspector_open: false,
            inspector_scroll: ScrollHandle::new(),
            inspector_title: None,
            inspector_title_seed: None,
            inspector_router_picker: false,
            inspector_picker_scroll: ScrollHandle::new(),
            inspector_participant_edit: None,
            inspector_participant_add: None,
            inspector_template_form: None,
            inspector_invite: None,
            inspector_invite_task: None,
            inspector_participant_picker: None,
            inspector_participant_picker_scroll: ScrollHandle::new(),
            inspector_invite_scroll: gpui::UniformListScrollHandle::new(),
            window_title: None,
        };
        this.rebuild(cx);
        this.ensure_participants(cx);
        // Name the window before its first frame; the observer keeps it current.
        this.sync_window_title(window, cx);
        if is_blank {
            // The blank notebook: a root draft, focused and ready.
            this.create_draft(None, window, cx);
        } else {
            // No composer yet — focus the root so action dispatch still works.
            window.focus(&this.focus_handle, cx);
        }
        this
    }

    /// Lazily load this space's participant list (the Ask menus + streaming
    /// bylines read it). A no-op until the space has a persisted id — called
    /// again from `on_space_event` once a blank space adopts one.
    fn ensure_participants(&mut self, cx: &mut Context<Self>) {
        if let Some(id) = self.space.read(cx).id().map(str::to_string) {
            self.stores
                .participants
                .update(cx, |p, cx| p.ensure(id, cx));
        }
    }

    // -- Test seams --------------------------------------------------------

    /// The view's focus handle (behavior tests dispatch actions through it).
    pub fn focus_handle(&self) -> FocusHandle {
        self.focus_handle.clone()
    }

    /// The shared space entity.
    pub fn space(&self) -> &Entity<Space> {
        &self.space
    }

    /// The active draft's editor state (tests set its value directly). `None`
    /// when no composer is open.
    #[doc(hidden)]
    pub fn composer_state_for_test(&self) -> Option<Entity<MarkdownEditorState>> {
        let id = self.active_draft.as_ref()?;
        self.drafts
            .iter()
            .find(|d| &d.id == id)
            .map(|d| d.editor.clone())
    }

    /// The **trailing** draft's editor state — the tail composer at the end of
    /// the current branch, whether or not it is the active one. The type-to-
    /// compose jump (task 38) targets exactly this, so its test needs to seed
    /// and read it while nothing is composing.
    #[doc(hidden)]
    pub fn tail_draft_state_for_test(&self) -> Option<Entity<MarkdownEditorState>> {
        self.drafts.last().map(|d| d.editor.clone())
    }

    /// Leave the composing session (the Escape gesture), so a test can reach
    /// the "nothing is composing" state the keyboard model needs.
    #[doc(hidden)]
    pub fn retire_draft_for_test(&mut self, cx: &mut Context<Self>) {
        self.deactivate_active_draft(cx);
    }

    /// The current per-row render snapshot length (tests assert tree shape).
    #[doc(hidden)]
    pub fn post_count_for_test(&self) -> usize {
        self.posts.len()
    }

    /// The number of unsent drafts (tests assert create/retire).
    #[doc(hidden)]
    pub fn draft_count_for_test(&self) -> usize {
        self.drafts.len()
    }

    /// The parents of all current drafts (tests assert tail/fork presence).
    #[doc(hidden)]
    pub fn draft_parents_for_test(&self) -> Vec<Option<String>> {
        self.drafts
            .iter()
            .map(|d| d.parent.as_ref().map(|s| s.to_string()))
            .collect()
    }

    /// The active draft's parent (the post it replies to), if any.
    #[doc(hidden)]
    pub fn active_draft_parent_for_test(&self) -> Option<String> {
        let id = self.active_draft.as_ref()?;
        self.drafts
            .iter()
            .find(|d| &d.id == id)?
            .parent
            .as_ref()
            .map(|s| s.to_string())
    }

    /// Whether a composer (active draft) is currently open.
    #[doc(hidden)]
    pub fn has_active_draft_for_test(&self) -> bool {
        self.active_draft.is_some()
    }

    /// Whether an edit has armed the composer's caret scroll-into-view (the
    /// `caret_into_view` canvas consumes this on the next paint). The geometry
    /// itself is unit-tested via `caret_scroll_offset` in `composer.rs`.
    #[doc(hidden)]
    pub fn caret_scroll_pending_for_test(&self) -> bool {
        self.composer_caret_scroll_pending.get()
    }

    /// The composer's internal vertical scroll offset (`<= 0`; more-negative =
    /// scrolled further down). Tests assert an edit that pushes the caret below
    /// the composer fold scrolls it into view (a negative offset).
    #[doc(hidden)]
    pub fn composer_scroll_offset_y_for_test(&self) -> f32 {
        self.composer_scroll.offset().y.as_f32()
    }

    /// The whole-page vertical scroll offset (`<= 0`; more-negative = scrolled
    /// further down). Tests assert that an edit in a **docked** composer (the
    /// blank ⌘N notebook) whose caret runs below the window scrolls the *page*
    /// to keep the caret visible.
    #[doc(hidden)]
    pub fn page_scroll_offset_y_for_test(&self) -> f32 {
        self.page_scroll.offset().y.as_f32()
    }

    #[doc(hidden)]
    pub fn set_page_scroll_for_test(&self, y: f32) {
        self.set_page_scroll_y(y);
    }

    /// The window-local floating-composer fraction and whether the sizing mode
    /// is pinned (`Exact`) — the resize-drag state machine's observables.
    #[doc(hidden)]
    pub fn composer_fraction_for_test(&self) -> f32 {
        self.composer_fraction
    }

    #[doc(hidden)]
    pub fn composer_sizing_is_exact_for_test(&self) -> bool {
        self.composer_sizing == composer::ComposerSizing::Exact
    }

    /// The floating bar height the current window state yields for a
    /// `window_h`-tall window — the same helper the render, the pre-dock
    /// glide, and the floating pad read.
    #[doc(hidden)]
    pub fn composer_float_bar_h_for_test(&self, window_h: f32) -> f32 {
        self.composer_float_bar_h(px(window_h))
    }

    /// The effective resize floor for this window (see
    /// `composer_min_fraction`).
    #[doc(hidden)]
    pub fn composer_min_fraction_for_test(&self, window_h: f32) -> f32 {
        self.composer_min_fraction(window_h)
    }

    /// Natural composer height (docked), its editor-and-chrome base, and the
    /// compact gutter occupancy used by every geometry consumer.
    #[doc(hidden)]
    pub fn composer_height_contract_for_test(&self) -> (f32, f32, f32) {
        let base = Self::composer_chrome() + self.composer_content_h.borrow().as_f32();
        let gutters = self.composer_gutters.get().total();
        (self.composer_natural_height(), base, gutters)
    }

    /// The floating bar's natural height — the docked one less the compact
    /// byline row, which never floats.
    #[doc(hidden)]
    pub fn composer_floating_natural_height_for_test(&self) -> f32 {
        self.composer_floating_natural_height()
    }

    /// Compact composer occupancy split around the editor.
    #[doc(hidden)]
    pub fn composer_gutter_contract_for_test(&self) -> (f32, f32) {
        let gutters = self.composer_gutters.get();
        (gutters.top, gutters.total())
    }

    /// Drive the separator-handle resize drag without synthesizing mouse
    /// events (the Library-archive precedent: tests call the same methods the
    /// element listeners route to). `begin` grabs the bar at its current
    /// height, exactly as the strip's mousedown does.
    #[doc(hidden)]
    pub fn begin_composer_resize_for_test(
        &mut self,
        pointer_y: f32,
        window_h: f32,
        cx: &mut Context<Self>,
    ) {
        let bar_h = self.composer_float_bar_h(px(window_h));
        self.start_composer_resize(pointer_y, bar_h, window_h, cx);
    }

    #[doc(hidden)]
    pub fn move_composer_resize_for_test(
        &mut self,
        pointer_y: f32,
        window_h: f32,
        cx: &mut Context<Self>,
    ) {
        self.update_composer_resize(pointer_y, window_h, cx);
    }

    #[doc(hidden)]
    pub fn end_composer_resize_for_test(&mut self, cx: &mut Context<Self>) {
        self.end_composer_resize(cx);
    }

    /// Scroll the whole page to the top. Tests use this to push an active tail
    /// draft's in-flow slot far below the fold, so the composer *floats* (capped
    /// at [`COMPOSER_MAX_FRACTION`]) rather than docking — the configuration in
    /// which the composer owns its own scroll.
    #[doc(hidden)]
    pub fn scroll_page_to_top_for_test(&self) {
        self.set_page_scroll_y(0.);
    }

    /// Scroll the whole page by `dy` (negative = toward the document end),
    /// clamped to the last rendered frame's valid range. Tests use it to walk
    /// the page toward the composer's dock threshold in increments, asserting
    /// the pre-dock glide at each step.
    #[doc(hidden)]
    pub fn scroll_page_by_for_test(&self, dy: f32) {
        let y = (self.page_scroll.offset().y.as_f32() + dy).clamp(self.scroll_min_y.get(), 0.0);
        self.set_page_scroll_y(y);
    }

    /// Scroll the whole page to the end of the current document — the position
    /// tail-following keeps a streaming turn pinned to. Tests use it to park
    /// the reader at the tail before deltas arrive.
    #[doc(hidden)]
    pub fn scroll_page_to_end_for_test(&self) {
        self.set_page_scroll_y(self.scroll_min_y.get());
    }

    /// Whether the post-submit tail pin is armed — the widened follow gate that
    /// carries a just-posted exchange from the save to its first delta (see
    /// [`Self::follow_streaming_tail`]).
    #[doc(hidden)]
    pub fn tail_pin_for_test(&self) -> bool {
        self.tail_pin.active()
    }

    /// Whether the post-submit pin is still allowed to override the reader's
    /// observed position while initial measured layout converges.
    #[doc(hidden)]
    pub fn tail_pin_forced_for_test(&self) -> bool {
        self.tail_pin.forced()
    }

    /// Scroll the page as the reader's own wheel does — through the takeover
    /// seam (`note_scroll_activity`), not a bare offset write — so tests can
    /// assert what reader-driven motion demotes.
    #[doc(hidden)]
    pub fn reader_scroll_page_by_for_test(&mut self, dy: f32, cx: &mut Context<Self>) {
        self.scroll_page_by_for_test(dy);
        self.note_scroll_activity(gpui::TouchPhase::Moved, true, cx);
    }

    /// Whether the minimap is currently shown (tests assert scroll reveals it).
    #[doc(hidden)]
    pub fn set_minimap_visible_for_test(&mut self, visible: bool, cx: &mut Context<Self>) {
        self.minimap_visible = visible;
        cx.notify();
    }

    #[doc(hidden)]
    pub fn minimap_visible_for_test(&self) -> bool {
        self.minimap_visible
    }

    /// The most-negative valid page scroll `y` for the last rendered frame —
    /// `0.0` means the content exactly fits (nothing to scroll). Tests assert a
    /// blank space's sole composer reserves no phantom scroll.
    #[doc(hidden)]
    pub fn scroll_min_y_for_test(&self) -> f32 {
        self.scroll_min_y.get()
    }

    /// The destination of the navigation glide in flight, if any (`None` when
    /// the page is at rest). Drives the animated-navigation regressions.
    #[doc(hidden)]
    pub fn page_glide_target_for_test(&self) -> Option<f32> {
        self.page_glide.get().map(|g| g.to_y)
    }

    /// Advance the navigation glide to progress `t` — the frame loop's body
    /// without its clock (no test dispatcher pumps `on_next_frame`).
    #[doc(hidden)]
    pub fn drive_page_glide_for_test(&mut self, t: f32) {
        self.apply_page_glide(t);
    }

    /// The page scroll `y` a following reader comes to rest at — the end of the
    /// **written** content (see `follow_streaming_tail`). Equals
    /// `scroll_min_y_for_test` except while a draft trails the selected path.
    #[doc(hidden)]
    pub fn content_end_for_test(&self) -> f32 {
        self.follow_anchor.get()
    }

    /// The recorded top (window-space y) of the minimap container — the origin a
    /// mousedown's window-y is measured from to get a minimap-local y. Must be
    /// the container's real top (≈ 0 for a window-filling space view), not the
    /// bottom of the stacked rows. Tests guard against the drag jumping to the
    /// top because a mis-recorded origin made every press read as a track press.
    #[doc(hidden)]
    pub fn minimap_bounds_top_for_test(&self) -> f32 {
        self.minimap_bounds
            .get()
            .map(|b| b.origin.y.as_f32())
            .unwrap_or(-1.0)
    }

    /// The slot-relative offset the docked `caret_into_view` folded into the
    /// caret's document position (`page_slot_doc_top + editor_top_offset`). See
    /// [`Self::docked_caret_slot_offset`].
    #[doc(hidden)]
    pub fn docked_caret_slot_offset_for_test(&self) -> f32 {
        self.docked_caret_slot_offset.get()
    }

    /// The document's top reserve — so a test can state a document position
    /// relative to it instead of restating the constant.
    #[doc(hidden)]
    pub fn doc_reserve_for_test(&self) -> f32 {
        self.doc_reserve()
    }

    #[doc(hidden)]
    pub fn runway_height_for_test(&self, window_h: f32) -> f32 {
        self.runway_height(px(window_h))
    }

    /// The inline slot height for one inactive draft. This reads the same
    /// measured-height-plus-runway-floor contract as document layout.
    #[doc(hidden)]
    pub fn inactive_draft_height_for_test(&self, index: usize, window_h: f32) -> Option<f32> {
        let draft = self.drafts.get(index)?;
        (self.active_draft.as_ref() != Some(&draft.id))
            .then(|| self.inactive_draft_height(&draft.id, px(window_h)))
    }

    #[doc(hidden)]
    pub fn composer_chrome_for_test(&self) -> f32 {
        Self::composer_chrome()
    }

    /// The compact composer's bottom action-bar occupancy — the room the
    /// editor's viewport ends above, and the surface the footnote rail must
    /// clear.
    #[doc(hidden)]
    pub fn compact_action_occupancy_for_test(&self, page_width: f32, rem_size: f32) -> f32 {
        match layout::page_layout(px(page_width)).gutters {
            layout::GutterPlacement::Sides => 0.0,
            layout::GutterPlacement::Stacked => layout::compact_action_bar_h(px(rem_size)),
        }
    }

    /// How many posts currently have a *measured* (not estimated) height in the
    /// layout cache — equals the post count once the warm pass has run.
    #[doc(hidden)]
    pub fn measured_post_count_for_test(&self) -> usize {
        (0..self.posts.len())
            .filter(|&i| {
                self.layout
                    .measured(&model::node_id(&self.posts, i))
                    .is_some()
            })
            .count()
    }

    /// Open a draft (a band's "+" / blank-page composer); `None` parent is a
    /// root draft. Tests can't synthesize the click.
    #[doc(hidden)]
    pub fn create_draft_for_test(
        &mut self,
        parent: Option<String>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.create_draft(parent.map(Into::into), window, cx);
    }

    /// Deactivate the active draft (Escape), for tests.
    #[doc(hidden)]
    pub fn deactivate_for_test(&mut self, cx: &mut Context<Self>) {
        self.deactivate_active_draft(cx);
    }

    /// Whether ⌥ is held per this window's `WindowInput` (tests assert the
    /// root's modifiers listener actually reaches the shared entity).
    #[doc(hidden)]
    pub fn alt_held_for_test(&self, cx: &gpui::App) -> bool {
        self.window_input.read(cx).alt_held()
    }

    /// The action id of the in-progress inline edit, if any.
    #[doc(hidden)]
    pub fn editing_action_id_for_test(&self) -> Option<String> {
        self.editing.as_ref().map(|e| e.action_id.to_string())
    }

    /// A persisted post's body editor, by node (action) id.
    #[doc(hidden)]
    pub fn post_body_editor_for_test(&self, node_id: &str) -> Option<Entity<MarkdownEditorState>> {
        self.bodies.get(node_id).cloned()
    }

    /// The last raw exchange a trace row asked the Record to open. Recorded
    /// before the `AppGlobal` guard, so a stub-store test sees the deep link
    /// without a real Record window.
    #[doc(hidden)]
    pub fn last_record_request_for_test(&self) -> Option<&str> {
        self.last_record_request.as_deref()
    }

    /// Drive the post-hover state (tests can't synthesize pointer moves).
    #[doc(hidden)]
    pub fn set_post_hover_for_test(&mut self, id: &str, hovering: bool, cx: &mut Context<Self>) {
        self.set_post_hover(&SharedString::from(id.to_string()), hovering, cx);
    }

    /// The node id of the post whose hover affordances are revealed.
    #[doc(hidden)]
    pub fn hovered_post_for_test(&self) -> Option<String> {
        self.hovered_post.as_ref().map(|s| s.to_string())
    }

    /// The recovery-notice error text, if the notice is showing.
    #[doc(hidden)]
    pub fn error_for_test(&self) -> Option<String> {
        self.error.clone()
    }

    /// The node id whose separator band menu is open, if any.
    #[doc(hidden)]
    pub fn band_menu_for_test(&self) -> Option<String> {
        self.band_menu.as_ref().map(|s| s.to_string())
    }

    /// Open/close a band's Reply-or-Ask menu directly (tests can't synthesize
    /// the "+" click).
    #[doc(hidden)]
    pub fn set_band_menu_for_test(&mut self, node_id: Option<&str>, cx: &mut Context<Self>) {
        self.band_menu = node_id.map(|s| SharedString::from(s.to_string()));
        cx.notify();
    }

    /// The cascade-paused notice `(depth, limit, target_action_id)`, if showing.
    #[doc(hidden)]
    pub fn cascade_notice_for_test(&self) -> Option<(i64, i64, String)> {
        self.cascade_notice
            .as_ref()
            .map(|n| (n.depth, n.limit, n.target_action_id.clone()))
    }

    /// The node id of the currently-selected leaf — where `effective_tree`
    /// attaches the synthetic streaming node. Drives the branched-retry
    /// regression (retry must select the failed post's branch).
    #[doc(hidden)]
    pub fn selected_leaf_for_test(&self, window: &Window) -> Option<String> {
        let page_width = self.page_width(window);
        let roots = model::build_tree(&self.posts);
        self.selected_leaf_id(&roots, page_width)
            .map(|s| s.to_string())
    }

    /// The selected path through the **effective** tree (streaming leaves and
    /// drafts included), root → leaf. Drives the ask/retry branch-selection
    /// regressions: a new turn's streaming leaf must be *on* this path, which
    /// the post-only tree above cannot say.
    #[doc(hidden)]
    pub fn selected_effective_path_for_test(&self, window: &Window, cx: &gpui::App) -> Vec<String> {
        let page_width = self.page_size(window).width;
        let turns = self.stream_overlays(cx);
        let tree = self.effective_tree(page_width, &turns);
        self.selected_levels(&tree, page_width)
            .into_iter()
            .map(|(sibs, active)| sibs[active].id.to_string())
            .collect()
    }

    /// Select an effective-tree node without rendering another frame, modeling
    /// branch navigation racing a turn-completion event.
    #[doc(hidden)]
    pub fn select_effective_path_for_test(
        &mut self,
        node_id: &str,
        window: &Window,
        cx: &gpui::App,
    ) {
        let page_width = self.page_size(window).width;
        let turns = self.stream_overlays(cx);
        let tree = self.effective_tree(page_width, &turns);
        self.select_path_to(&tree, node_id, page_width);
    }

    /// The branch dot's `on_click` body (`post.rs` → `glide_to_branch`),
    /// callable without the event plumbing — so a test can sequence a turn
    /// completion *before* the frame that would follow the click. That a real
    /// click reaches this handler is covered by
    /// `space_branch_dot_takes_the_page_from_a_glide_in_flight`, which clicks
    /// the painted dot.
    #[doc(hidden)]
    pub fn click_branch_dot_for_test(
        &mut self,
        node_id: &str,
        index: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let page_width = self.page_size(window).width;
        self.glide_to_branch(node_id.to_string().into(), index, page_width, window, cx);
    }

    /// A post's `(reasoning, expanded)` from the render snapshot.
    #[doc(hidden)]
    pub fn post_reasoning_for_test(&self, i: usize) -> Option<(String, bool)> {
        let post = self.posts.get(i)?;
        post.reasoning
            .as_ref()
            .map(|r| (r.to_string(), post.reasoning_expanded))
    }

    /// How many times the layout height cache has been invalidated (cleared).
    /// A resize that doesn't change the reading-column width must not bump this
    /// — that's the resize-jitter fix (the cache is keyed on `body_width`).
    #[doc(hidden)]
    pub fn layout_clears_for_test(&self) -> u32 {
        self.layout.clears()
    }

    // -- Quoted references (test seams) ------------------------------------

    /// Place a selection in post `node_id`'s read-only body directly (no
    /// pointer geometry) and record it as the quotable selection, exactly as
    /// the editor's `SelectionChanged` subscription would.
    #[doc(hidden)]
    pub fn select_in_post_for_test(
        &mut self,
        node_id: &str,
        range: std::ops::Range<usize>,
        cx: &mut Context<Self>,
    ) {
        let id = SharedString::from(node_id.to_string());
        let Some(editor) = self.bodies.get(&id).cloned() else {
            return;
        };
        editor.update(cx, |e, cx| {
            e.apply_event_for_test(
                gpui_markdown_editor::EditorEvent::SetSelection(
                    gpui_markdown_editor::Selection::range(range.start, range.end),
                ),
                cx,
            );
        });
        self.note_body_selection(&id, cx);
    }

    /// Whether a quotable post selection currently exists — what gates the
    /// Edit menu's Quote items (their handlers are registered only while it is
    /// true, so macOS greys them otherwise).
    #[doc(hidden)]
    pub fn has_post_selection_for_test(&self) -> bool {
        self.post_selection.is_some()
    }

    /// The active draft's pending references as `(ordinal, snippet)` pairs, in
    /// ordinal order.
    #[doc(hidden)]
    pub fn active_draft_references_for_test(&self) -> Vec<(u64, String)> {
        let Some(active) = self.active_draft.as_ref() else {
            return Vec::new();
        };
        let Some(draft) = self.drafts.iter().find(|d| &d.id == active) else {
            return Vec::new();
        };
        let mut refs: Vec<_> = draft
            .references
            .iter()
            .map(|r| (r.ordinal, r.snippet.to_string()))
            .collect();
        refs.sort_by_key(|(o, _)| *o);
        refs
    }

    /// The active draft's id, if any.
    #[doc(hidden)]
    pub fn active_draft_id_for_test(&self) -> Option<String> {
        self.active_draft.as_ref().map(|s| s.to_string())
    }

    /// The reference ordinals the current edit session has marked for removal.
    #[doc(hidden)]
    pub fn edit_removals_for_test(&self) -> Vec<i64> {
        self.editing
            .as_ref()
            .map(|e| e.removed_references.clone())
            .unwrap_or_default()
    }

    /// The highlight ranges painted on post `i`'s body, as
    /// `(buffer range, incoming-reference key)`.
    #[doc(hidden)]
    pub fn highlight_ranges_for_test(
        &self,
        i: usize,
        cx: &gpui::App,
    ) -> Vec<(std::ops::Range<usize>, u64)> {
        self.highlight_ranges(i, cx)
    }

    /// The open source-highlight picker's choices, as `(action id, label)`.
    #[doc(hidden)]
    pub fn highlight_picker_for_test(&self) -> Option<Vec<(String, String)>> {
        self.highlight_picker.as_ref().map(|p| {
            p.choices
                .iter()
                .map(|(a, _, label)| (a.clone(), label.to_string()))
                .collect()
        })
    }

    /// Drive a highlight click on post `node_id` with the given keys (the
    /// editor's `on_highlight_click` payload).
    #[doc(hidden)]
    pub fn click_highlight_for_test(
        &mut self,
        node_id: &str,
        keys: &[u64],
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.on_highlight_click(SharedString::from(node_id.to_string()), keys, window, cx);
    }

    // -- Snapshot & model resolution --------------------------------------

    /// Rebuild the per-row render snapshot from the shared `Space`'s transcript.
    /// Called on every space change (cheap `SharedString` projection) and at
    /// construction. Drafts are local UI state, but their reply antecedent is a
    /// reference *into* the transcript, so it is rethreaded here — see
    /// [`Self::rethread_drafts`]. Keyboard tree focus names a post the same
    /// way and is forwarded by the same rule — see [`Self::retarget_tree_focus`].
    pub(crate) fn rebuild(&mut self, cx: &mut Context<Self>) {
        let messages: Vec<crate::space::ChatMessageView> = self.space.read(cx).messages().to_vec();
        let posts: Vec<PostData> = messages
            .iter()
            .map(|m| {
                // Assistant rows carry the raw model selection id as their
                // participant label; split it into the human display pair
                // (model name over backend name) for the gutter. Everything
                // else ("You", "Error") passes through with no sub-line.
                let (byline, byline_backend) = if m.message.role == "assistant" {
                    let (name, backend) = self.model_display(&m.byline, cx);
                    (name, Some(backend))
                } else {
                    (SharedString::from(m.byline.clone()), None)
                };
                post_data_from(m, byline, byline_backend)
            })
            .collect();
        self.rethread_drafts(&posts);
        self.retarget_tree_focus(&posts);
        self.posts = posts;
    }

    /// Carry every draft's reply antecedent across a generation change of the
    /// post it replies to.
    ///
    /// **Reply threading follows item identity** (workspace `AGENTS.md`: an
    /// action id is causality, an item id is the intended logical flow) — but a
    /// draft names its parent by *action* id, because that is the node id the
    /// tree renders. When that generation is superseded — an edit committed
    /// from another window on the same shared `Space`, or this window's own
    /// Edit session, which *keeps* a non-empty fork draft — the reloaded
    /// transcript carries only the item's new tip, and the draft's parent names
    /// a post that is no longer there. Everything downstream then quietly
    /// mis-threads: [`Self::effective_tree`] re-attaches the draft as its own
    /// root, and `send_active_draft` (which only passes a parent it can find
    /// among the current posts) drops the antecedent entirely — so the post
    /// landed **durably** at the space tail instead of beside its sibling on
    /// the parent's branch.
    ///
    /// The cure is to keep the draft pointing at the *item*: resolve the
    /// vanished action id through the outgoing snapshot to its item, then
    /// forward it to that item's current tip in the incoming one. Only drafts
    /// whose parent actually vanished are touched (the common case does no
    /// work), and an id that resolves to nothing is left alone — an honestly
    /// orphaned draft, exactly as before.
    fn rethread_drafts(&mut self, next: &[PostData]) {
        if self.drafts.is_empty() {
            return;
        }
        let live: HashSet<&str> = next.iter().filter_map(|p| p.action_id.as_deref()).collect();
        let stale: Vec<SharedString> = self
            .drafts
            .iter()
            .filter_map(|d| d.parent.clone())
            .filter(|p| !live.contains(p.as_ref()))
            .collect();
        if stale.is_empty() {
            return;
        }
        for parent in stale {
            // The superseded generation's item, from the snapshot it was last
            // seen in, then that item's current tip in the fresh one.
            let Some(item) = self
                .posts
                .iter()
                .find(|p| p.action_id.as_deref() == Some(parent.as_ref()))
                .and_then(|p| p.item_id.clone())
            else {
                continue;
            };
            let Some(tip) = next
                .iter()
                .find(|p| p.item_id.as_deref() == Some(item.as_ref()))
                .and_then(|p| p.action_id.clone())
            else {
                continue;
            };
            for draft in &mut self.drafts {
                if draft.parent.as_deref() == Some(parent.as_ref()) {
                    draft.parent = Some(tip.clone());
                }
            }
        }
    }

    /// Split a model selection id into its human display pair: the model's
    /// display name and its backend's display name — `"Gemma 4 E2B"` over
    /// `"Local"` instead of `gemma-4-E2B_q4_0-it@local`. Engine-served
    /// models resolve through the local snapshot's display names; catalog
    /// models keep their wire id as the name (it *is* the published name).
    /// Falls back to the raw parts when nothing matches, so an unknown or
    /// since-deleted model still renders honestly.
    pub fn model_display(&self, selection: &str, cx: &gpui::App) -> (SharedString, SharedString) {
        let mref = eidola_app_core::parse_model_ref(selection);
        let backend_name = self
            .stores
            .backends
            .read(cx)
            .get(&mref.backend_id)
            .map(|b| b.display_name.clone())
            .unwrap_or_else(|| match mref.backend_id.as_str() {
                // The singletons' seeded names, for scenes where the
                // registry snapshot hasn't loaded (or stub fixtures).
                eidola_app_core::EIDOLA_BACKEND_ID => "Eidola".to_string(),
                eidola_app_core::LOCAL_BACKEND_ID => "Local".to_string(),
                other => other.to_string(),
            });
        let local = self.stores.local_models.read(cx);
        let model_name = local
            .models()
            .iter()
            .chain(local.external().iter().flat_map(|b| b.models.iter()))
            .find(|m| m.id == selection)
            .map(|m| m.display_name.clone())
            .unwrap_or_else(|| mref.model.clone());
        (model_name.into(), backend_name.into())
    }

    /// The parents a **tail draft** attaches to: every current branch leaf (a
    /// post nothing else replies to), or the root (`None`) when the space is
    /// empty. A *fork* draft attaches to a non-leaf (a post with a committed
    /// reply) and is therefore **not** in this set — which is how
    /// [`Self::retire_active_draft`] / [`Self::sync_tail_drafts`] tell the
    /// always-present tail composer from a transient branch.
    pub(crate) fn tail_parents(&self) -> Vec<SharedString> {
        if self.posts.is_empty() {
            return Vec::new();
        }
        let with_child: HashSet<&str> = self
            .posts
            .iter()
            .filter_map(|p| p.parent_action_id.as_deref())
            .collect();
        self.posts
            .iter()
            .filter_map(|p| p.action_id.as_deref())
            .filter(|aid| !with_child.contains(aid))
            .map(SharedString::from)
            .collect()
    }

    /// Whether a draft replying to `parent` is a tail draft (its parent is a
    /// current leaf, or it's the blank-space root draft).
    pub(crate) fn is_tail_parent(&self, parent: Option<&str>) -> bool {
        match parent {
            None => self.posts.is_empty(),
            Some(p) => self.tail_parents().iter().any(|t| t == p),
        }
    }

    /// The space's **agent** participants — `(id, label)` pairs feeding the
    /// separator Ask menus, the cascade notice's ask affordances, and the
    /// streaming bylines. Empty until the space has a persisted id and its
    /// participant list has loaded.
    pub(crate) fn space_agents(&self, cx: &gpui::App) -> Vec<(String, String)> {
        let Some(space_id) = self.space.read(cx).id() else {
            return Vec::new();
        };
        self.stores
            .participants
            .read(cx)
            .list(space_id)
            .iter()
            .filter(|p| p.kind == "agent")
            .map(|p| (p.id.clone(), p.label.clone()))
            .collect()
    }

    /// The display label for a streaming turn's responding participant.
    /// `None` (a stub/synthetic turn) and unknown ids fall back honestly.
    pub(crate) fn participant_label(
        &self,
        participant_id: Option<&str>,
        cx: &gpui::App,
    ) -> SharedString {
        let Some(pid) = participant_id else {
            return "Eidola".into();
        };
        self.space_agents(cx)
            .into_iter()
            .find(|(id, _)| id == pid)
            .map(|(_, label)| SharedString::from(label))
            .unwrap_or_else(|| "Eidola".into())
    }

    /// Whether a streaming turn is waiting on its **engine**, not on the model:
    /// the responding participant's effective model is engine-served (the
    /// managed `local` store or a `llamacpp` backend) and that model is
    /// currently warming (`LocalModelStatus::Loading`).
    ///
    /// Correlated entirely in the GUI, from two snapshots we already hold and
    /// already re-render on — the participant's `model_ref` (`ParticipantsStore`)
    /// and the engine's status (`LocalModelsStore`, refreshed on
    /// `Change::LocalModels`, which app-core emits the moment a request-triggered
    /// load reserves its engine). A turn against a remote backend, an unknown
    /// participant, or a stub turn with no participant is never "loading".
    pub(crate) fn turn_engine_is_warming(
        &self,
        participant_id: Option<&str>,
        cx: &gpui::App,
    ) -> bool {
        let Some(pid) = participant_id else {
            return false;
        };
        let Some(space_id) = self.space.read(cx).id() else {
            return false;
        };
        let Some(model) = self
            .stores
            .participants
            .read(cx)
            .list(space_id)
            .iter()
            .find(|p| p.id == pid)
            .and_then(|p| p.model_ref.clone())
        else {
            return false;
        };
        let want = eidola_app_core::parse_model_ref(&model);
        let local = self.stores.local_models.read(cx);
        local
            .models()
            .iter()
            .chain(local.external().iter().flat_map(|b| b.models.iter()))
            .filter(|m| {
                let have = eidola_app_core::parse_model_ref(&m.id);
                have.model == want.model && have.backend_id == want.backend_id
            })
            .any(|m| matches!(m.status, eidola_app_core::LocalModelStatus::Loading))
    }

    /// React to a semantic `SpaceEvent`: re-snapshot + re-render, surface a
    /// typed failure as the recovery notice (and a paused cascade as its
    /// quiet notice), and clear the error on success.
    fn on_space_event(
        &mut self,
        _space: &Entity<Space>,
        event: &SpaceEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match event {
            SpaceEvent::MessagesChanged | SpaceEvent::StreamDelta => {
                self.rebuild(cx);
            }
            SpaceEvent::StreamEnded => {
                // Clear the recovery notice only when the space has nothing left
                // to recover. The notice's lifetime is owned by the Space's
                // `failed_turn` record — NOT by whichever event fires last — so a
                // *sibling* turn of a fan-out succeeding must never hide a still-
                // recorded failed turn's Retry. Without this guard the sibling's
                // `StreamEnded` blanked `self.error` while `failed_turn` survived,
                // silently removing the user's only recovery path (the Retry
                // renders inside the notice). It persists until that turn is
                // retried or explicitly dismissed.
                if self.space.read(cx).failed_turn().is_none() {
                    self.error = None;
                }
                self.rebuild(cx);
                // A blank space just adopted its id — its participants are
                // loadable now (the Ask menus need them).
                self.ensure_participants(cx);
            }
            SpaceEvent::TurnEnded {
                seq,
                response_action_id,
            } => {
                // A selection aimed at this turn's streaming leaf — pending, or
                // the branch the reader is parked on — has to follow the turn
                // onto the post it wrote; the leaf is already gone (see
                // `follow_completed_turn`).
                self.follow_completed_turn(*seq, response_action_id.as_deref());
            }
            SpaceEvent::Failed(e) => {
                self.error = Some(error_copy(e));
                self.rebuild(cx);
                self.ensure_participants(cx);
            }
            SpaceEvent::CascadePaused {
                depth,
                limit,
                target_action_id,
            } => {
                self.cascade_notice = Some(CascadeNotice {
                    depth: *depth,
                    limit: *limit,
                    target_action_id: target_action_id.clone(),
                });
            }
        }
        cx.notify();
    }

    // -- Editor & scroller bookkeeping (run in render, has `window`) -------

    /// Ensure a read-only editor state exists for every persisted post, with
    /// its current content; create the composer/streaming editors are already
    /// owned. Prune editors for posts that are gone.
    fn sync_bodies(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let live: HashSet<SharedString> = (0..self.posts.len())
            .map(|i| model::node_id(&self.posts, i))
            .collect();

        for i in 0..self.posts.len() {
            let id = model::node_id(&self.posts, i);
            let content = self.posts[i].content.clone();
            match self.bodies.get(&id) {
                Some(editor) => {
                    // Re-seed an existing editor only when the *post's*
                    // content changed (an edit/regenerate replaced it in
                    // place) — compared against what we last seeded, never
                    // against the editor's live buffer (see `body_seeds`) —
                    // and never clobber the editor holding an in-progress
                    // inline edit: its divergence from the persisted content
                    // *is* the edit.
                    let is_editing = self.editing.as_ref().map(|e| &e.node_id) == Some(&id);
                    if !is_editing && self.body_seeds.get(&id) != Some(&content) {
                        editor.update(cx, |e, cx| e.set_value(content.to_string(), cx));
                        self.body_seeds.insert(id.clone(), content);
                    }
                }
                None => {
                    let editor = cx.new(|cx| {
                        let mut s = MarkdownEditorState::new(window, cx);
                        s.set_value(content.to_string(), cx);
                        s
                    });
                    // Track this post's selection: a read-only body is where
                    // a quote is *made*, so its `SelectionChanged` is the
                    // only signal that gates the Edit menu's Quote items.
                    let sub_id = id.clone();
                    let sub = cx.subscribe(&editor, move |this, _editor, event, cx| {
                        if matches!(
                            event,
                            gpui_markdown_editor::MarkdownEditorEvent::SelectionChanged
                        ) {
                            this.note_body_selection(&sub_id, cx);
                        }
                    });
                    self.body_subs.insert(id.clone(), sub);
                    self.bodies.insert(id.clone(), editor);
                    self.body_seeds.insert(id.clone(), content);
                }
            }
        }
        self.bodies.retain(|id, _| live.contains(id));
        self.body_seeds.retain(|id, _| live.contains(id));
        self.body_subs.retain(|id, _| live.contains(id));
        // A selection whose post left the transcript is no longer quotable.
        if let Some(sel) = &self.post_selection
            && !live.contains(&sel.node_id)
        {
            self.post_selection = None;
        }
        // An edit session whose post vanished (transcript reshaped under it)
        // has nothing to commit into — drop it with the editor.
        if let Some(ed) = &self.editing
            && !live.contains(&ed.node_id)
        {
            self.editing = None;
        }
        // Keep height-cache entries for live posts, live drafts, and any
        // in-flight streaming turn (their ids share the streaming prefix).
        let draft_ids: HashSet<SharedString> = self.drafts.iter().map(|d| d.id.clone()).collect();
        self.layout.retain(&|id| {
            live.contains(id)
                || draft_ids.contains(id)
                || id.starts_with(model::STREAMING_ID_PREFIX)
        });
    }

    /// Ensure a horizontal `ScrollHandle` exists for every node that has
    /// children (a scroller), plus the implicit top-level scroller when there's
    /// more than one root. Prune handles for scrollers that no longer exist.
    fn sync_scrolls(&mut self, roots: &[TreeNode]) {
        let mut live: HashSet<SharedString> = HashSet::new();
        self.scroller_counts.clear();
        if roots.len() > 1 {
            live.insert(model::ROOT_SCROLLER_ID.into());
            self.scroller_counts
                .insert(model::ROOT_SCROLLER_ID.into(), roots.len());
        }
        collect_scrollers(roots, &mut live, &mut self.scroller_counts);
        for id in &live {
            self.scrolls.entry(id.clone()).or_default();
        }
        self.scrolls.retain(|id, _| live.contains(id));
    }

    /// The in-flight turns' render inputs: `(seq, target_action_id)` per turn,
    /// in start order. Cheap clone of ids only.
    pub(crate) fn stream_overlays(&self, cx: &gpui::App) -> Vec<(u64, Option<SharedString>)> {
        self.space
            .read(cx)
            .streams()
            .iter()
            .map(|t| (t.seq, t.target_action_id.clone().map(SharedString::from)))
            .collect()
    }

    /// The effective render forest: the persisted post tree, plus one
    /// streaming leaf per in-flight turn attached at its **target** post (in
    /// start order — concurrent replies to one post land as timestamp-ordered
    /// siblings; a target-less synthetic turn falls back to the selected
    /// leaf), plus every draft attached as a leaf of its parent post (`None`
    /// parent → a root draft). Drafts attach after a node's persisted children
    /// and after any streaming leaf, in `self.drafts` order, so a draft's
    /// branch index is deterministic.
    fn effective_tree(
        &self,
        page_width: Pixels,
        turns: &[(u64, Option<SharedString>)],
    ) -> Vec<TreeNode> {
        let mut roots = model::build_tree(&self.posts);
        for (seq, target) in turns {
            let overlay = TreeNode::leaf(NodeSrc::Streaming(*seq), model::streaming_node_id(*seq));
            let attach_at = target
                .clone()
                .filter(|t| model::node_ref(&roots, t).is_some())
                .or_else(|| {
                    self.selected_leaf_id(&roots, page_width)
                        .filter(|t| model::node_ref(&roots, t).is_some())
                });
            match attach_at {
                Some(t) => {
                    model::attach_overlay(&mut roots, &t, overlay);
                }
                None => roots.push(overlay),
            }
        }
        for d in &self.drafts {
            let overlay = TreeNode::leaf(NodeSrc::Draft, d.id.clone());
            match &d.parent {
                Some(p) if model::node_ref(&roots, p).is_some() => {
                    model::attach_overlay(&mut roots, p, overlay);
                }
                _ => roots.push(overlay),
            }
        }
        roots
    }
}

/// The per-frame page-scroll delta (px) for a selection drag whose pointer is
/// at window-y `my` in a viewport `h` tall. Zero in the neutral middle; near
/// the top margin returns a positive delta (scroll the page toward its start —
/// a less-negative offset); near the bottom a negative one (toward the end).
/// The magnitude ramps linearly from zero at a margin's inner edge to
/// `max_speed` at the viewport edge (and holds `max_speed` past the edge, where
/// a drag pushed off-window sits). Pure so the direction/ramp/clamp is tested
/// without a window.
pub(crate) fn selection_autoscroll_delta(my: f32, h: f32, margin: f32, max_speed: f32) -> f32 {
    if margin <= 0.0 || h <= 0.0 {
        return 0.0;
    }
    if my < margin {
        // Depth into the top band; past the top edge (`my < 0`) holds full speed.
        let depth = ((margin - my) / margin).clamp(0.0, 1.0);
        depth * max_speed
    } else if my > h - margin {
        let depth = ((my - (h - margin)) / margin).clamp(0.0, 1.0);
        -depth * max_speed
    } else {
        0.0
    }
}

/// Project one transcript row into the render snapshot. The byline pair is
/// resolved by the caller (`rebuild`), which has store access for the
/// model/backend display names.
fn post_data_from(
    m: &ChatMessageView,
    byline: SharedString,
    byline_backend: Option<SharedString>,
) -> PostData {
    PostData {
        action_id: m.action_id.clone().map(SharedString::from),
        item_id: m.item_id.clone().map(SharedString::from),
        parent_action_id: m.parent_action_id.clone().map(SharedString::from),
        role: m.message.role.clone().into(),
        byline,
        byline_backend,
        time: fmt_clock(m.created_at).into(),
        content: m.message.content.clone().into(),
        model: m.model.clone().map(SharedString::from),
        generation_count: m.generation_count,
        reasoning: m.reasoning.clone().map(SharedString::from),
        reasoning_expanded: m.reasoning_expanded,
        references: m.references.clone(),
        blocks: m.blocks.clone(),
    }
}

/// Collect the ids of every node that has children (a horizontal scroller),
/// recording each scroller's branch count alongside.
fn collect_scrollers(
    roots: &[TreeNode],
    out: &mut HashSet<SharedString>,
    counts: &mut HashMap<SharedString, usize>,
) {
    for node in roots {
        if !node.children.is_empty() {
            out.insert(node.id.clone());
            counts.insert(node.id.clone(), node.children.len());
            collect_scrollers(&node.children, out, counts);
        }
    }
}

/// Window-facing copy for a typed submit failure (Phase 1 minimal surface;
/// onboarding is a later, separate window).
fn error_copy(e: &AppError) -> String {
    match e {
        AppError::NoAccount => "No account yet — create one to start a conversation.".to_string(),
        AppError::InsufficientBalance { .. } => {
            "Not enough credits to send. Add credits to continue.".to_string()
        }
        other => other.to_string(),
    }
}

impl Focusable for SpaceView {
    fn focus_handle(&self, _cx: &gpui::App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for SpaceView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let bg = theme.background;
        let fg = theme.foreground;
        let font_family = theme.font_family.clone();
        // The frame's content box, not the raw viewport: bottom-anchored
        // overlays (floating composer, minimap) and the scroll range must not
        // reach into the CSD shadow padding. The **conversation's** page is
        // that box less the inspector's column when it splits the window
        // (`page_size` is the accessor every other page-geometry consumer
        // reads), so the reading measure, branch strides, composer dock and
        // minimap all narrow together with it.
        let window_box = crate::chrome::content_size(window);
        let inspector_layout = self.inspector_layout(window_box.width);
        let viewport = self.page_size(window);
        let page_width = viewport.width;
        let window_h = viewport.height;

        // An open disclosure whose participant left the roster paints nothing;
        // retire it (and its dropdown, and the keyboard its field was holding)
        // before anything reads what it claims — `sync_tree_focus` below asks
        // whether an overlay owns the keyboard.
        self.sync_inspector_participant_edit(window, cx);
        // …and an invite form over a space that turns out to be a notebook: the
        // grant door is withheld there, and a form is a door left standing.
        self.sync_inspector_invite(window, cx);
        // Tree focus is *observed*, not merely bookkept: see
        // `keyboard::sync_tree_focus`.
        self.sync_tree_focus(window, cx);
        self.ensure_affordance_slots(cx);

        // Window resize: branch offsets are absolute pixels but pages are a
        // window-width apart, so remap every offset by the stride ratio to keep
        // the selected branch (and exact position) invariant to width.
        if let Some(prev) = self.last_page_width
            && prev != page_width
            && prev > px(0.)
            && page_width > px(0.)
        {
            let ratio = (page_width + BAND_HEIGHT).as_f32() / (prev + BAND_HEIGHT).as_f32();
            self.remap_for_resize(ratio);
        }
        self.last_page_width = Some(page_width);

        // Point the height cache at the current **reading-column** width — the
        // only thing a post's measured height depends on — not the raw window
        // width. Above `full_measure_page_width()` the column is capped at
        // `BODY_MAX_WIDTH`, so a resize there leaves `body_width` unchanged
        // (where a fresh window opens — see `lib.rs`) and the cache is *not*
        // invalidated: the (partially measured) heights survive, the document
        // height stays put, and the scroll offset doesn't ratchet. Keying on the
        // raw width instead cleared the whole cache on every resize, dropping
        // every post back to a rough estimate and jittering the page (and the
        // minimap) as the near-viewport posts re-measured estimate→real — even
        // where no text reflows. See `layout::body_width`.
        let page_layout = layout::page_layout(page_width);
        self.layout.ensure_width(
            page_layout.body_width,
            theme::font_scale(cx),
            page_layout.gutters,
        );
        self.sync_bodies(window, cx);
        // Keep each post's embed map + highlight set current, and request the
        // incoming-reference index for the posts that rendered last frame.
        self.sync_references(cx);
        // Ask the space for its trace index (idempotent; the entity owns it).
        self.sync_traces(cx);
        // Keep a docked tail draft at the end of every branch (the always-present
        // composer that replaces the leaf "+").
        self.sync_tail_drafts(window, cx);
        // A quote another window sent this space (task 37): take it and attach
        // it to a draft. **After** `sync_tail_drafts`, so it lands in the
        // branch's real tail composer rather than minting one that the sync
        // would then prune.
        self.adopt_offered_quotes(window, cx);
        self.sync_composer_gutters(page_layout, window.rem_size(), cx);

        let turns = self.stream_overlays(cx);
        let streaming = !turns.is_empty();
        let tree = self.effective_tree(page_width, &turns);
        self.sync_scrolls(&tree);

        // Warm the on-path posts: if any post the selected path renders is still
        // unmeasured (a cold open, a width change that emptied the cache, or a
        // freshly added post), force every post real for the next couple of
        // frames so it measures into the cache up front. This holds the document
        // height stable, so an off-screen post measuring lazily mid-scroll can't
        // shift everything below it and jump the composer's dock shadow / resize
        // the minimap. Self-terminating: once the path is fully measured it stops.
        if self.warm_remaining.get() == 0 && self.path_has_unmeasured(&tree, page_width) {
            self.warm_remaining.set(2);
        }

        // A freshly-created draft / freshly-started turn selects its branch on
        // the first frame it exists in the tree (computed here against the real
        // effective tree), then settles the page where that node lives.
        //
        // **The request lives exactly one frame** — the `take` is unconditional,
        // so a node that never appeared (a turn that failed, a draft retired
        // before it rendered) can't leave a request dangling for some later
        // frame to act on. A turn that *ended* is followed onto its post
        // instead, before this frame runs (`retarget_pending_turn_select`).
        if let Some(pending) = self.pending_select.take()
            && model::node_ref(&tree, &pending.node).is_some()
        {
            self.select_path_to(&tree, &pending.node, page_width);
            match pending.settle {
                PendingSettle::DockDraft => self.dock_active_draft(&tree, page_width, window_h),
                PendingSettle::BranchEnd => self.scroll_to_branch_end(&tree, page_width, window_h),
                PendingSettle::Stay => {}
            }
        }

        // Sync one read-only editor per in-flight turn to its live partial
        // (skip unchanged buffers), pruning editors whose turn has ended.
        {
            let live: Vec<(u64, String)> = self
                .space
                .read(cx)
                .streams()
                .iter()
                .map(|t| (t.seq, t.response.content.clone()))
                .collect();
            for (seq, content) in &live {
                let editor = match self.streaming_bodies.get(seq) {
                    Some(e) => e.clone(),
                    None => {
                        let e = cx.new(|cx| MarkdownEditorState::new(window, cx));
                        self.streaming_bodies.insert(*seq, e.clone());
                        e
                    }
                };
                if self.streaming_synced.get(seq) != Some(content) {
                    self.streaming_synced.insert(*seq, content.clone());
                    editor.update(cx, |e, cx| e.set_value(content.clone(), cx));
                }
            }
            self.streaming_bodies
                .retain(|seq, _| live.iter().any(|(s, _)| s == seq));
            self.streaming_synced
                .retain(|seq, _| live.iter().any(|(s, _)| s == seq));
        }

        // Cap the scroll position the frame *positions content from* to the
        // content's real scrollable range. The page hard-stops at the ends, but
        // the scroll handle's raw offset transiently overshoots during momentum;
        // every consumer reads `clamped_scroll_y()` (which clamps to
        // `[scroll_min_y, 0]`) so the docked composer / posts / minimap don't
        // drift past the end and flicker. Set before any consumer below.
        let floating_pad = self.floating_pad(&tree, page_width, window_h);
        // Top headroom for the first post (zero for an empty notebook); see
        // `doc_reserve`. Used for the scroll range, the forest origin, and the
        // content's top padding so all three agree.
        let doc_reserve = self.doc_reserve();
        // Both ends of the branch, from the one definition every settle reads
        // (`layout::page_end_ys`): the document end bounds the scroll range,
        // and the *content* end — the document less a trailing draft's
        // speculative runway — is where a reader comes to rest (task 46,
        // bug 2). They coincide on every frame a turn is actually streaming.
        let ends = self.page_end_ys(&tree, page_width, window_h);
        self.scroll_min_y.set(ends.document);
        let prev_content_end = self.follow_anchor.replace(ends.content);
        let content_end = ends.content;

        // **Follow the producing tail.** While a turn streams *on the branch
        // the reader is on*, the document grows with every delta; a reader
        // parked at the end wants to stay there, and a reader who has scrolled
        // away must never be yanked back. A sibling branch's stream is not this
        // reader's tail (see `selected_turn_seq`).
        //
        // A just-posted exchange follows too, via `tail_pin`: between the save
        // landing and the response's first delta there is no stream to observe,
        // yet the document keeps growing (the persisted post replaces the
        // optimistic one under a new node id and re-measures from its estimate),
        // which would drift the reader off the end before following could ever
        // engage. The pin ends the moment the exchange settles.
        // The same observation answers two questions: whether this frame's
        // selected path is producing, and *which* turn it is producing from —
        // recorded because the second question outlives the leaf that answers
        // it (`follow_completed_turn`).
        let selected_turn = self.selected_turn_seq(&tree, page_width);
        // While a branch switch animates, every frame observes the *rounded*
        // strip offset — which is the child being left until the slide is more
        // than half over. The switch already recorded its destination, and that
        // is the honest answer to "where is the reader?" until it lands. What
        // is *painted* still follows the observation, so `producing` (and the
        // tail-following it gates) reads the fresh value either way.
        if self.snap.is_none() {
            self.selected_turn.set(selected_turn);
        }
        let producing = selected_turn.is_some();
        if self.tail_pin.active() && !producing && !self.space.read(cx).is_busy() {
            self.tail_pin = TailPin::Inactive;
        }
        let pin_forced = self.tail_pin.forced();
        if self.follow_streaming_tail(
            producing || self.tail_pin.active(),
            pin_forced,
            prev_content_end,
            content_end,
        ) {
            let entity = cx.entity();
            window.on_next_frame(move |_, cx| entity.update(cx, |_, cx| cx.notify()));
        } else if pin_forced
            && !self.path_has_unmeasured(&tree, page_width)
            && self.page_glide.get().is_none()
            && (self.page_scroll.offset().y.as_f32() - content_end).abs() <= 0.5
        {
            self.tail_pin = TailPin::Observing;
        }

        // While a readonly post is being drag-selected, autoscroll the page when
        // the pointer sits against a viewport edge — so a selection can pull
        // off-screen content into view (the post editors have no internal
        // scroll; this page scroll is theirs). Runs after `scroll_min_y` is set
        // so the scroll stays clamped to the real range.
        self.autoscroll_selection(window_h, window, cx);

        // Schedule a single catch-up frame when the minimap's layout inputs
        // change, so it converges once the layout settles.
        let sig = self.minimap_signature(page_width, window_h, &turns);
        if self.minimap_sig.to_bits() != sig.to_bits() {
            self.minimap_sig = sig;
            let entity = cx.entity();
            window.on_next_frame(move |_, cx| entity.update(cx, |_, cx| cx.notify()));
        }

        // Keep the composer's frozen-scroll baseline in step between gestures.
        // Tick down the warm window, scheduling the next frame so the forced
        // real renders (and their measuring canvases) actually run.
        if self.warm_remaining.get() > 0 {
            self.warm_remaining.set(self.warm_remaining.get() - 1);
            let entity = cx.entity();
            window.on_next_frame(move |_, cx| entity.update(cx, |_, cx| cx.notify()));
        }

        // The pre-dock glide: while the dock threshold sits under the floating
        // composer (the last `float_bar_h` of page travel before docking), each
        // increment of page scroll consumed toward the threshold unwinds a
        // proportional share of the composer's internal scroll, so a scrolled
        // floating composer's content reaches its own top exactly as it docks —
        // instead of docking mid-scroll. Render-time, after this frame's page
        // offset is final, so every scroll source participates (wheel, minimap
        // drag, tail-follow, programmatic docks); must run before the
        // frozen-offset baseline below so the baseline picks up the eased value.
        self.glide_composer_toward_dock(&tree, page_width, window_h);

        self.composer_prev_off_y = self.composer_scroll.offset().y.as_f32();

        let body = self.render_forest(
            &tree,
            doc_reserve,
            page_width,
            window_h,
            window.rem_size(),
            streaming,
            cx,
        );

        // The window is a row: the conversation pane, then (when open and wide
        // enough) the inspector's column. The pane is `relative` and clipped,
        // so every absolutely-positioned surface inside it — title band,
        // composer, notices, minimap, context menu — anchors to the pane rather
        // than the window and can never paint over the panel.
        let pane = div()
            .relative()
            .flex_1()
            .min_w_0()
            .h_full()
            .overflow_hidden()
            .child({
                let mut scroll = div()
                    .id("space-scroll")
                    .size_full()
                    .overflow_y_scroll()
                    .track_scroll(&self.page_scroll)
                    .on_scroll_wheel(cx.listener(
                        |this, ev: &gpui::ScrollWheelEvent, window, cx| {
                            let delta = ev.delta.pixel_delta(window.line_height());
                            let moved = !delta.x.is_zero() || !delta.y.is_zero();
                            this.note_scroll_activity(ev.touch_phase, moved, cx);
                            if matches!(ev.touch_phase, gpui::TouchPhase::Started) {
                                this.scroll_owner = None;
                            }
                            if !delta.y.is_zero() {
                                this.scroll_owner.get_or_insert(ScrollOwner::Body);
                            }
                        },
                    ))
                    .child(
                        v_flex()
                            .id("space-conversation")
                            // The window's main landmark. Named here rather
                            // than on the scroll container so the probe's
                            // absolute canvas can't reach into the scrollable
                            // extent. What it *contains* is Wave C's job —
                            // today it is the branch dots, bands and asks.
                            .probe("space/conversation", gpui::Role::Main, "Conversation")
                            .tab_region(crate::focus::region::MAIN)
                            .w_full()
                            .pt(px(doc_reserve))
                            .pb(px(floating_pad))
                            .child(body),
                    );
                scroll.style().restrict_scroll_to_axis = Some(true);
                scroll
            })
            // The title band paints **after** the page it covers: gpui's
            // BoundsTree draw order is paint order, and a hitbox only blocks
            // what was painted before it — so as a *first* child the band both
            // sat under the posts (no fade) and let a press in the header reach
            // the `MarkdownEditor` beneath it, which then dragged out a
            // selection while the window moved (task 32). It stays ahead of the
            // composer and minimap, which paint over it as before.
            .child(self.render_title_bar(window, cx))
            .child(self.render_active_draft(&tree, page_width, window_h, window, cx))
            // The compact bottom action bar for a docked, *inactive* tail
            // draft — the active composer renders its own; this one fades in
            // as the draft's slot scrolls into view. Same paint position as
            // the composer (after the page, before the pickers/notices).
            .child(self.render_inactive_tail_action_bar(
                &tree,
                page_width,
                window_h,
                window.rem_size(),
                window,
                cx,
            ))
            // The source-highlight picker: which of several posts that quoted
            // the clicked passage to visit. Above the composer, below the
            // notices — it's a choice, not a state.
            .children(self.render_highlight_picker(cx))
            // The quote destination picker + its visibility statement: the
            // same layer as the highlight picker — a choice, not a state.
            .children(self.render_quote_destination(cx))
            .child(self.render_reference_notice(cx))
            .child(self.render_cascade_band(cx))
            .child(self.render_error_band(cx))
            // The minimap is the last sibling, so it paints after the composer
            // (an earlier sibling) and — overlapping it on the right edge — its
            // BoundsTree order lands above the composer's layer, keeping the scroll
            // map on top of the floating bar. No `deferred` needed now that neither
            // defers; staying in the normal pass keeps both below late overlays
            // like the gpui dev inspector.
            .child(self.render_minimap(&tree, page_width, window_h, window, cx))
            // The context menu is the last child of all: a menu opened at the
            // pointer must sit above every surface it can be opened over,
            // the minimap and the floating composer included.
            .children(self.render_context_menu(window, cx));

        crate::chrome::round_client_corners(div(), window)
            .track_focus(&self.focus_handle)
            .key_context("SpaceView")
            .on_action(cx.listener(Self::submit))
            .on_action(cx.listener(Self::post_only))
            .on_action(cx.listener(|_, _: &CloseWindow, window, _| {
                window.remove_window();
            }))
            // Space → Show/Hide Inspector (⌥⌘I). Registered per-view like
            // `CloseWindow`, so the item targets the focused space and macOS
            // greys it when no space window is open.
            .on_action(cx.listener(Self::toggle_inspector))
            // Edit → Quote / Quote in Reply. Registered **only while a
            // quotable post selection exists**, so `is_action_available` is
            // false otherwise and macOS greys both items — the same
            // registration-is-enablement mechanism as CloseWindow, with the
            // extra selection condition. `note_body_selection` re-renders on
            // exactly the transitions that flip this.
            .when(self.post_selection.is_some(), |d| {
                d.on_action(cx.listener(Self::quote))
                    .on_action(cx.listener(Self::quote_in_reply))
                    .on_action(cx.listener(Self::quote_elsewhere))
            })
            // **The sole owner of "Escape closes the context menu."** Key
            // dispatch bubbles inner→outer, so the root runs *last* — after
            // every inner Escape handler (the composer's, an edit session's)
            // has yielded via [`Self::context_menu_absorbs_escape`]. One
            // owner, consulted by one predicate, rather than a copy of the
            // close per handler that each has to remember to make first. A
            // no-op when no menu is open.
            .on_key_down(cx.listener(|this, ev: &gpui::KeyDownEvent, window, cx| {
                if ev.keystroke.key == "escape" {
                    // Rung 1 of the Escape chain (see `keyboard`): the menu
                    // wins, and consumes the press.
                    if this.close_context_menu(cx) {
                        return;
                    }
                    // Then the inspector's own dropdown — the same
                    // one-owner-for-Escape shape, for the same reason: it is a
                    // transient overlay, so the conversation's handler yields
                    // to it and something has to close it.
                    if this.close_inspector_picker(cx) {
                        return;
                    }
                    // …and the quote-destination picker, an overlay of the
                    // same kind over the conversation itself.
                    if this.close_quote_destination(window, cx) {
                        return;
                    }
                    // …and its Participants section's model dropdown, which is
                    // the same kind of overlay over the same panel.
                    if this.close_inspector_participant_picker(cx) {
                        return;
                    }
                }
                // The conversation's own keyboard model — arrows, Enter,
                // Escape's last two rungs, and type-to-compose. It runs as a
                // listener, so gpui's *binding* pass has already offered the
                // press to every inner context (the composer's editor, an
                // edit session); nothing here can shadow them.
                //
                // **A handled press must be consumed.** `gpui_macos`'s
                // `handle_key_event` reports the key as handled to AppKit only
                // when the callback comes back with `propagate == false`;
                // otherwise it falls through to
                // `[[self inputContext] handleEvent:]`, which hands the *same*
                // native event to whatever input handler is installed. For
                // type-to-compose that is the editor this press just focused,
                // so the character it already applied would be typed a second
                // time. `Window::dispatch_keystroke` (the test path) has the
                // same shape — `if !result.propagate { return true }`, else
                // `input_handler.dispatch_input(...)` — which is what makes
                // "the press was consumed" assertable here at all.
                if this.handle_conversation_key(ev, window, cx) {
                    cx.stop_propagation();
                }
            }))
            // The window's single modifiers listener (see `WindowInput`): the
            // root is an ancestor of the focused composer, so it sees every
            // modifier transition and mirrors it into the shared entity for
            // the action gutter's ⌥ reveal.
            .on_modifiers_changed(cx.listener(|this, event, _, cx| {
                this.window_input
                    .update(cx, |wi, cx| wi.update_modifiers(event, cx));
            }))
            .relative()
            .flex()
            .flex_row()
            .items_stretch()
            .size_full()
            .bg(bg)
            .font_family(font_family)
            .text_color(fg)
            .child(pane)
            // The inspector renders last: as a column it simply follows the
            // pane, and as an overlay it must paint after everything it covers
            // (the containment rule — `src/overlay.rs`).
            .children(self.render_inspector(inspector_layout, window, cx))
    }
}

impl SpaceView {
    /// The readonly post editor (or the streaming reply) currently mid
    /// drag-selection, if any. Drafts (the editable composer) are excluded —
    /// they float in their own scroll and route caret-into-view separately, so
    /// page autoscroll is only for the in-page readonly posts.
    fn selecting_editor(&self, cx: &gpui::App) -> Option<Entity<MarkdownEditorState>> {
        if let Some(e) = self
            .streaming_bodies
            .values()
            .find(|e| e.read(cx).is_selecting())
        {
            return Some(e.clone());
        }
        self.bodies
            .values()
            .find(|e| e.read(cx).is_selecting())
            .cloned()
    }

    /// **Tail-following.** While a response streams, the document grows with
    /// every delta. If the reader is parked at the end of the branch, keep them
    /// there — the answer should write itself into view. If they have scrolled
    /// away to re-read something, leave them exactly where they are.
    ///
    /// The "am I at the end?" question is answered by *observation*, not by a
    /// flag: the page is following iff its current offset still sits at the
    /// **previous** frame's end (`prev_end`, the target this function computed
    /// for the frame before). That makes every scroll source — the wheel, a
    /// minimap drag, selection autoscroll, a programmatic dock — participate for
    /// free, with no state to keep in sync: scrolling up leaves the band and
    /// following stops; scrolling back down re-enters it and following resumes.
    /// There is no "sticky" mode to get wedged.
    ///
    /// **"The end" is the end of what was written, not the end of the
    /// document** (`end`, the caller's `content_end`): a trailing draft's slot
    /// is a window of speculative runway, and a reader following a reply must
    /// come to rest on the reply — not be carried past it the moment the stream
    /// closes and `sync_tail_drafts` docks a fresh composer under it (task 46,
    /// bug 2). The two coincide whenever no draft trails the selected path,
    /// which is every frame a turn is actually streaming. The *observation*
    /// reads the same value, so following survives the frame the draft appears
    /// on rather than disengaging on its own correction.
    ///
    /// Deliberately gated on `producing` — **the selected path carries a live
    /// stream** ([`Self::selected_turn_seq`]), not merely "some turn in
    /// this space is streaming". A growing document is otherwise the composer's
    /// runway or a post measuring for the first time, and neither should move
    /// the reader; composer growth keeps its own caret-into-view path
    /// (`composer::caret_into_view`), which this must not race. Scoping to the
    /// space would have re-opened exactly that race whenever a fan-out streamed
    /// on a *sibling* branch — a routine Participants-v1 state, and one in which
    /// the reader's own branch is by definition not producing.
    ///
    /// The one deliberate widening is the post-submit pin ([`Self::tail_pin`]):
    /// a submit *is* the reader's own
    /// branch producing, but the stream that proves it only starts once the post
    /// has persisted and the notification plan resolves. Across that gap the
    /// document still grows, and a reader parked at the end by
    /// [`Self::settle_on_new_post`] would be left above it — with `at_tail`
    /// false by the time the first delta arrives, so following never engaged and
    /// the answer wrote itself off the bottom of the window. The pin is bounded
    /// by the exchange (`Space::is_busy`) and by the reader starting a new
    /// draft, so it never reaches the composer-runway growth the gate excludes.
    ///
    /// **Following yields to a navigation glide**, the one other motion that
    /// spans frames: a reader who clicked "See in context" while parked at the
    /// tail is on their way somewhere, and until they land the offset is not
    /// theirs to hold. Without the guard the two would trade the page for the
    /// frame or two it takes the glide to clear `TAIL_FOLLOW_EPSILON` (and, on
    /// a document still growing under them, for longer). The reverse direction
    /// needs nothing: the glide moves the reader off the end, `at_tail` reads
    /// false, and following stays disengaged on its own.
    ///
    /// Runs in `render` immediately after `scroll_min_y` is set for the frame,
    /// so every consumer of `clamped_scroll_y` sees the followed position.
    fn follow_streaming_tail(
        &self,
        producing: bool,
        pinned: bool,
        prev_end: f32,
        end: f32,
    ) -> bool {
        if !producing || self.page_glide.get().is_some() {
            return false;
        }
        let off = self.page_scroll.offset();
        let at_tail = off.y.as_f32() <= prev_end + TAIL_FOLLOW_EPSILON;
        if !pinned && !at_tail {
            return false; // the reader scrolled away — never yank them back
        }
        if (end - off.y.as_f32()).abs() > 0.5 {
            self.set_page_scroll_y(end);
            return true;
        }
        false
    }

    /// While a readonly post is being drag-selected, scroll the page toward
    /// whichever viewport edge the pointer is pressed against, then re-extend
    /// the drag from the (stationary) pointer against the freshly-scrolled
    /// geometry — so a selection started on-screen can pull off-screen rows into
    /// itself the way a native scroll-view text selection does. A no-op unless a
    /// post drag is live and the pointer is within
    /// [`SELECTION_AUTOSCROLL_MARGIN`] of an edge; self-terminating when the drag
    /// ends (the editor clears `is_selecting` on mouse-up).
    fn autoscroll_selection(
        &mut self,
        window_h: Pixels,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(editor) = self.selecting_editor(cx) else {
            return;
        };
        let mouse = window.mouse_position();
        let dy = selection_autoscroll_delta(
            mouse.y.as_f32(),
            window_h.as_f32(),
            SELECTION_AUTOSCROLL_MARGIN,
            SELECTION_AUTOSCROLL_MAX_SPEED,
        );
        if dy == 0.0 {
            return;
        }

        let off = self.page_scroll.offset();
        let new_y = (off.y.as_f32() + dy).clamp(self.scroll_min_y.get(), 0.0);
        let scrolled = (new_y - off.y.as_f32()).abs() > 0.01;
        if scrolled {
            self.set_page_scroll_y(new_y);
        }
        // Re-extend the selection to the row now under the (unmoved) pointer.
        editor.update(cx, |e, cx| e.drag_extend_to(mouse, cx));
        // Keep the autoscroll loop alive as long as the drag holds at the edge
        // *and* there is still room to scroll (there are no pointer-move events
        // while it sits still). Stopping once the page is clamped at an end
        // bounds the loop — so a headless `run_until_parked` can't spin — and is
        // correct: at an end there is no more off-screen content to reveal.
        if scrolled {
            let entity = cx.entity();
            window.on_next_frame(move |_, cx| {
                entity.update(cx, |_, cx| cx.notify());
            });
        }
    }

    /// Render the top of the forest: a single root is rendered directly; a
    /// multi-root forest gets the implicit top-level branch scroller.
    #[allow(clippy::too_many_arguments)]
    fn render_forest(
        &self,
        roots: &[TreeNode],
        doc_y: f32,
        page_width: Pixels,
        window_h: Pixels,
        rem_size: Pixels,
        streaming: bool,
        cx: &Context<Self>,
    ) -> AnyElement {
        match roots.len() {
            0 => div().into_any_element(),
            1 => self
                .render_subtree(
                    &roots[0], doc_y, page_width, window_h, rem_size, streaming, cx,
                )
                .into_any_element(),
            _ => self
                .render_strip(
                    model::ROOT_SCROLLER_ID,
                    roots,
                    doc_y,
                    page_width,
                    window_h,
                    rem_size,
                    streaming,
                    cx,
                )
                .into_any_element(),
        }
    }

    /// Render a node's whole subtree: its post, then (if it has replies) a
    /// separator band and the horizontal branch scroller whose pages are each
    /// child's entire subtree. Off-screen posts render as sized placeholders.
    #[allow(clippy::too_many_arguments)]
    fn render_subtree(
        &self,
        node: &TreeNode,
        doc_y: f32,
        page_width: Pixels,
        window_h: Pixels,
        rem_size: Pixels,
        streaming: bool,
        cx: &Context<Self>,
    ) -> gpui::Div {
        let mut column = v_flex().w(page_width);

        // The post, the draft's in-flow placeholder slot (when it's the active
        // floating composer), or an inactive draft rendered inline.
        match node.src {
            NodeSrc::Draft if self.active_draft.as_deref() == Some(&node.id) => {
                // Active draft: its editor floats in the overlay; the in-flow
                // slot is an empty placeholder that reserves the runway and
                // records the dock line + minimap bounds.
                let slot_h = self.draft_slot_height(node, page_width, window_h);
                column = column.child(
                    div()
                        .w(page_width)
                        .h(px(slot_h))
                        .flex_none()
                        .child(record_bounds(self.slot_bounds.clone(), node.id.clone())),
                );
            }
            NodeSrc::Draft => {
                // Inactive draft: render inline (an editable "Draft" post that
                // takes real vertical space); clicking it re-activates it.
                column = column.child(
                    self.render_inactive_draft(node, doc_y, page_width, window_h, rem_size, cx),
                );
            }
            _ => {
                column = column
                    .child(self.render_post_or_placeholder(node, doc_y, page_width, window_h, cx));
            }
        }

        if node.children.is_empty() {
            // A draft/streaming leaf simply ends; a normal leaf ends with a
            // separator band whose "+" starts a reply here.
            if matches!(node.src, NodeSrc::Draft | NodeSrc::Streaming(_)) {
                return column;
            }
            return column.child(self.render_band(node, page_width, cx));
        }

        let child_doc_y =
            doc_y + self.node_height(node, page_width, window_h) + BAND_HEIGHT.as_f32();
        column
            .child(self.render_band(node, page_width, cx))
            .child(self.render_strip(
                &node.id,
                &node.children,
                child_doc_y,
                page_width,
                window_h,
                rem_size,
                streaming,
                cx,
            ))
    }

    /// The horizontal branch scroller for a level: each page is one child's
    /// whole subtree (so a branch slides sideways as a unit). The scroller is
    /// sized to the *selected* child's subtree (dynamic height). Only the active
    /// page and its immediate neighbours render real (so a slide shows real
    /// content); distant siblings render as sized placeholders.
    #[allow(clippy::too_many_arguments)]
    fn render_strip(
        &self,
        scroller_id: &str,
        children: &[TreeNode],
        doc_y: f32,
        page_width: Pixels,
        window_h: Pixels,
        rem_size: Pixels,
        streaming: bool,
        cx: &Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme();
        let count = children.len();
        let active = self.active_child_index(scroller_id, page_width, count);
        let strip_h = self.selected_subtree_height(&children[active], page_width, window_h);
        let node_id = SharedString::from(scroller_id.to_string());

        let mut strip = h_flex()
            .id(SharedString::from(format!("{scroller_id}-children")))
            .w(page_width)
            .h(px(strip_h))
            .items_start()
            .overflow_x_scroll()
            .on_scroll_wheel(cx.listener({
                let node_id = node_id.clone();
                move |this, ev: &ScrollWheelEvent, window, cx| {
                    let delta = ev.delta.pixel_delta(window.line_height());
                    let moved = !delta.x.is_zero() || !delta.y.is_zero();
                    this.note_scroll_activity(ev.touch_phase, moved, cx);
                    match ev.touch_phase {
                        TouchPhase::Started => {
                            this.cancel_snap();
                            this.last_h_delta = px(0.);
                            // New gesture — release the previous owner so the
                            // first horizontal step below re-elects one.
                            this.h_scroll_owner = None;
                        }
                        TouchPhase::Moved => {
                            if !delta.x.is_zero() {
                                this.last_h_delta = delta.x;
                            }
                        }
                        // Nothing to unwind here — the per-gesture state this
                        // arm would reset is re-elected on the next `Started`,
                        // and `resolve_scroll_axis` already clears the axis
                        // lock for any non-`Moved` phase.
                        TouchPhase::Ended | TouchPhase::Cancelled => {}
                    }
                    let locked = this.scroll_axis;
                    match this.resolve_scroll_axis(ev.touch_phase, delta) {
                        ScrollAxis::Horizontal => {
                            cx.stop_propagation();
                            // The built-in scroller moved before this listener;
                            // invalidate the last frame's selected leaf even
                            // before the gesture ends and snap settles.
                            this.selected_turn.set(None);
                            // Elect the owner on the first horizontal step of the
                            // gesture (the strip actually under the cursor then)
                            // and hold it for the whole gesture.
                            let owner = this
                                .h_scroll_owner
                                .get_or_insert_with(|| node_id.clone())
                                .clone();
                            if owner == node_id {
                                // The owning strip: the built-in scroller already
                                // applied this step; just reassert any glide/pin.
                                this.reassert_horizontal(&node_id);
                            } else {
                                // A different strip drifted under the cursor as the
                                // owner's branches slid (siblings differ in height).
                                // Undo the step the built-in scroller applied to it
                                // and forward the delta to the owner instead.
                                if !delta.x.is_zero() {
                                    if let Some(handle) = this.scrolls.get(&node_id) {
                                        Self::undo_horizontal_nudge(handle, delta.x);
                                    }
                                    if let Some(handle) = this.scrolls.get(&owner) {
                                        let off = handle.offset();
                                        handle.set_offset(gpui::point(off.x + delta.x, off.y));
                                    }
                                }
                                this.reassert_horizontal(&owner);
                            }
                        }
                        ScrollAxis::Vertical => {
                            if !delta.x.is_zero()
                                && let Some(handle) = this.scrolls.get(&node_id)
                            {
                                Self::undo_horizontal_nudge(handle, delta.x);
                            }
                        }
                    }
                    if matches!(ev.touch_phase, TouchPhase::Ended)
                        && locked == Some(ScrollAxis::Horizontal)
                    {
                        // Snap the scroller that owned the gesture (which may not
                        // be the strip the cursor rests over at lift), sized by its
                        // own branch count.
                        let owner = this
                            .h_scroll_owner
                            .clone()
                            .unwrap_or_else(|| node_id.clone());
                        let owner_count =
                            this.scroller_counts.get(&owner).copied().unwrap_or(count);
                        this.start_snap(owner, page_width, owner_count, window, cx);
                    }
                }
            }));
        strip.style().restrict_scroll_to_axis = Some(true);
        strip.style().overflow.y = Some(Overflow::Hidden);
        if let Some(handle) = self.scrolls.get(scroller_id) {
            strip = strip.track_scroll(handle);
        }

        for (i, child) in children.iter().enumerate() {
            if i > 0 {
                // A vertical separator between branches, same ground as the
                // horizontal band, so crossing the seam reads as a boundary.
                strip = strip.child(div().w(BAND_HEIGHT).flex_none().h_full().bg(theme.muted));
            }
            let near = (i as i64 - active as i64).abs() <= 1;
            let page = if near {
                self.render_subtree(child, doc_y, page_width, window_h, rem_size, streaming, cx)
                    .into_any_element()
            } else {
                let h = self.selected_subtree_height(child, page_width, window_h);
                div().w(page_width).h(px(h)).into_any_element()
            };
            strip = strip.child(
                div()
                    .id(child.id.clone())
                    .w(page_width)
                    .flex_none()
                    .child(page),
            );
        }
        strip
    }

    /// A draggable band across the top standing in for the (transparent)
    /// titlebar: the shared [`crate::titlebar`] gesture (drag-to-move,
    /// double-click zoom, and — on Linux CSD — the right-click compositor
    /// window menu) over a fade-out gradient so posts scrolling under the
    /// band blend into the chrome. The band's top corners round with the CSD
    /// frame so it doesn't poke past the Adwaita arcs.
    fn render_title_bar(&self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let bg = cx.theme().background;
        // Built before the `make_draggable` call so the shared `&Window`
        // borrow it takes doesn't overlap the `&mut Window` the gesture needs.
        let band = crate::chrome::round_top_client_corners(div(), window)
            .id("space-title-bar")
            .absolute()
            .top_0()
            .left_0()
            .right_0()
            .h(TITLE_BAR_RESERVE)
            .bg(gpui::linear_gradient(
                180.,
                gpui::linear_color_stop(bg, 0.0),
                gpui::linear_color_stop(bg.opacity(0.0), 1.0),
            ));
        crate::titlebar::make_draggable(band, "space-title-bar", window, cx)
    }

    /// A minimal honest error band pinned over the bottom (Phase 1 surface for a
    /// typed submit failure; onboarding is a later, separate window). Renders
    /// nothing when there's no error.
    /// The failed-attempt **recovery notice** — a dismissible, on-theme card
    /// attached to the bottom of the failed exchange (never a terminal wall).
    /// It carries the error text (with a Copy affordance, since gpui text isn't
    /// selectable here — the app's established "get this text out" idiom, cf.
    /// onboarding's credential rows) and, when the saved user post can be
    /// re-requested ([`Space::can_retry`]), a **Retry** action that re-runs the
    /// ask without re-posting. Dismissing clears only the notice; the space
    /// itself is already recovered (the composer and per-post Edit remain live),
    /// so the user can also just keep typing a follow-up or edit their message.
    fn render_error_band(&self, cx: &Context<Self>) -> AnyElement {
        let Some(msg) = self.error.clone() else {
            return div().into_any_element();
        };
        let theme = cx.theme();
        let can_retry = self.space.read(cx).can_retry();
        let to_copy = msg.clone();

        // A quiet text action chip (Retry / Copy), matching the app's calm
        // hover-reveal verbs (cf. post.rs's action gutter).
        let chip = |id: SharedString,
                    probe: SharedString,
                    label: &'static str,
                    aria: SharedString,
                    accent: bool| {
            let base = if accent {
                theme.danger
            } else {
                theme.muted_foreground
            };
            let hover_fg = theme.foreground;
            let hover_bg = theme.muted;
            h_flex()
                .id(id)
                .probe(probe, Role::Button, aria)
                .px_2()
                .py_0p5()
                .rounded_md()
                .cursor_pointer()
                .text_sm()
                .text_color(base)
                .hover(move |s| s.text_color(hover_fg).bg(hover_bg))
                .child(label)
        };

        let mut actions = h_flex().items_center().gap_1();
        if can_retry {
            actions = actions.child(
                chip(
                    "space-error-retry".into(),
                    "space/error/retry".into(),
                    "Retry",
                    "Re-request a response".into(),
                    true,
                )
                .on_click(cx.listener(|this, _, window, cx| this.retry_failed(window, cx))),
            );
        }
        actions = actions.child(
            chip(
                "space-error-copy".into(),
                "space/error/copy".into(),
                "Copy",
                "Copy the error message".into(),
                false,
            )
            .on_click(move |_, _, cx| {
                cx.write_to_clipboard(ClipboardItem::new_string(to_copy.clone()));
            }),
        );

        div()
            .absolute()
            .bottom_0()
            .left_0()
            .right_0()
            .flex()
            .justify_center()
            .p_3()
            .child(
                v_flex()
                    .id("space-error-band")
                    // The band's own handle: its verbs are tab stops, so a
                    // dismiss has to know whether it is holding the keyboard.
                    .track_focus(&self.band_focus[Self::BAND_ERROR])
                    // Contained on the *card*, never on the transparent
                    // full-width wrapper above (which spans the window and
                    // would swallow clicks nowhere near the notice) — see
                    // `crate::overlay`.
                    .contain_mouse(Overlay::Popover)
                    // The notice was three unexplained buttons to a screen
                    // reader — "Dismiss", "Copy", "Retry" — because the message
                    // itself was a node-less `div`. The message rides as the
                    // **value**, not the label: the macOS adapter announces a
                    // live region from a node's value and re-announces only when
                    // that value changes, so this is the shape that starts
                    // speaking the day `aria_live` exists upstream (audit §7,
                    // U1). Until then it is at least perceivable.
                    .probe_value("space/error", Role::Alert, "Request failed", msg.clone())
                    .max_w(rems(34.))
                    .gap_2()
                    .px_4()
                    .py_3()
                    .rounded_lg()
                    .border_1()
                    .border_color(theme.danger.opacity(0.35))
                    .bg(theme.danger.opacity(0.1))
                    .child(
                        h_flex()
                            .items_start()
                            .justify_between()
                            .gap_3()
                            .child(
                                div()
                                    .flex_1()
                                    .min_w_0()
                                    .text_sm()
                                    .text_color(theme.danger)
                                    .child(msg),
                            )
                            .child(
                                div()
                                    .id("space-error-dismiss")
                                    .probe("space/error/dismiss", Role::Button, "Dismiss")
                                    .flex_none()
                                    .size_5()
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .rounded_md()
                                    .cursor_pointer()
                                    .text_color(theme.muted_foreground)
                                    .hover(|s| {
                                        s.text_color(cx.theme().foreground).bg(cx.theme().muted)
                                    })
                                    .child("×")
                                    .on_click(cx.listener(|this, _, window, cx| {
                                        this.dismiss_error(window, cx)
                                    })),
                            ),
                    )
                    .child(actions),
            )
            .into_any_element()
    }

    /// Dismiss the recovery notice — the user's explicit "end this recovery".
    /// Clears the view's message **and** the Space's recorded `failed_turn`, so
    /// a later sibling turn finishing can't resurrect an orphaned notice and
    /// `can_retry` reads honestly (`false`) afterward. The saved user post and
    /// the composer are untouched — the space is already recovered.
    pub fn dismiss_error(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let held = self.band_holds_focus(Self::BAND_ERROR, window, cx);
        self.error = None;
        self.space.update(cx, |s, cx| s.clear_failed_turn(cx));
        self.hand_back_band_focus(held, window, cx);
        cx.notify();
    }

    /// Whether the band at `slot` is the one holding the keyboard — asked
    /// **before** it is cleared, because the answer is about the subtree that
    /// is about to stop being painted.
    fn band_holds_focus(&self, slot: usize, window: &Window, cx: &gpui::App) -> bool {
        self.band_focus[slot].contains_focused(window, cx)
    }

    /// Give the keyboard back where a closing surface's keyboard belongs — the
    /// reader's place in the conversation if they have one, else the view root
    /// ([`Self::keyboard_home`]) — and only from a band that was holding it, so
    /// a reader typing beside a notice keeps their caret.
    fn hand_back_band_focus(&self, held: bool, window: &mut Window, cx: &mut Context<Self>) {
        if held {
            window.focus(&self.keyboard_home(), cx);
        }
    }

    /// Re-ask the failed turn's participant (the notice's Retry action).
    /// Clears the notice and routes through [`Space::retry`] — the same
    /// participant, the same target post, no re-posting; the streaming state
    /// (and later `StreamEnded`/`Failed`) drives the rest. Sibling turns
    /// still streaming are untouched.
    pub fn retry_failed(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.error = None;
        let seq = self.space.update(cx, |s, cx| s.retry(cx));
        // Select the branch the retried turn's streaming node lands on — not
        // whatever branch the user navigated to after the failure (PR #218
        // review), and not merely the target's path (the retry is a new
        // sibling; see `ask_participant`).
        match seq {
            Some(seq) => self.select_turn_branch(seq),
            None => self.scroll_to_tail(window, cx),
        }
        cx.notify();
    }

    /// Dismiss the "that quote leads somewhere you're not" notice.
    pub fn dismiss_reference_notice(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let held = self.band_holds_focus(Self::BAND_REFERENCE, window, cx);
        self.reference_notice = None;
        self.hand_back_band_focus(held, window, cx);
        cx.notify();
    }

    /// Test seam: a bottom band's own focus handle, if it is painting — the
    /// question a dismiss asks containment of (0 failure, 1 denied-follow,
    /// 2 cascade).
    #[doc(hidden)]
    pub fn band_focus_for_test(&self, slot: usize) -> Option<FocusHandle> {
        self.band_focus.get(slot).cloned()
    }

    /// What the reference notice says right now (behavior tests read it here
    /// rather than through the painted band).
    #[doc(hidden)]
    pub fn reference_notice_for_test(&self) -> Option<SharedString> {
        self.reference_notice.clone()
    }

    /// Slots in [`SpaceView::band_focus`] — the failure notice, the denied-follow
    /// notice, the cascade notice, in the precedence order they render in.
    const BAND_ERROR: usize = 0;
    const BAND_REFERENCE: usize = 1;
    const BAND_CASCADE: usize = 2;

    /// The quiet, dismissible **denied-follow** notice (task 37): a quote whose
    /// source conversation this reader takes no part in. Muted, not danger —
    /// nothing failed, and there is nothing to retry; the way onward is a
    /// membership the reader would have to be given. It carries no Copy either:
    /// the sentence is ours, fixed, and says all there is to say.
    fn render_reference_notice(&self, cx: &Context<Self>) -> AnyElement {
        let Some(notice) = self.reference_notice.clone() else {
            return div().into_any_element();
        };
        // The failure band still wins: an error is the more urgent state, and
        // both bottom-anchor.
        if self.error.is_some() {
            return div().into_any_element();
        }
        let theme = cx.theme();
        div()
            .absolute()
            .bottom_0()
            .left_0()
            .right_0()
            .flex()
            .justify_center()
            .p_3()
            .child(
                v_flex()
                    .id("space-reference-notice")
                    .track_focus(&self.band_focus[Self::BAND_REFERENCE])
                    .contain_mouse(Overlay::Popover)
                    // The sentence rides as the **value** — the announcement
                    // channel, the shape the other two notices already use.
                    .probe_value(
                        "space/reference-notice",
                        Role::Alert,
                        "Quote leads elsewhere",
                        notice.clone(),
                    )
                    .max_w(rems(34.))
                    .gap_2()
                    .px_4()
                    .py_3()
                    .rounded_lg()
                    .border_1()
                    .border_color(theme.border)
                    .bg(theme.muted)
                    .child(
                        h_flex()
                            .items_start()
                            .justify_between()
                            .gap_3()
                            .child(
                                div()
                                    .flex_1()
                                    .min_w_0()
                                    .text_sm()
                                    .text_color(theme.muted_foreground)
                                    .child(notice),
                            )
                            .child(
                                div()
                                    .id("space-reference-notice-dismiss")
                                    .probe(
                                        "space/reference-notice/dismiss",
                                        Role::Button,
                                        "Dismiss",
                                    )
                                    .flex_none()
                                    .size_5()
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .rounded_md()
                                    .cursor_pointer()
                                    .text_color(theme.muted_foreground)
                                    .hover(|s| {
                                        s.text_color(cx.theme().foreground).bg(cx.theme().muted)
                                    })
                                    .child("×")
                                    .on_click(cx.listener(|this, _, window, cx| {
                                        this.dismiss_reference_notice(window, cx)
                                    })),
                            ),
                    ),
            )
            .into_any_element()
    }

    /// Dismiss the cascade-paused notice (window-local; the paused state is
    /// re-announced if a later plan pauses again).
    pub fn dismiss_cascade(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let held = self.band_holds_focus(Self::BAND_CASCADE, window, cx);
        self.cascade_notice = None;
        self.hand_back_band_focus(held, window, cx);
        cx.notify();
    }

    /// The quiet, dismissible cascade-paused notice: the conversation reached
    /// its cascade limit; the way onward is an explicit ask (which bypasses
    /// the guard). One "Ask <agent>" chip per agent participant. Muted, not
    /// danger — nothing failed; the guard did its job.
    fn render_cascade_band(&self, cx: &Context<Self>) -> AnyElement {
        let Some(notice) = self.cascade_notice.clone() else {
            return div().into_any_element();
        };
        // The failure notice takes precedence over the pause notice — both
        // bottom-anchor, and an error is the more urgent state. So does the
        // reference notice, which answers a click the reader **just made**.
        if self.error.is_some() || self.reference_notice.is_some() {
            return div().into_any_element();
        }
        let theme = cx.theme();
        let agents = self.space_agents(cx);
        let notice_text = SharedString::from(format!(
            "Replies paused — the conversation reached its cascade limit ({}). \
             Ask to continue.",
            notice.limit
        ));

        let mut actions = h_flex().items_center().gap_1().flex_wrap();
        for (i, (pid, label)) in agents.iter().enumerate() {
            let fg = theme.muted_foreground;
            let hover_fg = theme.foreground;
            let hover_bg = theme.muted;
            let pid = pid.clone();
            let target = notice.target_action_id.clone();
            actions = actions.child(
                h_flex()
                    .id(SharedString::from(format!("space-cascade-ask-{i}")))
                    .probe(
                        format!("space/cascade/ask/{i}"),
                        Role::Button,
                        SharedString::from(format!("Ask {label} to continue")),
                    )
                    .px_2()
                    .py_0p5()
                    .rounded_md()
                    .cursor_pointer()
                    .text_sm()
                    .text_color(fg)
                    .hover(move |s| s.text_color(hover_fg).bg(hover_bg))
                    .child(SharedString::from(format!("Ask {label}")))
                    .on_click(cx.listener(move |this, _, window, cx| {
                        this.ask_participant(pid.clone(), target.clone(), window, cx);
                    })),
            );
        }

        div()
            .absolute()
            .bottom_0()
            .left_0()
            .right_0()
            .flex()
            .justify_center()
            .p_3()
            .child(
                v_flex()
                    .id("space-cascade-band")
                    .track_focus(&self.band_focus[Self::BAND_CASCADE])
                    // Contained on the card, not the full-width wrapper (see
                    // the failure notice above and `crate::overlay`).
                    .contain_mouse(Overlay::Popover)
                    // Same shape as the failure notice: the sentence is the
                    // value (the announcement channel), the label names the
                    // state. Muted, not danger — nothing failed.
                    .probe_value(
                        "space/cascade",
                        Role::Alert,
                        "Replies paused",
                        notice_text.clone(),
                    )
                    .max_w(rems(34.))
                    .gap_2()
                    .px_4()
                    .py_3()
                    .rounded_lg()
                    .border_1()
                    .border_color(theme.border)
                    .bg(theme.muted)
                    .child(
                        h_flex()
                            .items_start()
                            .justify_between()
                            .gap_3()
                            .child(
                                div()
                                    .flex_1()
                                    .min_w_0()
                                    .text_sm()
                                    .text_color(theme.muted_foreground)
                                    .child(notice_text),
                            )
                            .child(
                                div()
                                    .id("space-cascade-dismiss")
                                    .probe("space/cascade/dismiss", Role::Button, "Dismiss")
                                    .flex_none()
                                    .size_5()
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .rounded_md()
                                    .cursor_pointer()
                                    .text_color(theme.muted_foreground)
                                    .hover(|s| {
                                        s.text_color(cx.theme().foreground).bg(cx.theme().muted)
                                    })
                                    .child("×")
                                    .on_click(cx.listener(|this, _, window, cx| {
                                        this.dismiss_cascade(window, cx)
                                    })),
                            ),
                    )
                    .child(actions),
            )
            .into_any_element()
    }

    /// The space's title as the Library index knows it — what the window is
    /// named, what the inspector's title field edits, and what a saved template
    /// is proposed as. `None` for a blank ⌘N space or one the index hasn't
    /// caught up with.
    fn space_title(&self, cx: &gpui::App) -> Option<SharedString> {
        let space_id = self.space.read(cx).id()?.to_string();
        self.stores
            .spaces
            .read(cx)
            .list()
            .iter()
            .find(|s| s.id == space_id)
            .and_then(|s| s.title.clone())
            .map(SharedString::from)
    }

    /// Name the window after the conversation it holds, so VoiceOver's window
    /// chooser and the macOS Window menu can tell two spaces apart. Guarded on
    /// change — the title only moves when the space is auto-titled or renamed.
    ///
    /// **Called from the `stores.spaces` observer, never from `render`.**
    /// `Window::draw_roots` builds the frame's AccessKit root node — label and
    /// all — in `a11y.begin_frame()` *before* prepainting the root element, so
    /// a title written during `render` lands after that frame's root was
    /// already built from the previous one; and `set_window_title` marks
    /// nothing dirty, so no follow-up frame is scheduled. The platform title
    /// (Window menu, switcher) would update at once while the *accessible*
    /// root kept the stale name until some unrelated redraw. Writing it in the
    /// observer — which runs before the frame its own `notify` schedules —
    /// gets both from one pass.
    fn sync_window_title(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let title = self
            .space_title(cx)
            .unwrap_or_else(|| SharedString::from("New Space"));
        if self.window_title.as_ref() == Some(&title) {
            return;
        }
        window.set_window_title(&title);
        self.window_title = Some(title);
    }

    /// Test-only: the title last pushed to the window. `gpui`'s `TestWindow`
    /// doesn't implement `get_title`, so `Window::window_title()` reads back
    /// empty — this field is the only observable end of the call.
    #[doc(hidden)]
    pub fn window_title_for_test(&self) -> Option<&str> {
        self.window_title.as_deref()
    }
}

/// An invisible, zero-layout-impact overlay that records its (absolute) painted
/// bounds into `map` under `id` each frame — placed over a post/slot so the
/// minimap and dock can read its position.
pub(crate) fn record_bounds(
    map: Rc<RefCell<HashMap<SharedString, Bounds<Pixels>>>>,
    id: SharedString,
) -> impl IntoElement {
    gpui::canvas(
        move |bounds, _, _| {
            map.borrow_mut().insert(id.clone(), bounds);
        },
        |_, _, _, _| {},
    )
    .absolute()
    .size_full()
}

#[cfg(test)]
mod tests {
    use super::selection_autoscroll_delta;

    #[test]
    fn autoscroll_is_neutral_in_the_middle() {
        // Pointer well inside the viewport — no autoscroll.
        assert_eq!(selection_autoscroll_delta(340.0, 680.0, 56.0, 32.0), 0.0);
        assert_eq!(selection_autoscroll_delta(56.0, 680.0, 56.0, 32.0), 0.0);
        assert_eq!(
            selection_autoscroll_delta(680.0 - 56.0, 680.0, 56.0, 32.0),
            0.0
        );
    }

    #[test]
    fn autoscroll_scrolls_toward_the_pressed_edge() {
        // Near the top → positive delta (scroll toward the document start).
        assert!(selection_autoscroll_delta(20.0, 680.0, 56.0, 32.0) > 0.0);
        // Near the bottom → negative delta (scroll toward the end).
        assert!(selection_autoscroll_delta(660.0, 680.0, 56.0, 32.0) < 0.0);
    }

    #[test]
    fn autoscroll_ramps_and_caps_at_the_edge() {
        let margin = 56.0;
        let h = 680.0;
        let max = 32.0;
        // At the very top edge the up-speed is the full max…
        assert!((selection_autoscroll_delta(0.0, h, margin, max) - max).abs() < 1e-4);
        // …and past the edge (pointer dragged above the window) it holds max.
        assert!((selection_autoscroll_delta(-500.0, h, margin, max) - max).abs() < 1e-4);
        // Bottom edge: full negative speed, holding past the edge.
        assert!((selection_autoscroll_delta(h, h, margin, max) + max).abs() < 1e-4);
        assert!((selection_autoscroll_delta(h + 500.0, h, margin, max) + max).abs() < 1e-4);
        // Halfway into the top band → about half speed.
        let half = selection_autoscroll_delta(margin / 2.0, h, margin, max);
        assert!((half - max / 2.0).abs() < 1e-3, "half-depth ramp: {half}");
    }

    #[test]
    fn autoscroll_degrades_safely() {
        // A zero/negative margin or viewport never scrolls (guards div-by-zero).
        assert_eq!(selection_autoscroll_delta(10.0, 680.0, 0.0, 32.0), 0.0);
        assert_eq!(selection_autoscroll_delta(10.0, 0.0, 56.0, 32.0), 0.0);
    }
}
