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
//! - [`post`] — render one post + its separator band.
//! - [`composer`] — the floating/docking draft composer + submit routing.
//! - [`request`] — the composer's action gutter (Ask / Post / model) + the
//!   request panel (model selection; the home of future per-request config).
//! - [`minimap`] — the topology minimap.
//!
//! Performance: only posts intersecting the viewport render the real
//! `MarkdownEditor`; off-screen posts render as sized placeholders sized from
//! the cached layout, so per-frame text shaping is bounded to visible posts.

pub mod composer;
pub mod layout;
pub mod minimap;
pub mod model;
pub mod nav;
pub mod post;
pub mod request;

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use eidola_app_core::error::AppError;
use gpui::{
    AnyElement, AppContext, Bounds, Context, Entity, FocusHandle, Focusable, InteractiveElement,
    IntoElement, IsZero, Overflow, ParentElement, Pixels, Render, ScrollHandle, ScrollWheelEvent,
    SharedString, StatefulInteractiveElement, Styled, Subscription, Task, TouchPhase, Window, div,
    px, rems,
};
use gpui_component::{ActiveTheme, h_flex, v_flex};
use gpui_markdown_editor::{MarkdownEditorState, MarkdownStyle};

use crate::actions::CloseWindow;
use crate::space::{ChatMessageView, Space, SpaceEvent};
use crate::stores::Stores;
use crate::theme;
use crate::window_input::WindowInput;

use layout::Layout;
use model::{NodeSrc, PostData, TreeNode};
use nav::{ScrollAxis, ScrollOwner, SnapAnim};

// Re-export the composer actions (⌘↩ post & ask / ⌘⇧↩ post only, routed via
// the editor's `PressEnter` event) and the ⌥⌘M request-panel toggle.
pub use crate::actions::{PostOnly, Send, ToggleModelPicker};

// ---------------------------------------------------------------------------
// Layout constants — the book typography + the tree-navigation geometry.
// ---------------------------------------------------------------------------

/// Height reserved at the top of the window for the (transparent) titlebar.
#[cfg(target_os = "macos")]
pub(crate) const TITLE_BAR_RESERVE: Pixels = px(36.);
#[cfg(not(target_os = "macos"))]
pub(crate) const TITLE_BAR_RESERVE: Pixels = px(0.);

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

/// The composer grows with content up to this fraction of the window, then
/// scrolls internally; near the bottom of a branch it *docks* and grows into
/// the page.
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

/// `MarkdownStyle` for prose bodies and the composer: Newsreader at a book
/// size/leading with a gentle heading ramp and Courier-New inline code (its
/// x-height matches Newsreader's, where Menlo reads too large). `from_theme`
/// seeds the system font + theme colors, so we override the family back to
/// Newsreader for narrative content.
pub(crate) fn prose_style(cx: &gpui::App) -> MarkdownStyle {
    let mut style = MarkdownStyle::from_theme(cx)
        .font_size(PROSE_FONT_SIZE)
        .line_height(rems(PROSE_LINE_HEIGHT))
        .paragraph_gap(rems(PROSE_PARAGRAPH_GAP))
        .heading_base_font_size(PROSE_FONT_SIZE)
        .heading_font_size(|level, base| match level {
            1 => base * 1.5,
            2 => base * 1.25,
            3 => base * 1.125,
            _ => base,
        })
        .inline_code_font_family("Courier New");
    style.font_family = theme::FONT_FAMILY.into();
    style
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
    /// Focus → activate; `PressEnter` → submit/post-only. Held so it outlives
    /// the closure and is dropped with the draft.
    pub(crate) _sub: Subscription,
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
    /// A read-only editor synced to the live streaming partial each frame.
    pub(crate) streaming_body: Entity<MarkdownEditorState>,
    /// Last value pushed into `streaming_body`, to skip redundant re-parses.
    pub(crate) streaming_synced: String,

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
    /// A freshly-created draft to bring onto the selected path on the next
    /// render (computed there against the real effective tree).
    pub(crate) pending_select: Option<SharedString>,
    /// Internal scroll of the floating composer overlay.
    pub(crate) composer_scroll: ScrollHandle,
    pub(crate) composer_prev_off_y: f32,
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
    /// The composer's natural (unclipped) content height, recorded each frame.
    pub(crate) composer_content_h: Rc<RefCell<Pixels>>,
    /// Painted bounds of the composer's in-flow placeholder slot, keyed by the
    /// draft sentinel id — positions the dock and feeds the minimap.
    pub(crate) slot_bounds: Rc<RefCell<HashMap<SharedString, Bounds<Pixels>>>>,
    /// Owner of the current vertical scroll session.
    pub(crate) scroll_owner: Option<ScrollOwner>,
    /// Whether the request panel (model selection; the future home of
    /// per-request config) is open, anchored to the composer's action gutter.
    pub(crate) request_panel_open: bool,
    /// The post whose action gutter currently reveals its hover affordances
    /// (Edit / Regenerate), by node id.
    pub(crate) hovered_post: Option<SharedString>,
    /// The post currently being edited in place, if any.
    pub(crate) editing: Option<EditingPost>,
    /// The composer bar's top `y` as of the last `render_active_draft`, so the
    /// request panel (a later sibling) can anchor against it.
    pub(crate) composer_anchor_top: Cell<f32>,

    /// Vertical scroll of the whole page.
    pub(crate) page_scroll: ScrollHandle,
    /// The most-negative valid page scroll `y` for the current frame (the
    /// content hard-stops here). Set once per `render` from the real document
    /// height; everything that *positions* content from the scroll offset reads
    /// it via `clamped_scroll_y`, so transient momentum overshoot past the ends
    /// never moves the docked composer / posts / minimap (the flicker fix,
    /// generalized).
    pub(crate) scroll_min_y: Cell<f32>,
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

    /// A minimal honest error band (e.g. a submit failing before onboarding
    /// exists). Full onboarding is a later, separate window.
    pub(crate) error: Option<String>,
}

impl SpaceView {
    pub fn new(
        stores: Stores,
        space_id: Option<String>,
        window_input: Entity<WindowInput>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let streaming_body = cx.new(|cx| MarkdownEditorState::new(window, cx));
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

        let _subs = vec![
            // Any space change re-derives the render snapshot and re-renders.
            cx.observe(&space, |this: &mut Self, _, cx| {
                this.rebuild(cx);
                cx.notify();
            }),
            cx.subscribe_in(&space, window, Self::on_space_event),
            cx.observe(&window_input, |_, _, cx| cx.notify()),
            // The request panel renders the model list + config default.
            cx.observe(&stores.models, |_, _, cx| cx.notify()),
            cx.observe(&stores.config, |_, _, cx| cx.notify()),
        ];

        let mut this = Self {
            stores,
            space,
            window_input,
            focus_handle,
            _subs,
            posts: Vec::new(),
            bodies: HashMap::new(),
            streaming_body,
            streaming_synced: String::new(),
            drafts: Vec::new(),
            active_draft: None,
            next_draft_seq: 0,
            pending_select: None,
            composer_scroll: ScrollHandle::new(),
            composer_prev_off_y: 0.0,
            composer_overlayed: Cell::new(false),
            composer_scrollable: Cell::new(false),
            composer_content_h: Rc::new(RefCell::new(px(0.))),
            slot_bounds: Rc::new(RefCell::new(HashMap::new())),
            scroll_owner: None,
            request_panel_open: false,
            hovered_post: None,
            editing: None,
            composer_anchor_top: Cell::new(0.0),
            page_scroll: ScrollHandle::new(),
            scroll_min_y: Cell::new(0.0),
            scrolls: HashMap::new(),
            scroller_counts: HashMap::new(),
            h_scroll_owner: None,
            scroll_axis: None,
            last_h_delta: px(0.),
            snap: None,
            snap_pin: None,
            last_page_width: None,
            layout: Layout::new(),
            warm_remaining: Cell::new(0),
            minimap_sig: f32::NAN,
            minimap_visible: false,
            minimap_gesturing: false,
            minimap_hovered: false,
            minimap_fade_gen: 0,
            minimap_hide_task: None,
            error: None,
        };
        this.rebuild(cx);
        if is_blank {
            // The blank notebook: a root draft, focused and ready.
            this.create_draft(None, window, cx);
        } else {
            // No composer yet — focus the root so action dispatch still works.
            window.focus(&this.focus_handle, cx);
        }
        this
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

    /// Whether the minimap is currently shown (tests assert scroll reveals it).
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

    // -- Snapshot & model resolution --------------------------------------

    /// Rebuild the per-row render snapshot from the shared `Space`'s transcript.
    /// Called on every space change (cheap `SharedString` projection) and at
    /// construction. Drafts are local UI state and are left untouched (an
    /// orphaned draft whose parent vanished re-attaches as a root in
    /// [`Self::effective_tree`]).
    pub(crate) fn rebuild(&mut self, cx: &mut Context<Self>) {
        let posts: Vec<PostData> = self
            .space
            .read(cx)
            .messages()
            .iter()
            .map(post_data_from)
            .collect();
        self.posts = posts;
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

    /// Resolve the model for a send: space selection → config default →
    /// embedded fallback. (The ⌥ picker UI is deferred, but resolution is
    /// needed now.)
    pub(crate) fn current_model(&self, cx: &gpui::App) -> String {
        if let Some(model) = self.space.read(cx).selected_model() {
            return model.to_string();
        }
        self.stores
            .config
            .read(cx)
            .state()
            .map(|s| s.default_model.clone())
            .unwrap_or_else(|| eidola_app_core::config::DEFAULT_MODEL.to_string())
    }

    /// React to a semantic `SpaceEvent`: re-snapshot + re-render, surface a
    /// typed failure as the minimal error band, and clear it on success.
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
                self.error = None;
                self.rebuild(cx);
            }
            SpaceEvent::Failed(e) => {
                self.error = Some(error_copy(e));
                self.rebuild(cx);
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
                    // Keep an existing editor in sync if its content changed
                    // (an edit/regenerate replaced the post in place) — but
                    // never clobber the editor holding an in-progress inline
                    // edit: its divergence from the persisted content *is* the
                    // edit.
                    let is_editing = self.editing.as_ref().map(|e| &e.node_id) == Some(&id);
                    if !is_editing && editor.read(cx).value() != content.as_ref() {
                        editor.update(cx, |e, cx| e.set_value(content.to_string(), cx));
                    }
                }
                None => {
                    let editor = cx.new(|cx| {
                        let mut s = MarkdownEditorState::new(window, cx);
                        s.set_value(content.to_string(), cx);
                        s
                    });
                    self.bodies.insert(id.clone(), editor);
                }
            }
        }
        self.bodies.retain(|id, _| live.contains(id));
        // An edit session whose post vanished (transcript reshaped under it)
        // has nothing to commit into — drop it with the editor.
        if let Some(ed) = &self.editing
            && !live.contains(&ed.node_id)
        {
            self.editing = None;
        }
        // Keep height-cache entries for live posts, live drafts, and streaming.
        let draft_ids: HashSet<SharedString> = self.drafts.iter().map(|d| d.id.clone()).collect();
        self.layout
            .retain(&|id| live.contains(id) || draft_ids.contains(id) || id == model::STREAMING_ID);
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

    /// The effective render forest: the persisted post tree, plus the streaming
    /// reply (while streaming) attached at the selected leaf, plus every draft
    /// attached as a leaf of its parent post (`None` parent → a root draft).
    /// Drafts attach after a node's persisted children, in `self.drafts` order,
    /// so a draft's branch index is deterministic.
    fn effective_tree(&self, page_width: Pixels, streaming: bool) -> Vec<TreeNode> {
        let mut roots = model::build_tree(&self.posts);
        if streaming {
            let overlay = TreeNode::leaf(NodeSrc::Streaming, model::STREAMING_ID);
            match self.selected_leaf_id(&roots, page_width) {
                Some(t) if model::node_ref(&roots, &t).is_some() => {
                    model::attach_overlay(&mut roots, &t, overlay);
                }
                _ => roots.push(overlay),
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

/// Project one transcript row into the render snapshot.
fn post_data_from(m: &ChatMessageView) -> PostData {
    PostData {
        action_id: m.action_id.clone().map(SharedString::from),
        parent_action_id: m.parent_action_id.clone().map(SharedString::from),
        role: m.message.role.clone().into(),
        byline: m.byline.clone().into(),
        time: fmt_clock(m.created_at).into(),
        content: m.message.content.clone().into(),
        generation_count: m.generation_count,
        reasoning: m.reasoning.clone().map(SharedString::from),
        reasoning_expanded: m.reasoning_expanded,
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
        let viewport = window.viewport_size();
        let page_width = viewport.width;
        let window_h = viewport.height;

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
        // width. Above ~836px the column is capped at `BODY_MAX_WIDTH`, so a
        // resize there leaves `body_width` unchanged and the cache is *not*
        // invalidated: the (partially measured) heights survive, the document
        // height stays put, and the scroll offset doesn't ratchet. Keying on the
        // raw width instead cleared the whole cache on every resize, dropping
        // every post back to a rough estimate and jittering the page (and the
        // minimap) as the near-viewport posts re-measured estimate→real — even
        // where no text reflows. See `layout::body_width`.
        self.layout.ensure_width(layout::body_width(page_width));
        self.sync_bodies(window, cx);
        // Keep a docked tail draft at the end of every branch (the always-present
        // composer that replaces the leaf "+").
        self.sync_tail_drafts(window, cx);

        let streaming = self.space.read(cx).is_streaming();
        let tree = self.effective_tree(page_width, streaming);
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

        // A freshly-created draft selects its branch on the first frame it
        // exists in the tree (computed here against the real effective tree),
        // then docks the page so the composer lands at its "home" position.
        if let Some(sel) = self.pending_select.take()
            && model::node_ref(&tree, &sel).is_some()
        {
            self.select_path_to(&tree, &sel, page_width);
            self.dock_active_draft(&tree, page_width, window_h);
        }

        // Sync the streaming editor to the live partial (skip if unchanged).
        if streaming {
            let content = self
                .space
                .read(cx)
                .streaming()
                .map(|s| s.content.clone())
                .unwrap_or_default();
            if content != self.streaming_synced {
                self.streaming_synced = content.clone();
                self.streaming_body
                    .update(cx, |e, cx| e.set_value(content, cx));
            }
        }

        // Cap the scroll position the frame *positions content from* to the
        // content's real scrollable range. The page hard-stops at the ends, but
        // the scroll handle's raw offset transiently overshoots during momentum;
        // every consumer reads `clamped_scroll_y()` (which clamps to
        // `[scroll_min_y, 0]`) so the docked composer / posts / minimap don't
        // drift past the end and flicker. Set before any consumer below.
        let floating_pad = self.floating_pad(&tree, page_width, window_h, streaming);
        // Top headroom for the first post (zero for an empty notebook); see
        // `doc_reserve`. Used for the scroll range, the forest origin, and the
        // content's top padding so all three agree.
        let doc_reserve = self.doc_reserve();
        let total_doc =
            doc_reserve + self.selected_total_height(&tree, page_width, window_h) + floating_pad;
        self.scroll_min_y
            .set((window_h.as_f32() - total_doc).min(0.0));

        // Schedule a single catch-up frame when the minimap's layout inputs
        // change, so it converges once the layout settles.
        let sig = self.minimap_signature(page_width, window_h);
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

        self.composer_prev_off_y = self.composer_scroll.offset().y.as_f32();

        let body = self.render_forest(&tree, doc_reserve, page_width, window_h, streaming, cx);

        div()
            .track_focus(&self.focus_handle)
            .key_context("SpaceView")
            .on_action(cx.listener(Self::submit))
            .on_action(cx.listener(Self::post_only))
            .on_action(cx.listener(Self::toggle_request_panel_action))
            .on_action(cx.listener(|_, _: &CloseWindow, window, _| {
                window.remove_window();
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
            .size_full()
            .bg(bg)
            .font_family(font_family)
            .text_color(fg)
            .child(self.render_title_bar(window, cx))
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
                            .w_full()
                            .pt(px(doc_reserve))
                            .pb(px(floating_pad))
                            .child(body),
                    );
                scroll.style().restrict_scroll_to_axis = Some(true);
                scroll
            })
            .child(self.render_active_draft(&tree, page_width, window_h, cx))
            .child(self.render_request_panel(page_width, window_h, cx))
            .child(self.render_error_band(cx))
            // The minimap is the last sibling, so it paints after the composer
            // (an earlier sibling) and — overlapping it on the right edge — its
            // BoundsTree order lands above the composer's layer, keeping the scroll
            // map on top of the floating bar. No `deferred` needed now that neither
            // defers; staying in the normal pass keeps both below late overlays
            // like the gpui dev inspector.
            .child(self.render_minimap(&tree, page_width, window_h, cx))
    }
}

impl SpaceView {
    /// Render the top of the forest: a single root is rendered directly; a
    /// multi-root forest gets the implicit top-level branch scroller.
    fn render_forest(
        &self,
        roots: &[TreeNode],
        doc_y: f32,
        page_width: Pixels,
        window_h: Pixels,
        streaming: bool,
        cx: &Context<Self>,
    ) -> AnyElement {
        match roots.len() {
            0 => div().into_any_element(),
            1 => self
                .render_subtree(&roots[0], doc_y, page_width, window_h, streaming, cx)
                .into_any_element(),
            _ => self
                .render_strip(
                    model::ROOT_SCROLLER_ID,
                    roots,
                    doc_y,
                    page_width,
                    window_h,
                    streaming,
                    cx,
                )
                .into_any_element(),
        }
    }

    /// Render a node's whole subtree: its post, then (if it has replies) a
    /// separator band and the horizontal branch scroller whose pages are each
    /// child's entire subtree. Off-screen posts render as sized placeholders.
    fn render_subtree(
        &self,
        node: &TreeNode,
        doc_y: f32,
        page_width: Pixels,
        window_h: Pixels,
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
                column =
                    column.child(self.render_inactive_draft(node, doc_y, page_width, window_h, cx));
            }
            _ => {
                column = column
                    .child(self.render_post_or_placeholder(node, doc_y, page_width, window_h, cx));
            }
        }

        if node.children.is_empty() {
            // A draft/streaming leaf simply ends; a normal leaf ends with a
            // separator band whose "+" starts a reply here.
            if matches!(node.src, NodeSrc::Draft | NodeSrc::Streaming) {
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
                        TouchPhase::Ended => {}
                    }
                    let locked = this.scroll_axis;
                    match this.resolve_scroll_axis(ev.touch_phase, delta) {
                        ScrollAxis::Horizontal => {
                            cx.stop_propagation();
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
                self.render_subtree(child, doc_y, page_width, window_h, streaming, cx)
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
    /// titlebar: the shared [`crate::titlebar`] gesture over a fade-out
    /// gradient so posts scrolling under the band blend into the chrome.
    fn render_title_bar(&self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let bg = cx.theme().background;
        crate::titlebar::make_draggable(
            div()
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
                )),
            "space-title-bar",
            window,
            cx,
        )
    }

    /// A minimal honest error band pinned over the bottom (Phase 1 surface for a
    /// typed submit failure; onboarding is a later, separate window). Renders
    /// nothing when there's no error.
    fn render_error_band(&self, cx: &Context<Self>) -> AnyElement {
        let Some(msg) = self.error.clone() else {
            return div().into_any_element();
        };
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
                div()
                    .max_w(rems(34.))
                    .px_4()
                    .py_2()
                    .rounded_lg()
                    .bg(theme.danger.opacity(0.12))
                    .text_color(theme.danger)
                    .text_sm()
                    .child(msg),
            )
            .into_any_element()
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
