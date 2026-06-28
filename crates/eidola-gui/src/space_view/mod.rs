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

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use eidola_app_core::error::AppError;
use gpui::{
    AnyElement, AppContext, Bounds, Context, Entity, FocusHandle, Focusable, InteractiveElement,
    IntoElement, IsZero, MouseButton, Overflow, ParentElement, Pixels, Render, ScrollHandle,
    ScrollWheelEvent, SharedString, StatefulInteractiveElement, Styled, Subscription, Task,
    TouchPhase, Window, div, px, rems,
};
use gpui_component::{ActiveTheme, InteractiveElementExt, h_flex, v_flex};
use gpui_markdown_editor::{MarkdownEditorState, MarkdownStyle};

use crate::space::{ChatMessageView, Space, SpaceEvent};
use crate::stores::Stores;
use crate::theme;
use crate::window_input::WindowInput;

use layout::Layout;
use model::{NodeSrc, PostData, TreeNode};
use nav::{ScrollAxis, ScrollOwner, SnapAnim};

// Re-export the chat actions so the composer routes the same semantic gestures
// (⌘↩ post & ask / ⌘⇧↩ post only) the editor's `PressEnter` event carries.
pub use crate::chat::{PostOnly, Send};

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
        .paragraph_gap(rems(1.5))
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

pub struct SpaceView {
    /// The store bundle — held whole for model resolution (config default) and
    /// to open spaces through the `SpacesStore` registry.
    pub(crate) stores: Stores,
    /// The shared per-conversation entity. The view is a window-local lens over
    /// it; two windows on one space share this entity.
    pub(crate) space: Entity<Space>,
    /// Per-window modifier state (held for the deferred ⌥ model picker).
    #[allow(dead_code)]
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
    /// The single live, editable composer.
    pub(crate) composer: Entity<MarkdownEditorState>,
    /// A read-only editor synced to the live streaming partial each frame.
    pub(crate) streaming_body: Entity<MarkdownEditorState>,
    /// Last value pushed into `streaming_body`, to skip redundant re-parses.
    pub(crate) streaming_synced: String,

    /// The post the composer currently replies to (`None` = the selected leaf,
    /// i.e. continue the thread tail). Set by a band's "+" / reply affordance to
    /// branch the thread there.
    pub(crate) reply_to: Option<SharedString>,
    /// Internal scroll of the floating composer overlay.
    pub(crate) composer_scroll: ScrollHandle,
    pub(crate) composer_prev_off_y: f32,
    /// Whether the composer overlay is floating (vs docked), cached from the
    /// last render so the scroll handler can decide session ownership.
    pub(crate) composer_overlayed: Cell<bool>,
    /// The composer's natural (unclipped) content height, recorded each frame.
    pub(crate) composer_content_h: Rc<RefCell<Pixels>>,
    /// Painted bounds of the composer's in-flow placeholder slot, keyed by the
    /// draft sentinel id — positions the dock and feeds the minimap.
    pub(crate) slot_bounds: Rc<RefCell<HashMap<SharedString, Bounds<Pixels>>>>,
    /// Owner of the current vertical scroll session.
    pub(crate) scroll_owner: Option<ScrollOwner>,

    /// Vertical scroll of the whole page.
    pub(crate) page_scroll: ScrollHandle,
    /// One horizontal scroller per node that has children, keyed by node id.
    pub(crate) scrolls: HashMap<SharedString, ScrollHandle>,
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
    /// Signature of the minimap's layout inputs, so a reflow/scroll schedules
    /// exactly one catch-up frame.
    pub(crate) minimap_sig: f32,
    pub(crate) minimap_visible: bool,
    pub(crate) minimap_gesturing: bool,
    pub(crate) minimap_hovered: bool,
    pub(crate) minimap_fade_gen: usize,
    pub(crate) minimap_hide_task: Option<Task<()>>,

    /// Armed on mouse-down in the title-bar band, consumed on the first move to
    /// begin a native window drag.
    pub(crate) should_move_window: bool,

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
        let composer = cx.new(|cx| MarkdownEditorState::new(window, cx));
        let streaming_body = cx.new(|cx| MarkdownEditorState::new(window, cx));
        let focus_handle = cx.focus_handle();

        // Focus the composer so the cursor lands in it on open (like a fresh
        // journal page). The view's own `focus_handle` is still tracked on the
        // root for action dispatch.
        let composer_focus = composer.read(cx).focus_handle(cx);
        window.focus(&composer_focus, cx);

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
            // The composer reports its submit chords as outward `PressEnter`
            // events; route them to the semantic submit/post-only commands.
            cx.subscribe_in(&composer, window, Self::on_editor_event),
            cx.observe(&window_input, |_, _, cx| cx.notify()),
        ];

        let mut this = Self {
            stores,
            space,
            window_input,
            focus_handle,
            _subs,
            posts: Vec::new(),
            bodies: HashMap::new(),
            composer,
            streaming_body,
            streaming_synced: String::new(),
            reply_to: None,
            composer_scroll: ScrollHandle::new(),
            composer_prev_off_y: 0.0,
            composer_overlayed: Cell::new(false),
            composer_content_h: Rc::new(RefCell::new(px(0.))),
            slot_bounds: Rc::new(RefCell::new(HashMap::new())),
            scroll_owner: None,
            page_scroll: ScrollHandle::new(),
            scrolls: HashMap::new(),
            scroll_axis: None,
            last_h_delta: px(0.),
            snap: None,
            snap_pin: None,
            last_page_width: None,
            layout: Layout::new(),
            minimap_sig: f32::NAN,
            minimap_visible: false,
            minimap_gesturing: false,
            minimap_hovered: false,
            minimap_fade_gen: 0,
            minimap_hide_task: None,
            should_move_window: false,
            error: None,
        };
        this.rebuild(cx);
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

    /// The composer editor state (tests set its value directly).
    #[doc(hidden)]
    pub fn composer_state_for_test(&self) -> Entity<MarkdownEditorState> {
        self.composer.clone()
    }

    /// The current per-row render snapshot length (tests assert tree shape).
    #[doc(hidden)]
    pub fn post_count_for_test(&self) -> usize {
        self.posts.len()
    }

    /// The current reply target, if the composer is branching (tests).
    #[doc(hidden)]
    pub fn reply_to_for_test(&self) -> Option<String> {
        self.reply_to.as_ref().map(|s| s.to_string())
    }

    /// Whether the minimap is currently shown (tests assert scroll reveals it).
    #[doc(hidden)]
    pub fn minimap_visible_for_test(&self) -> bool {
        self.minimap_visible
    }

    /// Drive the band "+" reply affordance (tests can't synthesize the click).
    #[doc(hidden)]
    pub fn start_reply_for_test(
        &mut self,
        action_id: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.start_reply(action_id.into(), window, cx);
    }

    // -- Snapshot & model resolution --------------------------------------

    /// Rebuild the per-row render snapshot from the shared `Space`'s transcript.
    /// Called on every space change (cheap `SharedString` projection) and at
    /// construction. Also drops a stale reply target whose post no longer
    /// exists.
    pub(crate) fn rebuild(&mut self, cx: &mut Context<Self>) {
        let posts: Vec<PostData> = self
            .space
            .read(cx)
            .messages()
            .iter()
            .map(post_data_from)
            .collect();
        self.posts = posts;
        // Drop a reply target that's no longer present (e.g. transcript reload
        // re-keyed ids); the composer falls back to the tail.
        if let Some(target) = self.reply_to.clone()
            && !self
                .posts
                .iter()
                .any(|p| p.action_id.as_deref() == Some(target.as_str()))
        {
            self.reply_to = None;
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
                    // (an edit/regenerate replaced the post in place).
                    if editor.read(cx).value() != content.as_ref() {
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
        self.layout
            .retain(&|id| live.contains(id) || id == model::DRAFT_ID || id == model::STREAMING_ID);
    }

    /// Ensure a horizontal `ScrollHandle` exists for every node that has
    /// children (a scroller), plus the implicit top-level scroller when there's
    /// more than one root. Prune handles for scrollers that no longer exist.
    fn sync_scrolls(&mut self, roots: &[TreeNode]) {
        let mut live: HashSet<SharedString> = HashSet::new();
        if roots.len() > 1 {
            live.insert(model::ROOT_SCROLLER_ID.into());
        }
        collect_scrollers(roots, &mut live);
        for id in &live {
            self.scrolls.entry(id.clone()).or_default();
        }
        self.scrolls.retain(|id, _| live.contains(id));
    }

    /// The effective render forest: the persisted post tree plus the streaming
    /// reply (while streaming) or the composer draft (otherwise) attached as a
    /// leaf of the reply target. Returns the forest and the overlay's parent id
    /// (so the caller can ensure that scroller's handle).
    fn effective_tree(
        &self,
        page_width: Pixels,
        streaming: bool,
    ) -> (Vec<TreeNode>, Option<SharedString>) {
        let mut roots = model::build_tree(&self.posts);
        let (overlay, target) = if streaming {
            (
                TreeNode::leaf(NodeSrc::Streaming, model::STREAMING_ID),
                self.selected_leaf_id(&roots, page_width),
            )
        } else {
            (
                TreeNode::leaf(NodeSrc::Draft, model::DRAFT_ID),
                self.reply_target_id(&roots, page_width),
            )
        };
        match &target {
            Some(t) if model::node_ref(&roots, t).is_some() => {
                model::attach_overlay(&mut roots, t, overlay);
            }
            _ => roots.push(overlay),
        }
        (roots, target)
    }

    /// The reply target for the composer: an explicit `reply_to` (a branch) when
    /// its post still exists, else the selected leaf (continue the tail).
    fn reply_target_id(&self, roots: &[TreeNode], page_width: Pixels) -> Option<SharedString> {
        if let Some(t) = &self.reply_to
            && model::node_ref(roots, t).is_some()
        {
            return Some(t.clone());
        }
        self.selected_leaf_id(roots, page_width)
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

/// Collect the ids of every node that has children (a horizontal scroller).
fn collect_scrollers(roots: &[TreeNode], out: &mut HashSet<SharedString>) {
    for node in roots {
        if !node.children.is_empty() {
            out.insert(node.id.clone());
            collect_scrollers(&node.children, out);
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

        // Point the height cache at the current width (clears on change) and
        // make sure every post/scroller has its editor + scroll handle.
        self.layout.ensure_width(page_width.as_f32());
        self.sync_bodies(window, cx);

        let streaming = self.space.read(cx).is_streaming();
        let (tree, overlay_parent) = self.effective_tree(page_width, streaming);
        self.sync_scrolls(&tree);
        if let Some(parent) = overlay_parent {
            self.scrolls.entry(parent).or_default();
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

        // Schedule a single catch-up frame when the minimap's layout inputs
        // change, so it converges once the layout settles.
        let sig = self.minimap_signature(page_width, window_h);
        if self.minimap_sig.to_bits() != sig.to_bits() {
            self.minimap_sig = sig;
            let entity = cx.entity();
            window.on_next_frame(move |_, cx| entity.update(cx, |_, cx| cx.notify()));
        }

        // Keep the composer's frozen-scroll baseline in step between gestures.
        self.composer_prev_off_y = self.composer_scroll.offset().y.as_f32();

        let floating_pad = self.floating_pad(&tree, page_width, window_h, streaming);
        let body = self.render_forest(
            &tree,
            TITLE_BAR_RESERVE.as_f32(),
            page_width,
            window_h,
            streaming,
            cx,
        );

        div()
            .track_focus(&self.focus_handle)
            .key_context("SpaceView")
            .on_action(cx.listener(Self::submit))
            .on_action(cx.listener(Self::post_only))
            .relative()
            .size_full()
            .bg(bg)
            .font_family(font_family)
            .text_color(fg)
            .child(self.render_title_bar(cx))
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
                            .pt(TITLE_BAR_RESERVE)
                            .pb(px(floating_pad))
                            .child(body),
                    );
                scroll.style().restrict_scroll_to_axis = Some(true);
                scroll
            })
            .child(self.render_active_draft(&tree, page_width, window_h, streaming, cx))
            .child(self.render_error_band(cx))
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

        // The post (or the draft's in-flow placeholder slot).
        match node.src {
            NodeSrc::Draft => {
                let slot_h = self.draft_slot_height(node, page_width, window_h);
                column = column.child(
                    div()
                        .w(page_width)
                        .h(px(slot_h))
                        .flex_none()
                        .child(record_bounds(self.slot_bounds.clone(), node.id.clone())),
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
                            this.reassert_horizontal(&node_id);
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
                        this.start_snap(node_id.clone(), page_width, count, window, cx);
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
    /// titlebar. On macOS `WindowControlArea::Drag` is a no-op, so dragging is
    /// wired explicitly: arm on mouse-down, then `start_window_move` on the
    /// first move while armed.
    fn render_title_bar(&self, cx: &Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let bg = theme.background;
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
            ))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _, _, _| this.should_move_window = true),
            )
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, _, _, _| this.should_move_window = false),
            )
            .on_mouse_move(cx.listener(|this, _, window, _| {
                if this.should_move_window {
                    this.should_move_window = false;
                    window.start_window_move();
                }
            }))
            .on_double_click(|_, window, _| window.titlebar_double_click())
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
