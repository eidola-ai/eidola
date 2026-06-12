//! Library window — the book's table of contents. Lists the user's spaces
//! (most recently active first), reopens one on click, and quietly reveals two
//! ghost buttons on hover: a pencil that starts an inline rename and an × that
//! archives.
//!
//! Design notes: this is deliberately *not* a chat-app sidebar. One prose
//! column, hairline `theme.border` rules between entries, no cards or
//! avatars. Each row is a title (or, for untitled spaces, a muted snippet
//! of the first message) with a right-aligned relative date in `text_sm`
//! muted. An empty library is a single quiet line.

use std::ops::Range;

use eidola_app_core::SpaceInfo;
use gpui::{
    App, AppContext, Context, Entity, FocusHandle, InteractiveElement, IntoElement, ParentElement,
    Render, SharedString, StatefulInteractiveElement, Styled, Subscription,
    UniformListScrollHandle, Window, actions, div, px, rems, uniform_list,
};
use gpui_component::{
    ActiveTheme, IconName, Sizable,
    button::{Button, ButtonVariants},
    h_flex,
    input::{Input, InputEvent, InputState},
    v_flex,
};

use crate::actions::CloseWindow;
use crate::stores::{SpacesStore, Stores};

actions!(library, [CancelRename]);

/// Vertical reserve under the macOS traffic lights — same pattern as
/// `chat::TITLE_BAR_RESERVE` (the window uses the shared transparent
/// titlebar from `lib.rs::transparent_titlebar`).
#[cfg(target_os = "macos")]
const TITLE_BAR_RESERVE: gpui::Pixels = gpui::px(36.);
#[cfg(not(target_os = "macos"))]
const TITLE_BAR_RESERVE: gpui::Pixels = gpui::px(0.);

/// Fixed row height for the virtualized listing. Rows are single-line by
/// design (title + relative date), so `uniform_list`'s single-measure layout
/// holds. Matches the former `py_3` single-line row rhythm.
const ROW_H: gpui::Pixels = gpui::px(46.);

pub struct LibraryView {
    stores: Stores,
    spaces: Entity<SpacesStore>,
    /// Index of the row currently under the pointer, for the hover-revealed
    /// archive affordance.
    hovered: Option<usize>,
    /// When `Some`, a rename is in progress for the given space id.  The
    /// `Entity<InputState>` holds the current draft text; the `Subscription`
    /// listens for `InputEvent`s (Enter → commit, Blur → cancel).
    renaming: Option<(String, Entity<InputState>, Vec<Subscription>)>,
    /// Focus handle the root v_flex tracks, so the `CloseWindow` listener is
    /// in the dispatch path (same pattern as `SettingsView`).
    focus_handle: FocusHandle,
    /// Scroll handle for the virtualized listing.
    scroll: UniformListScrollHandle,
    /// Test-only: how many times `open_space` has been invoked. Lets the
    /// pencil-propagation regression test prove that clicking the rename pencil
    /// does NOT also trigger the row's open (`open_space` itself defers a real
    /// window open that a behavior test can't easily count).
    open_space_requests: usize,
    _subscriptions: Vec<Subscription>,
}

impl LibraryView {
    pub fn new(stores: Stores, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let spaces = stores.spaces.clone();
        let _subscriptions = vec![cx.observe(&spaces, |_, _, cx| cx.notify())];
        spaces.update(cx, |s, cx| s.refresh(cx));

        let focus_handle = cx.focus_handle();
        focus_handle.focus(window, cx);

        Self {
            stores,
            spaces,
            hovered: None,
            renaming: None,
            focus_handle,
            scroll: UniformListScrollHandle::new(),
            open_space_requests: 0,
            _subscriptions,
        }
    }

    /// The focus handle the view tracks. Exposed so behavior tests can
    /// dispatch actions through it the same way real keystrokes would.
    pub fn focus_handle(&self) -> FocusHandle {
        self.focus_handle.clone()
    }

    /// Test-only: force the hover state so snapshots can render the archive
    /// affordance.
    #[doc(hidden)]
    pub fn set_hovered_for_test(&mut self, hovered: Option<usize>) {
        self.hovered = hovered;
    }

    /// Apply a hover transition for row `idx`. On hover-true the row becomes the
    /// hovered one; on hover-false we clear **only if `idx` is still the hovered
    /// row**. gpui doesn't order `on_hover` events across rows: moving the cursor
    /// up the list, the row being *left* can fire `on_hover(false)` *after* the
    /// row being *entered* fired `on_hover(true)`, so an unconditional clear
    /// would wipe the new row's hover (the × flickering off when moving up the
    /// list). Driven by the row's `on_hover` listener; exposed for behavior
    /// tests so they can replay that out-of-order sequence directly.
    pub fn set_row_hover(&mut self, idx: usize, hovering: bool, cx: &mut Context<Self>) {
        if hovering {
            self.hovered = Some(idx);
        } else if self.hovered == Some(idx) {
            self.hovered = None;
        }
        cx.notify();
    }

    /// The row index currently hovered, if any. Exposed for behavior tests.
    pub fn hovered_row(&self) -> Option<usize> {
        self.hovered
    }

    /// Archive a space. Called by the hover-revealed × button; public so
    /// behavior tests can exercise the same path without synthesizing mouse
    /// events.
    pub fn archive(&mut self, space_id: String, cx: &mut Context<Self>) {
        self.spaces.update(cx, |s, cx| s.archive(space_id, cx));
    }

    /// Open the given space in a new chat window. Deferred so the window
    /// opens after the current update cycle completes.
    pub fn open_space(&mut self, space_id: String, cx: &mut Context<Self>) {
        self.open_space_requests += 1;
        let stores = self.stores.clone();
        cx.defer(move |cx: &mut App| {
            crate::open_space_window(cx, stores, space_id);
        });
    }

    /// Test-only: how many times `open_space` has fired. The pencil-rename
    /// propagation regression test asserts this stays `0` when only the rename
    /// pencil was clicked.
    #[doc(hidden)]
    pub fn open_space_requests_for_test(&self) -> usize {
        self.open_space_requests
    }

    /// Begin inline rename for the given space.  Creates an `InputState` seeded
    /// with the current title (or empty for untitled spaces), subscribes to its
    /// events, and triggers a re-render so the row shows the input field.
    pub fn begin_rename(
        &mut self,
        space_id: String,
        current_title: Option<String>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // If already renaming the same space, do nothing.
        if self
            .renaming
            .as_ref()
            .map(|(id, _, _)| id == &space_id)
            .unwrap_or(false)
        {
            return;
        }
        let initial = current_title.unwrap_or_default();
        let input_state = cx.new(|cx| InputState::new(window, cx).default_value(&initial));
        // Focus the input so the user can type immediately.
        input_state.update(cx, |s, cx| s.focus(window, cx));

        let subs = vec![cx.subscribe_in(
            &input_state,
            window,
            |this, _, ev: &InputEvent, window, cx| match ev {
                InputEvent::PressEnter { .. } => this.commit_rename(window, cx),
                InputEvent::Blur => this.cancel_rename(cx),
                _ => {}
            },
        )];
        self.renaming = Some((space_id, input_state, subs));
        cx.notify();
    }

    /// Commit the in-progress rename — write the new title to the store and
    /// close the input.
    pub fn commit_rename(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        if let Some((space_id, input_state, _)) = self.renaming.take() {
            let title = input_state.read(cx).value().to_string();
            let title = title.trim().to_string();
            if !title.is_empty() {
                self.spaces
                    .update(cx, |s, cx| s.rename(space_id, title, cx));
            }
        }
        cx.notify();
    }

    /// Cancel an in-progress rename without persisting anything.
    pub fn cancel_rename(&mut self, cx: &mut Context<Self>) {
        self.renaming = None;
        cx.notify();
    }

    /// The id of the space currently being renamed, if any. Used by
    /// `render_row` to decide whether to show the input or the static title;
    /// also exposed for behavior tests.
    pub fn renaming_space_id(&self) -> Option<&str> {
        self.renaming.as_ref().map(|(id, _, _)| id.as_str())
    }

    /// Render the visible window of listing rows. Indexer for the virtualized
    /// `uniform_list` — clones only the visible slice from the store, so the
    /// per-frame cost is O(visible), not O(loaded).
    fn render_rows(&self, range: Range<usize>, cx: &mut Context<Self>) -> Vec<gpui::AnyElement> {
        let visible: Vec<(usize, SpaceInfo)> = self
            .spaces
            .read(cx)
            .list()
            .get(range.clone())
            .map(|slice| range.clone().zip(slice.iter().cloned()).collect())
            .unwrap_or_default();
        visible
            .into_iter()
            .map(|(idx, space)| self.render_row(idx, &space, cx).into_any_element())
            .collect()
    }

    fn render_row(
        &self,
        idx: usize,
        space: &SpaceInfo,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme();
        let hovered = self.hovered == Some(idx);
        let space_id = space.id.clone();
        let archive_id = space.id.clone();
        let rename_id = space.id.clone();
        let rename_title = space.title.clone();
        let is_renaming = self.renaming_space_id() == Some(space.id.as_str());

        // Title content: when this row is being renamed, show the inline
        // input; otherwise show the static title or snippet.
        let title_content: gpui::AnyElement = if is_renaming {
            if let Some((_, input_state, _)) = &self.renaming {
                // Ghost-styled inline input: no border/background chrome,
                // flex_1 so it fills the title column, same font as the row.
                Input::new(input_state).flex_1().into_any_element()
            } else {
                div().flex_1().into_any_element()
            }
        } else {
            let (line, is_fallback) = match (&space.title, &space.snippet) {
                (Some(t), _) => (t.clone(), false),
                (None, Some(s)) => (s.clone(), true),
                (None, None) => ("Untitled space".to_string(), true),
            };
            let mut title_el = div().flex_1().truncate().child(SharedString::from(line));
            if is_fallback {
                title_el = title_el.text_color(theme.muted_foreground);
            }
            title_el.into_any_element()
        };

        // Fixed-width reveal slot for the row affordances (pencil then ×), so
        // their hover appearance doesn't shift the date column. Two quiet ghost
        // buttons: the pencil starts the inline rename, the × archives. Both are
        // revealed on hover and hidden while this row is itself being renamed.
        let mut reveal_slot = h_flex().w_12().gap_1().justify_end();
        if hovered && !is_renaming {
            // **Both-phase propagation block.** Each affordance is wrapped in a
            // slot div that stops propagation on *both* mouse-down and mouse-up
            // (not just the click). The button's own `on_click` already calls
            // `cx.stop_propagation()`, but that only covers the click's mouse-up
            // *bubble* phase — the row records its own `pending_mouse_down` on
            // mouse-DOWN and captures it on the mouse-up *capture* phase, both
            // before the button's bubble click runs (gpui dispatches capture
            // outer→inner, then bubble inner→outer; see gpui `div.rs` paint).
            // Blocking the down stops the row from ever arming its pending
            // click; blocking the up's capture is belt-and-suspenders. This is
            // the structural half of the "pencil both renames and opens the row"
            // race. The other half — `begin_rename` reshaping the row mid-event
            // (title → input, reveal slot hidden) so hitboxes move between down
            // and up — is closed by deferring `begin_rename` (below), so the
            // whole click sequence resolves against the pre-rename layout.
            reveal_slot = reveal_slot
                .child(
                    div()
                        .id(("rename-slot", idx))
                        .debug_selector(move || format!("rename-pencil-{idx}"))
                        .on_mouse_down(
                            gpui::MouseButton::Left,
                            cx.listener(|_, _, _, cx| cx.stop_propagation()),
                        )
                        .on_mouse_up(
                            gpui::MouseButton::Left,
                            cx.listener(|_, _, _, cx| cx.stop_propagation()),
                        )
                        .child(
                            Button::new(("rename-space", idx))
                                .ghost()
                                .xsmall()
                                // The bundled Lucide icon set has no
                                // pencil/`square-pen` glyph; `case-sensitive`
                                // ("Aa") is the quiet text-edit affordance that
                                // reads as "rename this title".
                                .icon(IconName::CaseSensitive)
                                .on_click(cx.listener(move |_, _, window, cx| {
                                    // Don't let the click also open the row.
                                    cx.stop_propagation();
                                    // Defer the reshape so the in-flight click
                                    // sequence finishes against the old layout.
                                    let id = rename_id.clone();
                                    let title = rename_title.clone();
                                    cx.defer_in(window, move |this, window, cx| {
                                        this.begin_rename(id, title, window, cx);
                                    });
                                })),
                        ),
                )
                .child(
                    div()
                        .id(("archive-slot", idx))
                        .debug_selector(move || format!("archive-x-{idx}"))
                        .on_mouse_down(
                            gpui::MouseButton::Left,
                            cx.listener(|_, _, _, cx| cx.stop_propagation()),
                        )
                        .on_mouse_up(
                            gpui::MouseButton::Left,
                            cx.listener(|_, _, _, cx| cx.stop_propagation()),
                        )
                        .child(
                            Button::new(("archive-space", idx))
                                .ghost()
                                .xsmall()
                                .icon(IconName::Close)
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    cx.stop_propagation();
                                    this.archive(archive_id.clone(), cx);
                                })),
                        ),
                );
        }

        let mut row = h_flex()
            .id(("space-row", idx))
            .w_full()
            .h(ROW_H)
            .gap_3()
            .items_center()
            .cursor_pointer()
            .on_action(cx.listener(|this, _: &CancelRename, _, cx| this.cancel_rename(cx)))
            .on_hover(cx.listener(move |this, hovering: &bool, _, cx| {
                this.set_row_hover(idx, *hovering, cx);
            }));

        if !is_renaming {
            // A single click opens the space. Rename is reached via the
            // hover-revealed pencil button (see `reveal_slot`), not a
            // double-click — a single click opens the row immediately, so the
            // second click of a double landed in the new window, making the old
            // double-click trigger unreachable.
            row = row.on_click(cx.listener(move |this, _, _, cx| {
                this.open_space(space_id.clone(), cx);
            }));
        }

        row = row
            .child(title_content)
            .child(
                div()
                    .text_sm()
                    .whitespace_nowrap()
                    .text_color(theme.muted_foreground)
                    .child(SharedString::from(relative_date(
                        space.last_activity_at,
                        eidola_app_core::now_ms(),
                    ))),
            )
            .child(reveal_slot);

        // Hairline rule between entries — a rule *between*, not a box
        // around, so the first row carries no leading rule.
        if idx > 0 {
            row = row.border_t_1().border_color(theme.border);
        }
        row
    }
}

impl Render for LibraryView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let count = self.spaces.read(cx).list().len();

        let mut root = v_flex()
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(|_, _: &CloseWindow, window, _| {
                window.remove_window();
            }))
            .size_full()
            .bg(theme.background)
            .text_color(theme.foreground)
            .pt(TITLE_BAR_RESERVE);

        // Chapter-style heading: a small italic label between hairline
        // rules, echoing the chat's chapter delimiters so the library reads
        // as another page of the same book.
        root = root.child(
            h_flex()
                .w_full()
                .items_center()
                .gap_4()
                .px_10()
                .pt_4()
                .pb_2()
                .child(div().h(px(1.)).flex_1().bg(theme.border))
                .child(
                    div()
                        .text_sm()
                        .italic()
                        .text_color(theme.muted_foreground)
                        .child("Library"),
                )
                .child(div().h(px(1.)).flex_1().bg(theme.border)),
        );

        if count == 0 {
            return root.child(
                h_flex()
                    .flex_1()
                    .w_full()
                    .justify_center()
                    .items_center()
                    .child(
                        div()
                            .text_color(theme.muted_foreground)
                            .child("Nothing here yet — ⌘N starts a new space."),
                    ),
            );
        }

        // The listing is virtualized: `uniform_list` renders only the visible
        // window of rows, so frame work is O(visible), not O(loaded). The
        // list self-scrolls; the centering wrapper caps it at the prose
        // measure and keeps it centered like the unvirtualized layout.
        let list = uniform_list(
            "library-list",
            count,
            cx.processor(|this, range: Range<usize>, _window, cx| this.render_rows(range, cx)),
        )
        .h_full()
        .w_full()
        .max_w(rems(34.))
        .px_10()
        .pt_4()
        .track_scroll(&self.scroll);

        root.child(
            h_flex()
                .w_full()
                .flex_1()
                .min_h_0()
                .justify_center()
                .child(list),
        )
    }
}

/// Quiet relative date for the listing: "today", "yesterday", "3d ago",
/// "2w ago", "4mo ago", "1y ago". Coarse on purpose — a table of contents
/// wants a sense of recency, not a timestamp.
fn relative_date(then_ms: i64, now_ms: i64) -> String {
    const MINUTE: i64 = 60_000;
    const HOUR: i64 = 60 * MINUTE;
    const DAY: i64 = 24 * HOUR;

    let delta = (now_ms - then_ms).max(0);
    if delta < DAY {
        "today".to_string()
    } else if delta < 2 * DAY {
        "yesterday".to_string()
    } else if delta < 7 * DAY {
        format!("{}d ago", delta / DAY)
    } else if delta < 30 * DAY {
        format!("{}w ago", delta / (7 * DAY))
    } else if delta < 365 * DAY {
        format!("{}mo ago", delta / (30 * DAY))
    } else {
        format!("{}y ago", delta / (365 * DAY))
    }
}
