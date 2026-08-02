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
    App, AppContext, Context, Entity, FocusHandle, Focusable as _, InteractiveElement, IntoElement,
    ParentElement, Render, SharedString, StatefulInteractiveElement, Styled, Subscription,
    UniformListScrollHandle, Window, actions, div, prelude::FluentBuilder as _, px, rems,
    uniform_list,
};
use gpui_component::{
    ActiveTheme, IconName, Sizable,
    button::{Button, ButtonVariants},
    h_flex,
    input::{Input, InputEvent, InputState},
    v_flex,
};

use crate::actions::CloseWindow;
use crate::focus::TabRegion as _;
use crate::probe::Probe as _;
use crate::stores::{SpacesStore, Stores};

actions!(library, [CancelRename]);

/// Where the roving cursor stands this frame, resolved once per rendered
/// window of rows — see [`LibraryView::cursor_and_reveal`] for the two
/// predicates, and why the ring takes a third.
#[derive(Clone, Copy)]
struct RowFocus {
    /// The row that *is* the cursor: ring identity and a11y focus.
    cursor: Option<usize>,
    /// The row that *draws* the ring — the cursor under keyboard modality.
    ring: Option<usize>,
    /// The row whose hover-gated verbs are revealed.
    revealed: Option<usize>,
}

impl RowFocus {
    fn resolve(view: &LibraryView, window: &Window, cx: &App) -> Self {
        let (cursor, revealed) = view.cursor_and_reveal(window, cx);
        Self {
            cursor,
            // A *programmatic* focus must not paint a keyboard cursor for a
            // pointer user.
            ring: cursor.filter(|_| window.last_input_was_keyboard()),
            revealed,
        }
    }
}

/// Vertical reserve at the top of the window — under the macOS traffic
/// lights, or hosting the Linux CSD window controls + drag strip (same
/// pattern as `space_view::TITLE_BAR_RESERVE`; the window uses the shared
/// transparent titlebar from `lib.rs::transparent_titlebar`).
const TITLE_BAR_RESERVE: gpui::Pixels = crate::titlebar::DRAG_BAND_HEIGHT;

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
    /// **The listing is one tab stop with a roving cursor** — the shape the
    /// space tree already ships, and the only shape a virtualized list can
    /// have. `uniform_list` materializes only the visible window, so a tab stop
    /// per row is a tab order that literally does not contain the rows you
    /// haven't scrolled to: Tab walked off the end of the visible slice and out
    /// of the library. So the list holds focus (this handle, the sole stop of
    /// the `MAIN` region), ↑/↓/Home/End move [`Self::focused_row`] — scrolling
    /// it into view — and Enter opens it. The cursor's row wears the ring and
    /// reveals its verbs (audit S7: gpui suppresses hover entirely under
    /// keyboard modality), and those verbs are ordinary tab stops painted
    /// inside the list, so Tab from the list reaches Rename then Archive for
    /// exactly that row.
    list_focus: FocusHandle,
    /// The roving cursor: which row the keyboard is on. Read through
    /// [`Self::cursor`], never directly — rows come and go under it (an
    /// archive, a bus-driven re-list, a rename that reorders), so the stored
    /// value is clamped at every use rather than chased at every mutation
    /// site. Always meaningful once clamped; it simply isn't *shown* while the
    /// list doesn't hold focus, so entering the listing lands on the first row.
    focused_row: usize,
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
            list_focus: cx
                .focus_handle()
                .tab_index(crate::focus::region::MAIN)
                .tab_stop(true),
            focused_row: 0,
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

    /// The **effective** cursor: clamped into the current listing, and `None`
    /// when the library is empty. Deriving it on read is what keeps a
    /// shrinking listing honest — archive the last row while the cursor is on
    /// it and the cursor lands on the new last row, rather than pointing one
    /// past the end where Enter is dead and no row draws the ring.
    fn cursor(&self, cx: &App) -> Option<usize> {
        self.spaces
            .read(cx)
            .list()
            .len()
            .checked_sub(1)
            .map(|last| self.focused_row.min(last))
    }

    /// The listing's roving-focus key map: ↑/↓ move the cursor, Home/End take
    /// its ends, Enter opens the row it sits on. Returns `true` when it
    /// consumed the press.
    ///
    /// Gated on the **list itself** holding focus, not on containing it: once
    /// Tab has moved on to the cursor row's Rename or Archive verb, that verb
    /// owns the keyboard — gpui fires its activation on key *up*, and a
    /// listener here would otherwise also open the space on the key *down* of
    /// the very same press.
    fn handle_list_key(
        &mut self,
        ev: &gpui::KeyDownEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        if !self.list_focus.is_focused(window) || ev.keystroke.modifiers.modified() {
            return false;
        }
        let count = self.spaces.read(cx).list().len();
        let (Some(last), Some(cursor)) = (count.checked_sub(1), self.cursor(cx)) else {
            return false;
        };
        let target = match ev.keystroke.key.as_str() {
            "up" => cursor.saturating_sub(1),
            "down" => (cursor + 1).min(last),
            "home" => 0,
            "end" => last,
            "enter" => {
                let Some(space) = self.spaces.read(cx).list().get(cursor) else {
                    return false;
                };
                let id = space.id.clone();
                self.open_space(id, cx);
                return true;
            }
            _ => return false,
        };
        self.focus_row(target, cx);
        true
    }

    /// Move the roving cursor to `idx` and scroll it into view. The scroll is
    /// what makes one tab stop equivalent to a per-row one: an off-screen row
    /// is materialized by the list before it can be read.
    fn focus_row(&mut self, idx: usize, cx: &mut Context<Self>) {
        self.focused_row = idx;
        self.scroll.scroll_to_item(idx, gpui::ScrollStrategy::Top);
        cx.notify();
    }

    /// **Two questions, two predicates** — `(cursor, revealed)`, and the row
    /// render reads exactly this pair, so the rule has one statement.
    ///
    /// The *cursor* — the ring and the row's focus identity — belongs to the row
    /// only while the list **itself** holds focus: Tab moving on to that row's
    /// Rename verb makes the verb the focused element, and it paints its own
    /// ring, so keeping the row's would be two focus indications for one focus
    /// (and would report a row as focused while the real focus is a button
    /// inside it). The *reveal* stays on `contains_focused`, because the verbs
    /// must not vanish out from under the Tab that just reached them.
    fn cursor_and_reveal(&self, window: &Window, cx: &App) -> (Option<usize>, Option<usize>) {
        let cursor = self.cursor(cx);
        (
            self.list_focus
                .is_focused(window)
                .then_some(cursor)
                .flatten(),
            self.list_focus
                .contains_focused(window, cx)
                .then_some(cursor)
                .flatten(),
        )
    }

    /// Test seam: whether the listing holds the window's focus — what ending an
    /// inline rename must hand back.
    #[doc(hidden)]
    pub fn list_is_focused_for_test(&self, window: &Window) -> bool {
        self.list_focus.is_focused(window)
    }

    /// Test seam over [`Self::cursor_and_reveal`] — the same computation the
    /// rows render from, so pinning it pins the render.
    #[doc(hidden)]
    pub fn cursor_and_reveal_for_test(
        &self,
        window: &Window,
        cx: &App,
    ) -> (Option<usize>, Option<usize>) {
        self.cursor_and_reveal(window, cx)
    }

    /// Test seam: where the roving cursor effectively sits.
    #[doc(hidden)]
    pub fn focused_row_for_test(&self, cx: &App) -> Option<usize> {
        self.cursor(cx)
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
                InputEvent::Blur => this.cancel_rename(window, cx),
                _ => {}
            },
        )];
        self.renaming = Some((space_id, input_state, subs));
        cx.notify();
    }

    /// Commit the in-progress rename — write the new title to the store and
    /// close the input.
    pub fn commit_rename(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if let Some((space_id, input_state, _)) = self.renaming.take() {
            let title = input_state.read(cx).value().to_string();
            let title = title.trim().to_string();
            if !title.is_empty() {
                self.spaces
                    .update(cx, |s, cx| s.rename(space_id, title, cx));
            }
            self.return_focus_to_list(&input_state, window, cx);
        }
        cx.notify();
    }

    /// Cancel an in-progress rename without persisting anything.
    pub fn cancel_rename(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if let Some((_, input, _)) = self.renaming.take() {
            self.return_focus_to_list(&input, window, cx);
        }
        cx.notify();
    }

    /// Hand focus back to the listing after an inline rename ends — but only
    /// if the rename's input still **held** it.
    ///
    /// `begin_rename` focuses the row's input, and ending the session removes
    /// that input, so without this the window keeps a handle whose element is
    /// gone: the dispatch tree has no node for it, the roving keys reach
    /// nothing, and `focus_next` restarts the walk from the top of the window.
    /// The same cure as `RecordView::close_detail`, for the same reason.
    ///
    /// The guard is what makes it safe to call from every exit. `Blur` ends a
    /// session precisely *because* focus went somewhere else — refocusing there
    /// would drag it back out of whatever the user just clicked. Enter and
    /// Escape end it while the input still has focus, which is the case that
    /// needs rescuing.
    fn return_focus_to_list(
        &mut self,
        input: &Entity<InputState>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if input.read(cx).focus_handle(cx).is_focused(window) {
            window.focus(&self.list_focus, cx);
        }
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
    fn render_rows(
        &mut self,
        range: Range<usize>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Vec<gpui::AnyElement> {
        let total = self.spaces.read(cx).list().len();
        let visible: Vec<(usize, SpaceInfo)> = self
            .spaces
            .read(cx)
            .list()
            .get(range.clone())
            .map(|slice| range.clone().zip(slice.iter().cloned()).collect())
            .unwrap_or_default();
        let focus = RowFocus::resolve(self, window, cx);
        visible
            .into_iter()
            .map(|(idx, space)| {
                self.render_row(idx, &space, total, focus, cx)
                    .into_any_element()
            })
            .collect()
    }

    fn render_row(
        &self,
        idx: usize,
        space: &SpaceInfo,
        total: usize,
        focus: RowFocus,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme();
        let is_keyboard_row = focus.cursor == Some(idx);
        // Hover **or** the keyboard cursor: gpui suppresses hover entirely
        // under keyboard modality, so without the second half a keyboard user
        // could reach a row and find its verbs gone (audit S7). This half uses
        // the *reveal* predicate, which survives Tab moving into the verbs.
        let hovered = self.hovered == Some(idx) || focus.revealed == Some(idx);
        let space_id = space.id.clone();
        let archive_id = space.id.clone();
        let rename_id = space.id.clone();
        let rename_title = space.title.clone();
        let is_renaming = self.renaming_space_id() == Some(space.id.as_str());
        // Accessible name for the row: the same text the title column renders.
        let row_label: SharedString = space
            .title
            .clone()
            .or_else(|| space.snippet.clone())
            .unwrap_or_else(|| "Untitled space".to_string())
            .into();

        // Title content: when this row is being renamed, show the inline
        // input; otherwise show the static title or snippet.
        let title_content: gpui::AnyElement = if is_renaming {
            if let Some((_, input_state, _)) = &self.renaming {
                // Ghost-styled inline input: no border/background chrome,
                // flex_1 so it fills the title column, same font as the row.
                // The probed wrapper carries the a11y role/label (probe the
                // wrapping div, not the gpui-component Input); it takes over
                // the input's flex_1 slot so bounds stay honest.
                div()
                    .id("rename-input-wrap")
                    .probe(
                        "library/rename-input",
                        gpui::Role::TextInput,
                        "Rename space",
                    )
                    .flex_1()
                    .flex()
                    .child(Input::new(input_state).flex_1())
                    .into_any_element()
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
                        .probe(
                            format!("library/row/{idx}/rename"),
                            gpui::Role::Button,
                            format!("Rename {row_label}"),
                        )
                        .on_click(cx.listener(move |_, _, window, cx| {
                            cx.stop_propagation();
                            let id = rename_id.clone();
                            let title = rename_title.clone();
                            cx.defer_in(window, move |this, window, cx| {
                                this.begin_rename(id, title, window, cx);
                            });
                        }))
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
                                .tab_stop(false),
                        ),
                )
                .child(
                    div()
                        .id(("archive-slot", idx))
                        .probe(
                            format!("library/row/{idx}/archive"),
                            gpui::Role::Button,
                            format!("Archive {row_label}"),
                        )
                        .on_click(cx.listener(move |this, _, _, cx| {
                            cx.stop_propagation();
                            this.archive(archive_id.clone(), cx);
                        }))
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
                                .tab_stop(false),
                        ),
                );
        }

        let mut row = h_flex()
            .id(("space-row", idx))
            // The list holds the keyboard and moves a cursor over its rows, so
            // a row is a *managed descendant*, never a tab stop — see
            // `probe_delegating` and [`Self::list_focus`].
            .probe_delegating(
                format!("library/row/{idx}"),
                gpui::Role::ListItem,
                row_label,
            )
            // `uniform_list` renders only the visible window, so without the
            // set metadata AT sees "6 spaces" in a library of six hundred.
            .aria_position_in_set(idx + 1)
            .aria_size_of_set(total)
            .aria_selected(is_keyboard_row)
            // The active-descendant half of the roving pattern: the focused
            // node is the `List` above, and this row is what AT should report
            // as focused inside it. gpui honours it only when the focused node
            // really is an ancestor of this one, which is exactly the shape
            // here (`uniform_list` contributes no node, so the row's a11y
            // parent *is* the list).
            .when(is_keyboard_row, |d| d.aria_active_descendant())
            // The cursor's ring is drawn here rather than by `focus_visible`,
            // because the focused *element* is the list; this is the row its
            // cursor is on. The modality guard is what keeps a *programmatic*
            // focus (the Record's back-out precedent) from painting a keyboard
            // cursor for a pointer user.
            .when(focus.ring == Some(idx), |d| {
                d.shadow(crate::focus::ring_shadows(crate::focus::ring_colors()))
            })
            .w_full()
            .h(ROW_H)
            .gap_3()
            .items_center()
            .cursor_pointer()
            .on_action(
                cx.listener(|this, _: &CancelRename, window, cx| this.cancel_rename(window, cx)),
            )
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
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // Built ahead of the chain: `drag_band` needs `&mut cx` (element-owned
        // arm state), which can't overlap the `theme` borrow below.
        let drag_band =
            crate::titlebar::drag_band("library-titlebar", TITLE_BAR_RESERVE, window, cx);
        let theme = cx.theme();
        let count = self.spaces.read(cx).list().len();
        // A "New Space from Template" failure is surfaced here (the natural home
        // for a failed new-space) rather than silently discarded by its owning
        // store task.
        let new_space_error = self.spaces.read(cx).new_space_error().map(str::to_string);

        let mut root = crate::chrome::round_client_corners(v_flex(), window)
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(|_, _: &CloseWindow, window, _| {
                window.remove_window();
            }))
            .relative()
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
                        .id("library-title")
                        .probe("library/title", gpui::Role::Heading, "Library")
                        .aria_level(1)
                        .text_sm()
                        .italic()
                        .text_color(theme.muted_foreground)
                        .child("Library"),
                )
                .child(div().h(px(1.)).flex_1().bg(theme.border)),
        );

        // A failed "New Space from Template" — a dismissible danger strip.
        if let Some(err) = new_space_error {
            root = root.child(
                h_flex()
                    .id("library-new-space-error")
                    .probe("library/new-space-error", gpui::Role::Alert, err.clone())
                    .mx_10()
                    .mb_2()
                    .px_3()
                    .py_2()
                    .gap_2()
                    .items_center()
                    .justify_between()
                    .rounded_md()
                    .bg(theme.danger.opacity(0.08))
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .text_sm()
                            .text_color(theme.danger)
                            .child(SharedString::from(format!(
                                "Couldn't create the space: {err}"
                            ))),
                    )
                    .child(
                        div()
                            .id("library-new-space-error-dismiss")
                            .probe(
                                "library/new-space-error/dismiss",
                                gpui::Role::Button,
                                "Dismiss",
                            )
                            .cursor_pointer()
                            .text_color(theme.muted_foreground)
                            .hover(|s| s.text_color(theme.foreground))
                            .child("×")
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.spaces.update(cx, |s, cx| s.clear_new_space_error(cx));
                            })),
                    ),
            );
        }

        if count == 0 {
            return root.child(
                h_flex()
                    .flex_1()
                    .w_full()
                    .justify_center()
                    .items_center()
                    .child(div().text_color(theme.muted_foreground).child(format!(
                        "Nothing here yet — {} starts a new space.",
                        crate::actions::primary_chord("N")
                    ))),
            );
        }

        // The listing is virtualized: `uniform_list` renders only the visible
        // window of rows, so frame work is O(visible), not O(loaded). The
        // list self-scrolls; the centering wrapper caps it at the prose
        // measure and keeps it centered like the unvirtualized layout.
        let list = uniform_list(
            "library-list",
            count,
            cx.processor(|this, range: Range<usize>, window, cx| {
                this.render_rows(range, window, cx)
            }),
        )
        .h_full()
        .w_full()
        .px_10()
        .pt_4()
        .track_scroll(&self.scroll);

        // The scroll indicator rides the *list's* right edge, not the window's:
        // a `relative` column capped at the prose measure holds the list plus
        // the overlay strip as siblings, so the indicator appears where the
        // scrollable content actually is on a wide window.
        // The drag band, over the traffic-light reserve, is appended **last**:
        // a blocking hitbox only suppresses hitboxes registered before it (see
        // `crate::overlay`), so a band painted first contains nothing. It is
        // absolute, so flow layout is unaffected either way.
        let root = root.child(drag_band);
        root.child(
            h_flex().w_full().flex_1().min_h_0().justify_center().child(
                div()
                    .id("library-list-wrap")
                    // The listing's container: `uniform_list` itself can't
                    // carry a role (it implements `InteractiveElement` but not
                    // `StatefulInteractiveElement`, where gpui's aria builders
                    // live), so the wrapper that already spans the viewport is
                    // where the `List` parent goes.
                    .probe("library/list", gpui::Role::List, "Spaces")
                    .tab_region(crate::focus::region::MAIN)
                    // **The listing's single tab stop lives on the element that
                    // carries the role.** `uniform_list` cannot take one
                    // (`InteractiveElement` but not `StatefulInteractiveElement`,
                    // where gpui's aria builders live), so a handle tracked
                    // there focuses a node the AccessKit tree does not contain
                    // — `A11y::set_focus` is a no-op for a node-less element and
                    // `TreeUpdate.focus` falls back to the window root. Tracked
                    // here, focus lands on the `List` node and the cursor row
                    // reports itself as its active descendant. The handle
                    // carries `tab_index(MAIN)` because gpui reads tab order off
                    // the *handle* once one is tracked, and the element's own
                    // `tab_region` index only opens the group.
                    .track_focus(&self.list_focus)
                    .on_key_down(cx.listener(|this, ev: &gpui::KeyDownEvent, window, cx| {
                        if this.handle_list_key(ev, window, cx) {
                            cx.stop_propagation();
                        }
                    }))
                    .relative()
                    .h_full()
                    .w_full()
                    .max_w(rems(34.))
                    .child(list)
                    .child(crate::scrollbar::vertical(
                        "library-scrollbar",
                        &self.scroll,
                        window,
                    )),
            ),
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
