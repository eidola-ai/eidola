//! Library window — the book's table of contents. Lists the user's spaces
//! (most recently active first), reopens one on click, and quietly archives
//! on the hover-revealed ×.
//!
//! Design notes: this is deliberately *not* a chat-app sidebar. One prose
//! column, hairline `theme.border` rules between entries, no cards or
//! avatars. Each row is a title (or, for untitled spaces, a muted snippet
//! of the first message) with a right-aligned relative date in `text_sm`
//! muted. An empty library is a single quiet line.

use eidola_app_core::SpaceInfo;
use gpui::{
    App, Context, Entity, FocusHandle, InteractiveElement, IntoElement, ParentElement, Render,
    SharedString, StatefulInteractiveElement, Styled, Subscription, Window, div, px, rems,
};
use gpui_component::{
    ActiveTheme, IconName, Sizable,
    button::{Button, ButtonVariants},
    h_flex, v_flex,
};

use crate::actions::CloseWindow;
use crate::stores::{SpacesStore, Stores};

/// Vertical reserve under the macOS traffic lights — same pattern as
/// `chat::TITLE_BAR_RESERVE` (the window uses the shared transparent
/// titlebar from `lib.rs::transparent_titlebar`).
#[cfg(target_os = "macos")]
const TITLE_BAR_RESERVE: gpui::Pixels = gpui::px(36.);
#[cfg(not(target_os = "macos"))]
const TITLE_BAR_RESERVE: gpui::Pixels = gpui::px(0.);

pub struct LibraryView {
    stores: Stores,
    spaces: Entity<SpacesStore>,
    /// Index of the row currently under the pointer, for the hover-revealed
    /// archive affordance.
    hovered: Option<usize>,
    /// Focus handle the root v_flex tracks, so the `CloseWindow` listener is
    /// in the dispatch path (same pattern as `SettingsView`).
    focus_handle: FocusHandle,
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
            focus_handle,
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

    /// Archive a space. Called by the hover-revealed × button; public so
    /// behavior tests can exercise the same path without synthesizing mouse
    /// events.
    pub fn archive(&mut self, space_id: String, cx: &mut Context<Self>) {
        self.spaces.update(cx, |s, cx| s.archive(space_id, cx));
    }

    /// Open the given space in a new chat window. Deferred so the window
    /// opens after the current update cycle completes.
    pub fn open_space(&mut self, space_id: String, cx: &mut Context<Self>) {
        let stores = self.stores.clone();
        cx.defer(move |cx: &mut App| {
            crate::open_space_window(cx, stores, space_id);
        });
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

        // Title line: the space's title in full foreground; for untitled
        // spaces, the snippet of the first message in muted — visibly a
        // fallback, not a name. A space with neither is brand new.
        let (line, is_fallback) = match (&space.title, &space.snippet) {
            (Some(t), _) => (t.clone(), false),
            (None, Some(s)) => (s.clone(), true),
            (None, None) => ("Untitled space".to_string(), true),
        };

        let mut title_el = div().flex_1().truncate().child(SharedString::from(line));
        if is_fallback {
            title_el = title_el.text_color(theme.muted_foreground);
        }

        // Fixed-width slot for the archive button so its hover appearance
        // doesn't shift the date column.
        let mut archive_slot = h_flex().w_6().justify_end();
        if hovered {
            archive_slot = archive_slot.child(
                Button::new(("archive-space", idx))
                    .ghost()
                    .xsmall()
                    .icon(IconName::Close)
                    .on_click(cx.listener(move |this, _, _, cx| {
                        cx.stop_propagation();
                        this.archive(archive_id.clone(), cx);
                    })),
            );
        }

        let mut row = h_flex()
            .id(("space-row", idx))
            .w_full()
            .py_3()
            .gap_3()
            .items_center()
            .cursor_pointer()
            .on_hover(cx.listener(move |this, hovering: &bool, _, cx| {
                this.hovered = if *hovering { Some(idx) } else { None };
                cx.notify();
            }))
            .on_click(cx.listener(move |this, _, _, cx| {
                this.open_space(space_id.clone(), cx);
            }))
            .child(title_el)
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
            .child(archive_slot);

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
        let spaces = self.spaces.read(cx).list().to_vec();

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

        if spaces.is_empty() {
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

        let mut list = v_flex().w_full().max_w(rems(34.)).px_10().pt_4().pb_8();
        for (idx, space) in spaces.iter().enumerate() {
            list = list.child(self.render_row(idx, space, cx));
        }

        root.child(
            div()
                .id("library-scroll")
                .w_full()
                .flex_1()
                .overflow_y_scroll()
                .child(h_flex().w_full().justify_center().child(list)),
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

#[cfg(test)]
mod tests {
    use super::relative_date;

    const DAY: i64 = 24 * 60 * 60 * 1000;

    #[test]
    fn relative_date_buckets() {
        let now = 1_900_000_000_000;
        assert_eq!(relative_date(now, now), "today");
        assert_eq!(relative_date(now - DAY / 2, now), "today");
        assert_eq!(relative_date(now - DAY - 1, now), "yesterday");
        assert_eq!(relative_date(now - 3 * DAY, now), "3d ago");
        assert_eq!(relative_date(now - 14 * DAY, now), "2w ago");
        assert_eq!(relative_date(now - 90 * DAY, now), "3mo ago");
        assert_eq!(relative_date(now - 400 * DAY, now), "1y ago");
        // Clock skew (future timestamps) clamps to "today".
        assert_eq!(relative_date(now + DAY, now), "today");
    }
}
