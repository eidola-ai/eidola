//! The **space inspector** — the per-space settings panel inside the space
//! window (task 26, wave 26.2).
//!
//! Per-space settings belong visually inside the space, so the inspector is a
//! **real split** of the window rather than an overlay: the conversation keeps
//! its own page (its reading measure, composer, minimap — everything reads the
//! narrowed width through [`SpaceView::page_width`]) and the inspector sits
//! beside it. Only when the window is too narrow for both does it float over
//! the conversation behind a scrim — see [`inspector_layout`], which is the
//! whole decision and is pure.
//!
//! **There is deliberately no visual toggle in the space** (Mike, 2026-08-01):
//! the space stays clean, and the doors are `Space ▸ Show/Hide Inspector` and
//! ⌥⌘I. Open state is per window (a view field), like every other window-local
//! UI state in STATE.md's scoping table.
//!
//! **Voice.** Settings are not part of the app's content interactions, so the
//! panel is *not* forced into the paper feel: it inherits the window's chrome
//! type (the 14px UI ramp Settings uses) rather than `prose_style`, with quiet
//! section headers and label/control rows — the Eidola ▸ Settings language,
//! sitting beside the paper rather than pretending to be it.
//!
//! **Where the data lives.** The cascade limit and router model are per-space
//! *domain* state and live in [`crate::stores::SpaceSettingsStore`] (keyed per
//! space, refreshed on `Change::Space`), so two windows on one space agree and
//! neither owns a private copy. The **title** is the Library index's (written
//! through `SpacesStore::rename`, exactly as the Library's inline rename does),
//! which is also what names the window. Only the transient UI bits — open,
//! scroll, which picker is open, the title editor's buffer — are view fields.

use gpui::{
    AnyElement, AppContext, Context, Entity, Focusable as _, InteractiveElement, IntoElement,
    ParentElement, Pixels, ScrollHandle, SharedString, StatefulInteractiveElement, Styled, Window,
    div, prelude::FluentBuilder as _, px,
};
use gpui_component::{
    ActiveTheme, StyledExt as _, h_flex,
    input::{Input, InputEvent, InputState},
    v_flex,
};

use crate::focus::TabRegion as _;
use crate::overlay::{Contain as _, Overlay};
use crate::participants_view::{
    RouterField, error_banner, field_label, ghost_button_labeled, load_error_panel, router_field,
};
use crate::probe::Probe as _;

use super::{SpaceView, TITLE_BAR_RESERVE};

/// The inspector's width when it splits the window. Wide enough for a model
/// picker's `name · backend` line without stealing a whole reading column.
pub(crate) const INSPECTOR_WIDTH: Pixels = px(320.);

/// The narrowest the conversation pane may become before the inspector stops
/// splitting and starts overlaying. It is the minimum width we let a space
/// *window* be (`lib.rs::open_chat_window`), so a split pane is never narrower
/// than a window the user could already have made — below that the conversation
/// would be a sliver, and covering it honestly beats squeezing it.
pub(crate) const MIN_CONTENT_WIDTH: Pixels = px(480.);

/// The mandatory cost copy under this space's router row when it holds a
/// **remote** reference: a remote router bills an inference on every post here,
/// where an engine-served one is genuinely free. Always visible, never a
/// tooltip.
pub const ROUTER_REMOTE_COST_NOTE: &str = "Every post in this space is routed through this model, billed per call. \
     Local models route free.";

/// What the router does, said once under the row.
const ROUTER_HELP: &str = "A small model that decides which participants a post is worth waking. When off, \
     notifications simply follow each participant's notify setting.";

/// How the inspector sits in the window this frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InspectorLayout {
    /// Closed — the conversation has the whole window.
    Hidden,
    /// A real split: the conversation compresses, the inspector takes its own
    /// column beside it.
    Split,
    /// Too narrow for both: the inspector floats over the conversation behind a
    /// scrim (the conversation keeps its full width underneath).
    Overlay,
}

/// Decide the frame's inspector layout from the window's content width.
///
/// The rule, in order: closed wins; then **the content column compresses
/// first** — the inspector keeps its width and the conversation gives up its
/// gutters and measure — until the pane would fall below [`MIN_CONTENT_WIDTH`],
/// at which point the inspector overlays instead of squeezing further.
pub(crate) fn inspector_layout(open: bool, viewport_width: Pixels) -> InspectorLayout {
    if !open {
        return InspectorLayout::Hidden;
    }
    if viewport_width - INSPECTOR_WIDTH >= MIN_CONTENT_WIDTH {
        InspectorLayout::Split
    } else {
        InspectorLayout::Overlay
    }
}

impl InspectorLayout {
    /// The width the conversation pane gets. Only a split takes any away — an
    /// overlaying inspector covers the page rather than resizing it, so the
    /// reader's layout doesn't reflow when it appears.
    pub(crate) fn content_width(self, viewport_width: Pixels) -> Pixels {
        match self {
            InspectorLayout::Split => viewport_width - INSPECTOR_WIDTH,
            _ => viewport_width,
        }
    }
}

impl SpaceView {
    /// The conversation pane's size — the window's content box less whatever
    /// the inspector's split takes. **Every page-width consumer reads this**,
    /// not `chrome::content_size` directly: the composer's dock geometry, the
    /// minimap, branch offsets and the context menu all live inside the pane.
    pub(crate) fn page_size(&self, window: &Window) -> gpui::Size<Pixels> {
        let mut size = crate::chrome::content_size(window);
        size.width = self.inspector_layout(size.width).content_width(size.width);
        size
    }

    pub(crate) fn page_width(&self, window: &Window) -> Pixels {
        self.page_size(window).width
    }

    pub(crate) fn inspector_layout(&self, viewport_width: Pixels) -> InspectorLayout {
        inspector_layout(self.inspector_open, viewport_width)
    }

    /// `Space ▸ Show/Hide Inspector` (⌥⌘I) — the only door, by design.
    pub fn toggle_inspector(
        &mut self,
        _: &crate::actions::ToggleInspector,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.set_inspector_open(!self.inspector_open, window, cx);
    }

    /// The **one** door both openers and closers go through (the menu action,
    /// ⌥⌘I, the overlay scrim), which is what makes the focus handoff below a
    /// single place rather than a call at each close site.
    pub(crate) fn set_inspector_open(
        &mut self,
        open: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.inspector_open = open;
        if open {
            self.ensure_inspector_settings(cx);
        } else {
            self.inspector_router_picker = false;
            // **Focus comes back from the panel** (`RecordView::close_detail`'s
            // rule). The title field is a view field that survives the close,
            // so its handle stays the window's focus while its element is gone
            // — a dead handle: keystrokes reach nothing, `focus_next` restarts
            // from the top of the window, and type-to-compose is inert until a
            // click revives it. Hand the keyboard to the conversation the panel
            // annotated.
            //
            // **Only from a panel that is actually holding it** — the
            // `overlay_borrowed_focus` rule. A reader composing beside an open
            // inspector never lent the keyboard, and yanking their caret to the
            // view root on a close would be exactly what they did not ask for.
            if self.inspector_field_focused(window, cx) {
                window.focus(&self.focus_handle, cx);
            }
        }
        cx.notify();
    }

    /// Ask for this space's settings once the panel can show them. Idempotent
    /// (`ensure` declines when a snapshot exists), and run again each frame the
    /// panel renders — a blank ⌘N space has no id to fetch by until its first
    /// post adopts one, and that moment is not a toggle.
    fn ensure_inspector_settings(&mut self, cx: &mut Context<Self>) {
        if let Some(id) = self.space.read(cx).id().map(str::to_string) {
            self.stores
                .space_settings
                .update(cx, |s, cx| s.ensure(id, cx));
        }
    }

    /// The conversation pane's width this frame (test seam): what the split
    /// actually takes away from the page.
    #[doc(hidden)]
    pub fn page_width_for_test(&self, window: &Window) -> f32 {
        self.page_width(window).as_f32()
    }

    /// Whether the inspector is open in this window (test seam + the driver).
    #[doc(hidden)]
    pub fn inspector_open_for_test(&self) -> bool {
        self.inspector_open
    }

    /// Whether the router dropdown is open (test seam).
    #[doc(hidden)]
    pub fn inspector_picker_open_for_test(&self) -> bool {
        self.inspector_router_picker
    }

    /// Open the router dropdown without a click (tests).
    #[doc(hidden)]
    pub fn inspector_toggle_router_picker_for_test(&mut self, cx: &mut Context<Self>) {
        self.inspector_toggle_router_picker(cx);
    }

    /// Press the cascade stepper without a click (tests).
    #[doc(hidden)]
    pub fn inspector_step_cascade_for_test(&mut self, delta: i64, cx: &mut Context<Self>) {
        self.inspector_step_cascade(delta, cx);
    }

    /// Open/close the inspector without a menu action (driver scenes, tests).
    #[doc(hidden)]
    pub fn set_inspector_open_for_test(
        &mut self,
        open: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.set_inspector_open(open, window, cx);
    }

    /// This space's settings cell, for the rows below.
    fn inspector_settings(
        &self,
        cx: &gpui::App,
    ) -> Option<crate::loadable::Loadable<eidola_app_core::SpaceSettings>> {
        let id = self.space.read(cx).id()?;
        Some(self.stores.space_settings.read(cx).settings(id).clone())
    }

    /// The cascade limit currently shown, if the settings have loaded.
    pub(crate) fn inspector_cascade(&self, cx: &gpui::App) -> Option<i64> {
        self.inspector_settings(cx)?
            .value()
            .map(|s| s.cascade_limit)
    }

    #[doc(hidden)]
    pub fn inspector_cascade_for_test(&self, cx: &gpui::App) -> Option<i64> {
        self.inspector_cascade(cx)
    }

    /// The router reference currently shown — outer `None` = no settings
    /// loaded, inner `None` = Off (test seam).
    #[doc(hidden)]
    pub fn inspector_router_for_test(&self, cx: &gpui::App) -> Option<Option<String>> {
        self.inspector_settings(cx)?
            .value()
            .map(|s| s.router_model.clone())
    }

    /// The title editor's live buffer (test seam).
    #[doc(hidden)]
    pub fn inspector_title_state_for_test(&self) -> Option<Entity<InputState>> {
        self.inspector_title.as_ref().map(|(s, _)| s.clone())
    }

    // -- Writes ------------------------------------------------------------

    pub(crate) fn inspector_step_cascade(&mut self, delta: i64, cx: &mut Context<Self>) {
        let Some(space_id) = self.space.read(cx).id().map(str::to_string) else {
            return;
        };
        let Some(current) = self.inspector_cascade(cx) else {
            return; // nothing loaded yet — a stepper with no value writes nothing
        };
        let next = (current + delta).clamp(1, 99);
        if next == current {
            return;
        }
        self.stores
            .space_settings
            .update(cx, |s, cx| s.set_cascade_limit(space_id, next, cx));
        cx.notify();
    }

    pub(crate) fn inspector_set_router(&mut self, model_id: Option<&str>, cx: &mut Context<Self>) {
        self.inspector_router_picker = false;
        let Some(space_id) = self.space.read(cx).id().map(str::to_string) else {
            return;
        };
        let value = model_id.map(str::to_string);
        self.stores
            .space_settings
            .update(cx, |s, cx| s.set_router_model(space_id, value, cx));
        cx.notify();
    }

    /// Whether one of the inspector's text fields holds the window's focus —
    /// the conversation's keyboard model yields to it (see
    /// [`SpaceView::handle_conversation_key`]).
    pub(crate) fn inspector_field_focused(&self, window: &Window, cx: &gpui::App) -> bool {
        self.inspector_title
            .as_ref()
            .is_some_and(|(state, _)| state.read(cx).focus_handle(cx).is_focused(window))
    }

    /// Close the router dropdown, reporting whether it was open — the Escape
    /// rung the view root owns (the context-menu pattern).
    pub(crate) fn close_inspector_picker(&mut self, cx: &mut Context<Self>) -> bool {
        if !self.inspector_router_picker {
            return false;
        }
        self.inspector_router_picker = false;
        cx.notify();
        true
    }

    pub(crate) fn inspector_toggle_router_picker(&mut self, cx: &mut Context<Self>) {
        self.inspector_router_picker = !self.inspector_router_picker;
        if self.inspector_router_picker {
            // A freshly opened picker starts at the top.
            self.inspector_picker_scroll = ScrollHandle::new();
        }
        cx.notify();
    }

    pub(crate) fn inspector_retry_settings(&mut self, cx: &mut Context<Self>) {
        if let Some(id) = self.space.read(cx).id().map(str::to_string) {
            self.stores
                .space_settings
                .update(cx, |s, cx| s.refresh(id, cx));
        }
        cx.notify();
    }

    /// Commit the title field. Empty means "no title" here — a space's title is
    /// generated, so blanking the field is a mistake rather than an intent, and
    /// the field re-seeds from the stored value on the next frame it is not
    /// being typed in.
    pub fn inspector_commit_title(&mut self, cx: &mut Context<Self>) {
        let Some(space_id) = self.space.read(cx).id().map(str::to_string) else {
            return;
        };
        let Some((state, _)) = self.inspector_title.as_ref() else {
            return;
        };
        let title = state.read(cx).value().trim().to_string();
        if title.is_empty() || Some(title.as_str()) == self.space_title(cx).as_deref() {
            // Nothing to write — but the field is not necessarily showing the
            // stored title either (it was blanked, or padded with whitespace).
            // **Every rejection invalidates the seed**, because the seed is the
            // only thing `sync_inspector_title` consults: leave it equal to the
            // stored title and the sync reads "already synchronized" and never
            // repairs the field, so a cleared title stays blank on screen while
            // the space is still called what it was called.
            self.inspector_title_seed = None;
            cx.notify();
            return;
        }
        self.inspector_title_seed = Some(title.clone().into());
        self.stores
            .spaces
            .update(cx, |s, cx| s.rename(space_id, title, cx));
        cx.notify();
    }

    /// Keep the title field in step with the space's real title: mint it on
    /// first use, and re-seed it whenever the stored title moves **while the
    /// field is not focused** (another window's rename, or the auto-title
    /// landing after the first exchange). Typing is never clobbered.
    fn sync_inspector_title(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let title = self.space_title(cx).unwrap_or_default();
        let Some((state, _)) = self.inspector_title.as_ref() else {
            let state = cx.new(|cx| {
                InputState::new(window, cx)
                    .placeholder("Untitled space")
                    .default_value(title.to_string())
            });
            let sub = cx.subscribe_in(&state, window, |this, _, ev: &InputEvent, _, cx| match ev {
                InputEvent::PressEnter { .. } | InputEvent::Blur => this.inspector_commit_title(cx),
                _ => {}
            });
            self.inspector_title_seed = Some(title);
            self.inspector_title = Some((state, sub));
            return;
        };
        if state.read(cx).focus_handle(cx).is_focused(window) {
            return;
        }
        if self.inspector_title_seed.as_ref() == Some(&title) {
            return;
        }
        let state = state.clone();
        state.update(cx, |s, cx| s.set_value(title.to_string(), window, cx));
        self.inspector_title_seed = Some(title);
    }

    // -- Render ------------------------------------------------------------

    /// The inspector as this frame's layout wants it: nothing when hidden, the
    /// in-flow column for a split, and the scrim + floating panel for an
    /// overlay. Rendered as the space root's **last** children so the overlay
    /// form paints over everything it covers (the containment rule).
    pub(crate) fn render_inspector(
        &mut self,
        layout: InspectorLayout,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Vec<AnyElement> {
        if layout == InspectorLayout::Hidden {
            return Vec::new();
        }
        self.ensure_inspector_settings(cx);
        self.sync_inspector_title(window, cx);
        // The panel meets the window's right edge in both forms, so it owns
        // those corner notches under Linux CSD (no-ops elsewhere).
        let panel = crate::chrome::round_br_client_corner(
            crate::chrome::round_tr_client_corner(self.render_inspector_panel(window, cx), window),
            window,
        );
        match layout {
            InspectorLayout::Hidden => Vec::new(),
            InspectorLayout::Split => vec![
                panel
                    .h_full()
                    .w(INSPECTOR_WIDTH)
                    .flex_shrink_0()
                    .into_any_element(),
            ],
            InspectorLayout::Overlay => {
                // The scrim: an opaque-to-the-mouse wash over the conversation,
                // painted before the panel and after everything it covers. A
                // click on it closes the inspector — the pointer's way out of a
                // surface that is covering the reader's page (the menu and ⌥⌘I
                // remain the only *visible* doors).
                let scrim = div()
                    .id("space-inspector-scrim")
                    .absolute()
                    .top_0()
                    .left_0()
                    .right_0()
                    .bottom_0()
                    .bg(gpui::black().opacity(0.28))
                    .contain_mouse(Overlay::Popover)
                    .on_click(cx.listener(|this, _, window, cx| {
                        this.set_inspector_open(false, window, cx)
                    }));
                vec![
                    scrim.into_any_element(),
                    panel
                        .absolute()
                        .top_0()
                        .right_0()
                        .bottom_0()
                        .w(INSPECTOR_WIDTH)
                        .shadow_lg()
                        .contain_mouse(Overlay::Scrolling)
                        .into_any_element(),
                ]
            }
        }
    }

    /// The panel itself — chrome type, a quiet section header, ruled rows.
    fn render_inspector_panel(
        &self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> gpui::Stateful<gpui::Div> {
        let (bg, border) = {
            let theme = cx.theme();
            (theme.sidebar, theme.border)
        };
        // The panel's top band is draggable like the rest of the window's
        // chrome — the reader shouldn't lose the grab strip where the inspector
        // covers it.
        let drag_band = crate::titlebar::drag_band(
            "space-inspector-titlebar",
            crate::titlebar::DRAG_BAND_HEIGHT,
            window,
            cx,
        );
        v_flex()
            .id("space-inspector")
            // A complementary landmark: settings *about* the conversation,
            // beside it. Role-less containers collapse in the a11y tree, and
            // the panel must be reachable from the landmark rotor.
            .probe("space/inspector", gpui::Role::Complementary, "Inspector")
            // After the conversation and the floating chrome in Tab order — it
            // is secondary to the page it annotates.
            .tab_region(crate::focus::region::AUX)
            .relative()
            .h_full()
            .w_full()
            .bg(bg)
            .border_l_1()
            .border_color(border)
            .child(
                div()
                    .relative()
                    .flex_1()
                    .min_h_0()
                    .w_full()
                    .child(
                        div()
                            .id("space-inspector-body")
                            .w_full()
                            .h_full()
                            .overflow_y_scroll()
                            .track_scroll(&self.inspector_scroll)
                            .child(
                                v_flex()
                                    .w_full()
                                    .px_4()
                                    .pt(TITLE_BAR_RESERVE)
                                    .pb_5()
                                    .gap_3()
                                    .child(self.render_inspector_space_section(cx)),
                            ),
                    )
                    .child(crate::scrollbar::vertical(
                        "space-inspector-scrollbar",
                        &self.inspector_scroll,
                        window,
                    )),
            )
            .child(drag_band)
    }

    /// Section 1 — **Space**: title, cascade limit, router model.
    fn render_inspector_space_section(&self, cx: &mut Context<Self>) -> AnyElement {
        let (muted, link, fg) = {
            let theme = cx.theme();
            (theme.muted_foreground, theme.link, theme.foreground)
        };
        let mut col = v_flex().w_full().gap_2().child(
            div()
                .pt_2()
                .text_sm()
                .font_medium()
                .text_color(muted)
                .child("Space"),
        );

        // A blank ⌘N space has no id yet, so it has no settings row to edit.
        // Say so rather than showing controls that would write nowhere.
        let Some(cell) = self.inspector_settings(cx) else {
            return col
                .child(
                    div()
                        .id("space-inspector-unsaved")
                        .probe_value(
                            "space/inspector/unsaved",
                            gpui::Role::Label,
                            "Space settings",
                            "This space is saved with its first post. Its settings appear then.",
                        )
                        .text_xs()
                        .text_color(muted)
                        .child(
                            "This space is saved with its first post. Its settings appear then.",
                        ),
                )
                .into_any_element();
        };

        let space_id = self
            .space
            .read(cx)
            .id()
            .map(str::to_string)
            .unwrap_or_default();
        // One banner for this panel's write refusals, from either store that
        // takes a write from it: the settings rows (`SpaceSettingsStore`) and
        // the title (`SpacesStore`, which owns the Library index). The index is
        // a store-wide snapshot, so its refusal is read **tagged with this
        // space** — a rename refused in another window's space belongs under
        // that space's field, not this one's.
        let op_error = self
            .stores
            .space_settings
            .read(cx)
            .op_error(&space_id)
            .map(str::to_string)
            .or_else(|| {
                self.stores
                    .spaces
                    .read(cx)
                    .op_error_for(&space_id)
                    .map(str::to_string)
            });
        let load_error = cell.error().map(|e| e.to_string());

        // Title first: it is the space's name, and renaming works whether or
        // not the settings row loaded (it is the Library index's field).
        col = col.child(field_label("Title", cx)).child(
            div()
                .id("space-inspector-title-wrap")
                .w_full()
                // The `Input` owns the focus and is therefore the accessible
                // node (the two-regime rule); the wrapper is bounds-only.
                .probe_bounds(
                    "space/inspector/title",
                    gpui::Role::TextInput,
                    "Space title",
                )
                .when_some(self.inspector_title.as_ref(), |el, (state, _)| {
                    el.child(Input::new(state).aria_label("Space title"))
                }),
        );

        // "Failed is not empty": a failed *initial* read must not render as a
        // plausible default (cascade 4, router Off) with live controls that
        // would write over settings we never managed to read.
        if load_error.is_some() && !cell.has_value() {
            col = col.child(load_error_panel(
                "space/inspector/retry",
                "Couldn't load this space's settings.",
                load_error.as_deref().unwrap_or_default(),
                cx,
                cx.listener(|this, _, _, cx| this.inspector_retry_settings(cx)),
            ));
            if let Some(err) = op_error {
                col = col.child(self.render_inspector_error(&err, cx));
            }
            return col.into_any_element();
        }

        let Some(settings) = cell.value() else {
            // Not loaded yet — a quiet placeholder, never a fake default.
            return col
                .child(
                    div()
                        .text_xs()
                        .text_color(muted.opacity(0.8))
                        .child("Loading…"),
                )
                .into_any_element();
        };

        col = col
            .child(self.render_inspector_cascade(settings.cascade_limit, cx))
            .child(field_label("Router model", cx))
            .child(router_field(
                &self.stores,
                RouterField {
                    id_prefix: "space-inspector-router",
                    probe_prefix: "space/inspector/router",
                    selection: settings.router_model.as_deref(),
                    open: self.inspector_router_picker,
                    cost_note: ROUTER_REMOTE_COST_NOTE,
                    help: ROUTER_HELP,
                    picker_scroll: &self.inspector_picker_scroll,
                    scrollbar_id: "space-inspector-router-scrollbar",
                },
                cx,
                |this, _, _, cx| this.inspector_toggle_router_picker(cx),
                |id, this: &mut Self, cx| this.inspector_set_router(id, cx),
            ));

        if let Some(err) = op_error {
            col = col.child(self.render_inspector_error(&err, cx));
        }
        // A failed *refresh* over a value we still hold: keep the rows, offer a
        // quiet retry beside them.
        if load_error.is_some() {
            col = col.child(
                div()
                    .id("space-inspector-refresh-retry")
                    .probe("space/inspector/retry", gpui::Role::Button, "Retry")
                    .cursor_pointer()
                    .text_xs()
                    .text_color(link)
                    .hover(move |s| s.text_color(fg))
                    .child("Couldn't refresh — retry")
                    .on_click(cx.listener(|this, _, _, cx| this.inspector_retry_settings(cx))),
            );
        }
        col.into_any_element()
    }

    /// The cascade-limit row: label left, a small −/+ stepper right. The value
    /// carries its own settled `Label` node — otherwise the number a screen
    /// reader most wants is the one thing the row doesn't say.
    fn render_inspector_cascade(&self, limit: i64, cx: &mut Context<Self>) -> AnyElement {
        let theme = cx.theme();
        v_flex()
            .w_full()
            .gap_1()
            .child(
                h_flex()
                    .w_full()
                    .items_center()
                    .justify_between()
                    .gap_2()
                    .child(field_label("Cascade limit", cx))
                    .child(
                        h_flex()
                            .gap_1()
                            .items_center()
                            .child(ghost_button_labeled(
                                "space-inspector-cascade-dec".into(),
                                "space/inspector/cascade/dec".into(),
                                "−",
                                "Decrease cascade limit",
                                false,
                                cx,
                                cx.listener(|this, _, _, cx| this.inspector_step_cascade(-1, cx)),
                            ))
                            .child(
                                div()
                                    .id("space-inspector-cascade-value")
                                    .probe_value(
                                        "space/inspector/cascade",
                                        gpui::Role::Label,
                                        "Cascade limit",
                                        SharedString::from(limit.to_string()),
                                    )
                                    .min_w(px(24.))
                                    .text_center()
                                    .text_sm()
                                    .child(SharedString::from(limit.to_string())),
                            )
                            .child(ghost_button_labeled(
                                "space-inspector-cascade-inc".into(),
                                "space/inspector/cascade/inc".into(),
                                "+",
                                "Increase cascade limit",
                                false,
                                cx,
                                cx.listener(|this, _, _, cx| this.inspector_step_cascade(1, cx)),
                            )),
                    ),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(theme.muted_foreground.opacity(0.8))
                    .child("How many agent replies in a row before this space pauses."),
            )
            .into_any_element()
    }

    fn render_inspector_error(&self, err: &str, cx: &Context<Self>) -> AnyElement {
        div()
            .id("space-inspector-error")
            .probe("space/inspector/error", gpui::Role::Alert, err.to_string())
            .child(error_banner(err, cx))
            .into_any_element()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_closed_inspector_leaves_the_whole_window_to_the_conversation() {
        let layout = inspector_layout(false, px(1200.));
        assert_eq!(layout, InspectorLayout::Hidden);
        assert_eq!(layout.content_width(px(1200.)), px(1200.));
    }

    #[test]
    fn the_content_column_compresses_before_the_inspector_overlays() {
        // Comfortably wide: a real split, and the conversation gives up width.
        let wide = inspector_layout(true, px(1200.));
        assert_eq!(wide, InspectorLayout::Split);
        assert_eq!(wide.content_width(px(1200.)), px(1200.) - INSPECTOR_WIDTH);

        // Exactly at the floor is still a split — the pane is as narrow as the
        // narrowest window we allow, not narrower.
        let edge = MIN_CONTENT_WIDTH + INSPECTOR_WIDTH;
        assert_eq!(inspector_layout(true, edge), InspectorLayout::Split);
        assert_eq!(
            inspector_layout(true, edge).content_width(edge),
            MIN_CONTENT_WIDTH
        );
    }

    #[test]
    fn below_the_floor_the_inspector_overlays_and_the_page_keeps_its_width() {
        let narrow = MIN_CONTENT_WIDTH + INSPECTOR_WIDTH - px(1.);
        let layout = inspector_layout(true, narrow);
        assert_eq!(layout, InspectorLayout::Overlay);
        assert_eq!(
            layout.content_width(narrow),
            narrow,
            "an overlay covers the page rather than reflowing it"
        );
    }
}
