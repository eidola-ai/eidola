//! Settings window — a calm two-pane surface. A narrow nav list (General ·
//! Backends · Templates · Agents · Account · Wallet) sits on a `theme.sidebar` band down the left
//! edge; the selected pane renders in the content column. No primary-button
//! tab strip, no boxes-in-boxes: the nav is quiet text, the content is
//! hairline rows.
//!
//! **Account and Wallet are gated on the eidola backend being enabled** — the
//! account *is* the eidola backend's configuration, and with the backend
//! disabled ("on-device only") there is nothing to bill. Nav visibility
//! doubles as state; disabling eidola while one of those panes is selected
//! falls the selection back to Backends (see `effective_selected`). The
//! gating is optimistic: until the `BackendsStore` snapshot loads, the
//! singleton reads as enabled (matching `BackendsStore::is_enabled`), so the
//! panes render rather than flashing hidden on a cold open.
//!
//! Settings deliberately keeps **no raw-data dumps** — measurement hex,
//! attestation documents, and the request log live in the Record window
//! (⇧⌘L); the panes here summarize and link there.

use gpui::{
    AppContext, Context, Entity, FocusHandle, InteractiveElement, IntoElement, ParentElement,
    Render, ScrollHandle, StatefulInteractiveElement, Styled, Window, div, px,
};
use gpui_component::{ActiveTheme, h_flex, v_flex};

use crate::account::AccountView;
use crate::actions::CloseWindow;
use crate::agents_settings::AgentsSettingsView;
use crate::backends_settings::BackendsSettingsView;
use crate::focus::TabRegion as _;
use crate::general::GeneralView;
use crate::probe::Probe as _;
use crate::stores::{BackendsStore, Stores};
use crate::templates_settings::TemplatesSettingsView;
use crate::wallet::WalletView;

/// Vertical reserve at the top of the nav band so the macOS traffic lights
/// (at `point(14, 11)` per `lib.rs::transparent_titlebar`) / the Linux CSD
/// window controls sit on empty chrome rather than over the first nav item.
const NAV_TOP_RESERVE: gpui::Pixels = crate::titlebar::DRAG_BAND_HEIGHT;

/// Width of the nav band. Narrow on purpose — three words, not a sidebar.
const NAV_WIDTH: gpui::Pixels = gpui::px(132.);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SettingsPane {
    General,
    Backends,
    Templates,
    Agents,
    Account,
    Wallet,
}

impl SettingsPane {
    fn label(self) -> &'static str {
        match self {
            SettingsPane::General => "General",
            SettingsPane::Backends => "Backends",
            SettingsPane::Templates => "Templates",
            SettingsPane::Agents => "Agents",
            SettingsPane::Account => "Account",
            SettingsPane::Wallet => "Wallet",
        }
    }

    /// Whether this pane is only shown while the eidola backend is enabled.
    /// Account (its config) and Wallet (its credentials) both are.
    fn requires_eidola(self) -> bool {
        matches!(self, SettingsPane::Account | SettingsPane::Wallet)
    }
}

pub struct SettingsView {
    selected: SettingsPane,
    /// The pane whose `pane_activated` hook has already run for the current
    /// visit. All six panes are built at *window* creation, so construction
    /// is not activation; this is what tells the two apart. See
    /// `sync_active_pane`.
    activated: Option<SettingsPane>,
    general: Entity<GeneralView>,
    backends: Entity<BackendsSettingsView>,
    templates: Entity<TemplatesSettingsView>,
    agents: Entity<AgentsSettingsView>,
    account: Entity<AccountView>,
    wallet: Entity<WalletView>,
    /// The backend registry — read to gate the Account/Wallet nav items on
    /// the eidola singleton being enabled (observed so a bus-driven flip
    /// re-renders the nav and reconciles the selection).
    backends_store: Entity<BackendsStore>,
    /// Focus handle the root tracks. We attach `CloseWindow`'s listener to
    /// the root; the focused node has to be at-or-below it for the listener
    /// to be in the dispatch path, so we `focus()` the handle on
    /// construction.
    focus_handle: FocusHandle,
    /// Tracks the content column's scroll so the right-edge overlay indicator
    /// (shown only while scrolling) can bind to it. One handle for every pane —
    /// switching panes resets to the top, which the shared container does.
    body_scroll: ScrollHandle,
}

impl SettingsView {
    pub fn new(stores: Stores, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let general = cx.new(|cx| GeneralView::new(stores.config.clone(), window, cx));
        let backends = cx.new(|cx| BackendsSettingsView::new(stores.clone(), window, cx));
        let templates = cx.new(|cx| TemplatesSettingsView::new(stores.clone(), window, cx));
        let agents = cx.new(|cx| AgentsSettingsView::new(stores.clone(), window, cx));
        let account = cx.new(|cx| AccountView::new(stores.clone(), window, cx));
        let wallet = cx.new(|cx| WalletView::new(stores.clone(), window, cx));
        let backends_store = stores.backends.clone();

        // Observe the registry: an eidola enable/disable flip both re-renders
        // the nav (visibility) and reconciles a now-hidden selection back to
        // Backends (see `effective_selected`) — which is a pane change like
        // any other, so it runs the activation hook too.
        cx.observe(&backends_store, |this, _, cx| {
            this.selected = this.effective_selected(cx);
            this.sync_active_pane(cx);
            cx.notify();
        })
        .detach();

        let focus_handle = cx.focus_handle();
        focus_handle.focus(window, cx);

        let mut this = Self {
            selected: SettingsPane::General,
            activated: None,
            general,
            backends,
            templates,
            agents,
            account,
            wallet,
            backends_store,
            focus_handle,
            body_scroll: ScrollHandle::new(),
        };
        this.sync_active_pane(cx);
        this
    }

    /// Run the shown pane's activation hook when the shown pane changes.
    ///
    /// **Construction is not activation.** All six panes are built in
    /// `new`, at *window* creation, so a pane's constructor runs once for a
    /// reader who may never select it — and never again for one who selects
    /// it, leaves, and comes back. A pane whose data no `Change` can
    /// invalidate (see `AccountView`) therefore has no other moment to ask.
    ///
    /// Called from every place the *shown* pane can change and nowhere else:
    /// `new`, `select`, and the registry observer (the two inputs to
    /// `effective_selected`). Panes not listed here need nothing — their
    /// stores are refreshed by the invalidation bus.
    fn sync_active_pane(&mut self, cx: &mut Context<Self>) {
        let pane = self.effective_selected(cx);
        if self.activated == Some(pane) {
            return;
        }
        self.activated = Some(pane);
        match pane {
            SettingsPane::Account => self.account.update(cx, |v, cx| v.pane_activated(cx)),
            SettingsPane::General
            | SettingsPane::Backends
            | SettingsPane::Templates
            | SettingsPane::Agents
            | SettingsPane::Wallet => {}
        }
    }

    /// Whether the eidola backend is enabled (gates the Account/Wallet panes).
    /// Optimistic while the registry snapshot hasn't loaded — matches
    /// `BackendsStore::is_enabled`, so a cold open shows the panes rather than
    /// hiding them until the first fetch lands.
    fn eidola_enabled(&self, cx: &gpui::App) -> bool {
        self.backends_store.read(cx).is_enabled("eidola")
    }

    /// The pane actually shown: `selected`, unless it requires eidola and the
    /// backend is disabled — then Backends (never a blank body / phantom nav
    /// highlight).
    fn effective_selected(&self, cx: &gpui::App) -> SettingsPane {
        if self.selected.requires_eidola() && !self.eidola_enabled(cx) {
            SettingsPane::Backends
        } else {
            self.selected
        }
    }

    /// The nav panes currently visible: General and Backends always, then
    /// Account and Wallet while the eidola backend is enabled. The one source
    /// of truth for both the rendered nav and the gating behavior tests.
    pub fn visible_panes(&self, cx: &gpui::App) -> Vec<SettingsPane> {
        let mut panes = vec![
            SettingsPane::General,
            SettingsPane::Backends,
            SettingsPane::Templates,
            SettingsPane::Agents,
        ];
        if self.eidola_enabled(cx) {
            panes.push(SettingsPane::Account);
            panes.push(SettingsPane::Wallet);
        }
        panes
    }

    /// The focus handle the view tracks. Exposed so behavior tests can
    /// dispatch actions through it the same way real keystrokes would.
    pub fn focus_handle(&self) -> FocusHandle {
        self.focus_handle.clone()
    }

    pub fn selected(&self) -> SettingsPane {
        self.selected
    }

    /// Switch panes. Public so the nav rows and behavior tests share one
    /// path.
    pub fn select(&mut self, pane: SettingsPane, cx: &mut Context<Self>) {
        if self.selected != pane {
            self.selected = pane;
            self.sync_active_pane(cx);
            cx.notify();
        }
    }

    /// The General pane entity — exposed for behavior tests asserting the
    /// option-reveal state.
    pub fn general(&self) -> Entity<GeneralView> {
        self.general.clone()
    }

    /// The Backends pane entity — exposed for behavior tests asserting the
    /// backend + local-model affordances and the Eidola tab's trust surface.
    pub fn backends_pane(&self) -> Entity<BackendsSettingsView> {
        self.backends.clone()
    }

    /// The Account pane entity — a top-level pane again, exposed for tests
    /// asserting the reset-confirm / checkout flows.
    pub fn account_pane(&self) -> Entity<AccountView> {
        self.account.clone()
    }

    /// The Space Templates pane entity — exposed for behavior tests asserting
    /// the template CRUD + set-default flows.
    pub fn templates_pane(&self) -> Entity<TemplatesSettingsView> {
        self.templates.clone()
    }

    /// The Agents pane entity — exposed for behavior tests asserting the
    /// shared-agent edit / retire / notebook flows.
    pub fn agents_pane(&self) -> Entity<AgentsSettingsView> {
        self.agents.clone()
    }

    fn nav_item(
        &self,
        pane: SettingsPane,
        active_pane: SettingsPane,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let theme = cx.theme();
        let active = active_pane == pane;
        let mut item = div()
            .id(pane.label())
            .probe(
                format!("settings/nav/{}", pane.label().to_lowercase()),
                gpui::Role::Tab,
                pane.label(),
            )
            .aria_selected(active)
            .w_full()
            .px_2p5()
            .py_1()
            .rounded(px(6.))
            .cursor_pointer()
            .on_click(cx.listener(move |this, _, _, cx| this.select(pane, cx)))
            .child(pane.label());
        if active {
            item = item
                .bg(theme.sidebar_accent)
                .text_color(theme.sidebar_foreground);
        } else {
            item = item
                .text_color(theme.muted_foreground)
                .hover(|s| s.text_color(theme.sidebar_foreground));
        }
        item
    }
}

impl Render for SettingsView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // The pane actually shown, and the nav highlight, both track the
        // effective selection so disabling eidola while on Account/Wallet
        // never leaves a blank body or a phantom highlight.
        let effective = self.effective_selected(cx);
        let visible = self.visible_panes(cx);

        let body: gpui::AnyElement = match effective {
            SettingsPane::General => self.general.clone().into_any_element(),
            SettingsPane::Backends => self.backends.clone().into_any_element(),
            SettingsPane::Templates => self.templates.clone().into_any_element(),
            SettingsPane::Agents => self.agents.clone().into_any_element(),
            SettingsPane::Account => self.account.clone().into_any_element(),
            SettingsPane::Wallet => self.wallet.clone().into_any_element(),
        };

        // Nav items: General/Backends always, Account/Wallet gated. Built with
        // a plain loop (each `nav_item` takes `&mut cx` in turn) *before* the
        // `theme` borrow below, so the mutable-cx loop and the theme's
        // immutable borrow don't overlap.
        let mut nav_items: Vec<gpui::AnyElement> = Vec::with_capacity(visible.len());
        for pane in visible {
            nav_items.push(self.nav_item(pane, effective, cx).into_any_element());
        }

        let theme = cx.theme();
        crate::chrome::round_client_corners(h_flex(), window)
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(|_, _: &CloseWindow, window, _| {
                window.remove_window();
            }))
            .relative()
            .size_full()
            .items_start()
            .bg(theme.background)
            .text_color(theme.foreground)
            .child(
                crate::chrome::round_bl_client_corner(
                    crate::chrome::round_tl_client_corner(v_flex(), window),
                    window,
                )
                .id("settings-nav")
                // The band's items are already `Role::Tab`; naming their
                // container makes them a set rather than five loose tabs
                // hanging off the window root.
                .probe("settings/nav", gpui::Role::TabList, "Settings sections")
                .tab_region(crate::focus::region::NAV)
                .aria_orientation(gpui::Orientation::Vertical)
                .w(NAV_WIDTH)
                .h_full()
                .flex_none()
                .bg(theme.sidebar)
                .border_r_1()
                .border_color(theme.sidebar_border)
                .pt(NAV_TOP_RESERVE)
                .px_2()
                .gap_0p5()
                .children(nav_items),
            )
            // The scroll container needs the same width discipline as the
            // chat transcript (see the scroll-container invariant in
            // crates/eidola-gui/AGENTS.md): wrap it in a flex column that
            // owns the leftover width, and give the scroll div `.w_full()`
            // so taffy stretches it instead of content-sizing it — without
            // this, long pane text refuses to wrap and rows shrink to
            // content width.
            .child(
                v_flex()
                    .id("settings-content")
                    // The pane's landmark, named for the pane it holds — the
                    // one place AT can jump past the nav band to the settings
                    // themselves.
                    .probe(
                        "settings/content",
                        gpui::Role::Main,
                        format!("{} settings", effective.label()),
                    )
                    .tab_region(crate::focus::region::MAIN)
                    .relative()
                    .flex_1()
                    .h_full()
                    .min_w_0()
                    .child(
                        div()
                            .id("settings-body")
                            .w_full()
                            .flex_1()
                            .overflow_y_scroll()
                            .track_scroll(&self.body_scroll)
                            .child(body),
                    )
                    // Overlay indicator: a sibling of the scroll container, so
                    // it tracks the viewport's right edge without scrolling off.
                    .child(crate::scrollbar::vertical(
                        "settings-scrollbar",
                        &self.body_scroll,
                        window,
                    )),
            )
            // Drag band last so it paints atop the sidebar/body columns and
            // wins hit-testing across the full-width traffic-light reserve.
            .child(crate::titlebar::drag_band(
                "settings-titlebar",
                NAV_TOP_RESERVE,
                window,
                cx,
            ))
    }
}
