//! Settings ▸ Proxy — the local inference proxy's surface.
//!
//! A lens over [`ProxyStore`], which owns both halves of the answer: what the
//! reader asked for (the stored settings) and what is actually bound. **The
//! status line reads the listener**, never the setting — a surface that tells a
//! person where to point their tool must not name somewhere nothing is
//! listening.
//!
//! Five rows, in the order a reader needs them:
//!
//! 1. **Serve requests** — the switch, with the endpoint beside it.
//! 2. **Address** — edit-in-place (the Backends pane's idiom: the value is
//!    always visible, a quiet verb swaps in the input). A non-loopback address
//!    carries the danger band, because there is no TLS yet and the prompts go
//!    over the wire in the clear.
//! 3. **Backends** — a checkbox per configured backend. **Opt-in**: an empty
//!    set offers nothing, which is the honest default for a surface that can
//!    spend money.
//! 4. **On-device models** — only loaded, or every downloaded one (loading on
//!    demand).
//! 5. **API keys** — generated here and **shown once**. Only a digest is
//!    stored, so the pane cannot show a key again and says so where the key
//!    appears rather than in a note nobody reads afterwards.
//!
//! The pane's *nav* label is an English literal beside the other five (the nav
//! row's probe name derives from it, and a probe name is a stable selector that
//! must never localize); everything the pane itself says is Fluent.

use eidola_app_core::BackendInfo;
use eidola_app_core::proxy::{LocalExposure, ProxyKeyInfo, ProxySettings};
use gpui::{
    App, AppContext, ClipboardItem, Context, Entity, FocusHandle, Focusable, InteractiveElement,
    IntoElement, ParentElement, Render, SharedString, StatefulInteractiveElement, Styled, Window,
    div, prelude::FluentBuilder, px,
};
use gpui_component::{
    ActiveTheme, Sizable, StyledExt,
    input::{Input, InputState},
    switch::Switch,
    v_flex,
};
use gpui_component::{h_flex, label::Label};

use std::cell::RefCell;
use std::collections::HashMap;

use crate::i18n::msg;
use crate::participants::{ghost_button, ghost_button_labeled, load_error_panel};
use crate::probe::Probe as _;
use crate::stores::{BackendsStore, ProxyStore, Stores};

/// The subtrees a verb in this pane can unmount from under the keyboard.
///
/// **The class is "a verb whose press removes the verb"**, and it has five
/// members here rather than the two the binding editor and the minted banner
/// make obvious: Revoke takes its own row's only verb away (the row stays and
/// the button goes), Generate is replaced by the reason it is unavailable, and
/// either Retry replaces the surface it stands in with the load it started.
/// Each needs a handle on the subtree that disappears, because that is the only
/// thing that can answer whether the keyboard was in it.
pub const BINDING_SLOT: &str = "binding";
pub const MINTED_SLOT: &str = "minted";
pub const CREATE_SLOT: &str = "create";
pub const RETRY_SLOT: &str = "retry";

/// One key row's slot — its Revoke verb is the only tab stop in it.
fn key_slot(id: &str) -> String {
    format!("key:{id}")
}

pub struct ProxySettingsView {
    proxy: Entity<ProxyStore>,
    backends: Entity<BackendsStore>,
    /// The pane's own handle — the destination a verb that unmounts itself
    /// hands the keyboard back to (`RecordView::close_detail`'s rule). It
    /// carries a `Region` role, because the adapter reports focus only on a
    /// node the a11y tree actually has.
    focus_handle: FocusHandle,
    /// The binding editor, revealed by "Change…". `None` while the row is
    /// showing the value rather than editing it.
    binding_edit: Option<BindingEdit>,
    /// The name a new key will carry.
    key_label: Entity<InputState>,
    /// One handle per subtree of this pane that a verb can unmount from under
    /// the keyboard — see [`ProxySettingsView::slot`]. Interior-mutable because
    /// the row builders are `&self` (the Local pane's `row_focus` shape).
    slot_focus: RefCell<HashMap<String, FocusHandle>>,
    _subscriptions: Vec<gpui::Subscription>,
}

/// The address row's in-place editor.
struct BindingEdit {
    address: Entity<InputState>,
    port: Entity<InputState>,
}

impl ProxySettingsView {
    pub fn new(stores: Stores, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let proxy = stores.proxy.clone();
        let backends = stores.backends.clone();
        // Both stores are rendered, so both are observed: the settings and the
        // key listing arrive asynchronously and alone, and so does the backend
        // registry the checkbox rows are drawn from.
        let _subscriptions = vec![
            cx.observe(&proxy, |_, _, cx| cx.notify()),
            cx.observe(&backends, |_, _, cx| cx.notify()),
        ];
        let key_label = cx.new(|cx| {
            InputState::new(window, cx).placeholder(msg::proxy_key_label_placeholder(cx))
        });
        Self {
            proxy,
            backends,
            focus_handle: cx.focus_handle(),
            binding_edit: None,
            key_label,
            slot_focus: RefCell::new(HashMap::new()),
            _subscriptions,
        }
    }

    /// Test seam: the handle a disappearing subtree tracks, once it has
    /// painted. `None` before that — the slots are minted lazily by `render`.
    #[doc(hidden)]
    pub fn slot_focus_for_test(&self, key: &str) -> Option<FocusHandle> {
        self.slot_focus.borrow().get(key).cloned()
    }

    /// Test seam: the slot key a key row's Revoke verb lives in.
    #[doc(hidden)]
    pub fn key_slot_for_test(id: &str) -> String {
        key_slot(id)
    }

    /// Whether the address row is being edited (test seam).
    pub fn is_editing_binding(&self) -> bool {
        self.binding_edit.is_some()
    }

    pub fn begin_binding_edit(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(settings) = self.proxy.read(cx).settings().value().cloned() else {
            return;
        };
        let address = cx.new(|cx| {
            let mut state = InputState::new(window, cx);
            state.set_value(settings.bind_address.clone(), window, cx);
            state
        });
        let port = cx.new(|cx| {
            let mut state = InputState::new(window, cx);
            state.set_value(settings.bind_port.to_string(), window, cx);
            state
        });
        self.binding_edit = Some(BindingEdit { address, port });
        cx.notify();
    }

    /// Abandon the edit. The keyboard goes back to the pane, because the fields
    /// this unmounts are where it was.
    pub fn cancel_binding_edit(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.hand_back_focus_from(BINDING_SLOT, window, cx);
        self.binding_edit = None;
        cx.notify();
    }

    /// Commit the edit.
    ///
    /// A port that does not parse is refused **here**, before the write, so a
    /// typo leaves the stored binding untouched and the field still holding
    /// what was typed — there is nothing to reconcile because nothing moved.
    pub fn commit_binding_edit(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(edit) = self.binding_edit.as_ref() else {
            return;
        };
        let address = edit.address.read(cx).value().to_string();
        let Ok(port) = edit.port.read(cx).value().trim().parse::<u16>() else {
            return;
        };
        self.proxy
            .update(cx, |s, cx| s.set_binding(address, port, cx));
        self.hand_back_focus_from(BINDING_SLOT, window, cx);
        self.binding_edit = None;
        cx.notify();
    }

    pub fn set_enabled(&mut self, enabled: bool, cx: &mut Context<Self>) {
        self.proxy.update(cx, |s, cx| s.set_enabled(enabled, cx));
        cx.notify();
    }

    pub fn set_exposure(&mut self, exposure: LocalExposure, cx: &mut Context<Self>) {
        self.proxy
            .update(cx, |s, cx| s.set_local_exposure(exposure, cx));
        cx.notify();
    }

    pub fn set_backend_exposed(&mut self, id: String, exposed: bool, cx: &mut Context<Self>) {
        self.proxy
            .update(cx, |s, cx| s.set_backend_exposed(id, exposed, cx));
        cx.notify();
    }

    /// Generate a key named by the field. A blank name is refused core-side,
    /// so the press simply does nothing rather than inventing a name.
    pub fn create_key(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let label = self.key_label.read(cx).value().trim().to_string();
        if label.is_empty() {
            return;
        }
        // The verb is replaced by "Generating…" the moment this lands, so the
        // press takes its own control away.
        self.hand_back_focus_from(CREATE_SLOT, window, cx);
        self.proxy.update(cx, |s, cx| s.create_key(label, cx));
        self.key_label
            .update(cx, |s, cx| s.set_value(String::new(), window, cx));
        cx.notify();
    }

    /// Revoke a key. **Its own verb does not survive this**: the row stays (its
    /// label is what tells a reader which tool lost access) and loses the only
    /// tab stop in it, so the press unmounts the control that made it.
    pub fn revoke_key(&mut self, id: String, window: &mut Window, cx: &mut Context<Self>) {
        self.hand_back_focus_from(&key_slot(&id), window, cx);
        self.proxy.update(cx, |s, cx| s.revoke_key(id, cx));
        cx.notify();
    }

    /// Acknowledge the generated key. **The value goes with the press** — only
    /// its digest was ever stored, so nothing can bring it back.
    pub fn dismiss_minted(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.hand_back_focus_from(MINTED_SLOT, window, cx);
        self.proxy.update(cx, |s, cx| s.dismiss_minted(cx));
        cx.notify();
    }

    /// The focus handle for one subtree that can disappear under the keyboard,
    /// minted on first use.
    ///
    /// Every verb in this pane whose press removes the verb needs one: the
    /// question a handback has to ask is whether **the subtree that is about to
    /// stop being painted** was holding the keyboard, and nothing smaller can
    /// answer it (a probed button rides gpui's implicit handle, which this code
    /// never receives). Keyed by a string so a per-key row gets its own.
    fn slot(&self, key: &str, cx: &App) -> FocusHandle {
        self.slot_focus
            .borrow_mut()
            .entry(key.to_string())
            .or_insert_with(|| cx.focus_handle())
            .clone()
    }

    /// Put the keyboard back on the pane when the subtree named by `key` — the
    /// one this verb is about to unmount — is where it was.
    ///
    /// **The question is about the disappearing subtree, not the pane.** Asking
    /// whether the *pane* contains focus is true for every control in it,
    /// including the one being activated, so the helper declined exactly when
    /// it was needed: Save, Cancel and Done each ran from the keyboard, found
    /// "the pane has it", moved nothing, and then removed the editor or the
    /// banner around the focused control — leaving the window on a handle
    /// nobody paints, with Tab restarting from the window root. Asked of the
    /// subtree, a reader working *elsewhere in the pane* still keeps their
    /// caret, which is the property the original predicate was reaching for.
    fn hand_back_focus_from(&self, key: &str, window: &mut Window, cx: &mut Context<Self>) {
        if !self.slot(key, cx).contains_focused(window, cx) {
            return;
        }
        self.focus_handle.focus(window, cx);
    }

    /// Re-read the proxy's state. **Every door into this is a verb that
    /// replaces the surface it stands in** — a failure panel becomes the
    /// loading line, a stale strip stands down over its rows — so the press
    /// hands the keyboard back from whichever surface carried it.
    fn refresh(&mut self, slot: &str, window: &mut Window, cx: &mut Context<Self>) {
        self.hand_back_focus_from(slot, window, cx);
        self.proxy.update(cx, |s, cx| s.refresh(cx));
        cx.notify();
    }
}

impl Focusable for ProxySettingsView {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for ProxySettingsView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let store = self.proxy.read(cx);
        let settings = store.settings().clone();
        let op_error = store.op_error().map(str::to_string);
        let listen_error = store.listen_error();
        let address = store.address().map(|a| a.to_string());
        let minted = store
            .minted()
            .map(|m| (m.key.clone(), m.info.label.clone()));
        let keys = store.keys().clone();
        let can_create_key = store.can_create_key();
        let create_key_pending = store.create_key_pending();
        let backends: Vec<BackendInfo> = self.backends.read(cx).list().to_vec();

        let mut col = v_flex()
            .id("proxy-pane")
            .track_focus(&self.focus_handle)
            // A handback target must carry a role, or the adapter has no node
            // to report focus on (`settings/backends/pane`'s rule).
            .probe("settings/proxy/pane", gpui::Role::Region, "Proxy")
            .px_6()
            .py_5()
            .gap_4()
            .w_full();

        // A failed *initial* read leaves nothing here to act on, so the way
        // back is a Retry rather than a plausible-looking empty pane.
        if let crate::loadable::Loadable::Failed { error, prior: None } = &settings {
            return col.child(
                // The panel is what its own Retry replaces, so the press asks
                // *this* subtree whether it was holding the keyboard. One slot
                // serves this panel and the keys' one below: the settings
                // failure returns early, so no frame can paint both.
                div()
                    .id("proxy-retry-slot")
                    .track_focus(&self.slot(RETRY_SLOT, cx))
                    .child(load_error_panel(
                        "settings/proxy/retry",
                        msg::proxy_failed(cx),
                        &error.to_string(),
                        msg::proxy_retry(cx),
                        cx,
                        cx.listener(|this, _, window, cx| this.refresh(RETRY_SLOT, window, cx)),
                    )),
            );
        }
        // A refresh that failed over a snapshot we still hold keeps the rows and
        // says so quietly — "Failed is not empty", and its mirror: a read still
        // in flight is not an empty configuration either, so it says *that*
        // rather than painting a pane with nothing in it.
        let stale_settings = matches!(
            &settings,
            crate::loadable::Loadable::Failed { prior: Some(_), .. }
        );
        let Some(settings) = settings.value().cloned() else {
            return col.child(loading_line(
                "settings/proxy/loading",
                msg::proxy_loading(cx),
                cx,
            ));
        };
        if stale_settings {
            col = col.child(self.stale_strip("settings/proxy/stale", cx));
        }

        col = col.child(
            div()
                .max_w(px(520.))
                .text_xs()
                .text_color(theme.muted_foreground)
                .child(msg::proxy_lead(cx)),
        );

        // --- 1. The switch, and where a tool should point -------------------
        let status = match &address {
            Some(address) => msg::proxy_listening(cx, address.clone()),
            None => msg::proxy_stopped(cx),
        };
        col = col.child(field_row(
            msg::proxy_serve(cx),
            cx,
            v_flex()
                .gap_1()
                .child(
                    h_flex()
                        .gap_3()
                        .items_center()
                        .child(self.serve_switch(settings.enabled, cx))
                        .child(
                            div()
                                .id("proxy-status")
                                // The endpoint is a settled readout, so it
                                // rides its own `Label` node with the sentence
                                // as its value — the notices' shape.
                                .probe_value(
                                    "settings/proxy/status",
                                    gpui::Role::Label,
                                    status.clone(),
                                    status.clone(),
                                )
                                .text_xs()
                                .text_color(theme.muted_foreground)
                                .child(status),
                        ),
                )
                .when_some(listen_error, |el, message| {
                    el.child(
                        div()
                            .id("proxy-listen-error")
                            .probe(
                                "settings/proxy/listen-error",
                                gpui::Role::Alert,
                                msg::proxy_listen_failed(cx, message.clone()),
                            )
                            .child(notice_band(msg::proxy_listen_failed(cx, message), cx)),
                    )
                }),
        ));

        // --- 2. The binding ------------------------------------------------
        col = col.child(field_row(
            msg::proxy_address(cx),
            cx,
            self.binding_row(&settings, cx),
        ));
        if !settings.is_loopback() {
            col = col.child(
                div()
                    .id("proxy-exposed-warning")
                    .probe(
                        "settings/proxy/exposed-warning",
                        gpui::Role::Alert,
                        msg::proxy_exposed_warning(cx),
                    )
                    .child(notice_band(msg::proxy_exposed_warning(cx), cx)),
            );
        }

        // --- 3. Which backends -----------------------------------------------
        col = col.child(section_header(msg::proxy_backends(cx), cx));
        col = col.child(
            div()
                .max_w(px(520.))
                .text_xs()
                .text_color(theme.muted_foreground.opacity(0.8))
                .child(msg::proxy_backends_note(cx)),
        );
        if backends.is_empty() {
            col = col.child(
                div()
                    .id("proxy-backends-empty")
                    .probe(
                        "settings/proxy/backends/empty",
                        gpui::Role::Label,
                        msg::proxy_backends_empty(cx),
                    )
                    .text_xs()
                    .text_color(theme.muted_foreground)
                    .child(msg::proxy_backends_empty(cx)),
            );
        }
        for backend in &backends {
            col = col.child(self.backend_row(backend, &settings, cx));
        }

        // --- 4. What an on-device backend offers -----------------------------
        col = col.child(section_header(msg::proxy_exposure(cx), cx));
        let mut chips = h_flex().gap_2();
        for (value, id, probe_name, label) in [
            (
                LocalExposure::Loaded,
                "proxy-exposure-loaded",
                "settings/proxy/exposure/loaded",
                msg::proxy_exposure_loaded(cx),
            ),
            (
                LocalExposure::Downloaded,
                "proxy-exposure-downloaded",
                "settings/proxy/exposure/downloaded",
                msg::proxy_exposure_downloaded(cx),
            ),
        ] {
            let active = settings.local_exposure == value;
            chips = chips.child(crate::participants::mode_chip(
                SharedString::from(id),
                SharedString::from(probe_name),
                label,
                active,
                cx,
                cx.listener(move |this, _, _, cx| this.set_exposure(value, cx)),
            ));
        }
        col = col.child(
            v_flex().gap_1().child(chips).child(
                div()
                    .max_w(px(520.))
                    .text_xs()
                    .text_color(theme.muted_foreground.opacity(0.8))
                    .child(msg::proxy_exposure_note(cx)),
            ),
        );

        // --- 5. Keys ---------------------------------------------------------
        col = col.child(section_header(msg::proxy_keys(cx), cx));
        col = col.child(
            div()
                .max_w(px(520.))
                .text_xs()
                .text_color(theme.muted_foreground.opacity(0.8))
                .child(msg::proxy_keys_note(cx)),
        );
        if let Some((key, label)) = minted {
            col = col.child(self.minted_banner(&key, &label, cx));
        }
        // The listing's four states read apart. A key cell that has not answered
        // is not "no keys": that sentence, over a proxy that in fact has live
        // keys, invites the reader to generate another — and a failed read left
        // nothing on the page that could ever cause a second one.
        match &keys {
            crate::loadable::Loadable::Failed { error, prior: None } => {
                col = col.child(
                    div()
                        .id("proxy-keys-retry-slot")
                        .track_focus(&self.slot(RETRY_SLOT, cx))
                        .child(load_error_panel(
                            "settings/proxy/keys/retry",
                            msg::proxy_keys_failed(cx),
                            &error.to_string(),
                            msg::proxy_retry(cx),
                            cx,
                            cx.listener(|this, _, window, cx| this.refresh(RETRY_SLOT, window, cx)),
                        )),
                );
            }
            crate::loadable::Loadable::NotLoaded | crate::loadable::Loadable::Loading => {
                col = col.child(loading_line(
                    "settings/proxy/keys/loading",
                    msg::proxy_loading(cx),
                    cx,
                ));
            }
            _ => {
                if matches!(
                    &keys,
                    crate::loadable::Loadable::Failed { prior: Some(_), .. }
                ) {
                    col = col.child(self.stale_strip("settings/proxy/keys/stale", cx));
                }
                let rows: &[ProxyKeyInfo] = keys.value().map(|v| v.as_slice()).unwrap_or(&[]);
                if rows.is_empty() {
                    col = col.child(
                        div()
                            .id("proxy-keys-empty")
                            .probe(
                                "settings/proxy/keys/empty",
                                gpui::Role::Label,
                                msg::proxy_keys_empty(cx),
                            )
                            .text_xs()
                            .text_color(theme.muted_foreground)
                            .child(msg::proxy_keys_empty(cx)),
                    );
                }
                for (index, key) in rows.iter().enumerate() {
                    col = col.child(self.key_row(index, key, cx));
                }
            }
        }
        col = col.child(
            h_flex()
                .id("proxy-key-create-row")
                // Generate is replaced by the reason it is unavailable, so the
                // press unmounts its own control; the row it stands in stays.
                .track_focus(&self.slot(CREATE_SLOT, cx))
                .w_full()
                .gap_2()
                .child(
                    div()
                        .id("proxy-key-label-wrap")
                        .probe_bounds(
                            "settings/proxy/keys/label",
                            gpui::Role::TextInput,
                            msg::proxy_key_label_placeholder(cx),
                        )
                        .flex_1()
                        .min_w_0()
                        .child(
                            Input::new(&self.key_label)
                                .aria_label(msg::proxy_key_label_placeholder(cx)),
                        ),
                )
                // **A control over a settled decision is not a control.** While
                // a generation is in flight, or a minted key is still waiting to
                // be read, the verb is replaced by the reason — no id, no probe,
                // so no tab stop and nothing to activate. One predicate decides
                // the press and the painting alike, so an offered verb and an
                // accepted press cannot disagree.
                .child(if can_create_key {
                    ghost_button(
                        SharedString::from("proxy-key-create"),
                        SharedString::from("settings/proxy/keys/create"),
                        msg::proxy_key_create(cx),
                        true,
                        cx,
                        cx.listener(|this, _, window, cx| this.create_key(window, cx)),
                    )
                    .into_any_element()
                } else {
                    let reason = if create_key_pending {
                        msg::proxy_key_creating(cx)
                    } else {
                        msg::proxy_key_show_first(cx)
                    };
                    div()
                        .flex_none()
                        .text_xs()
                        .text_color(theme.muted_foreground)
                        .child(reason)
                        .into_any_element()
                }),
        );

        if let Some(message) = op_error {
            col = col.child(
                div()
                    .id("proxy-error")
                    .probe("settings/proxy/error", gpui::Role::Alert, message.clone())
                    .child(notice_band(SharedString::from(message), cx)),
            );
        }
        col
    }
}

impl ProxySettingsView {
    /// The enable switch. The probed wrapper **is** the control (role, label,
    /// keyboard activation) because `Switch` tracks no focus handle at this
    /// gpui-component rev; the widget handles the pointer press itself with
    /// `stop_propagation`, so the two never double-fire.
    fn serve_switch(&self, enabled: bool, cx: &Context<Self>) -> impl IntoElement + use<> {
        let next = !enabled;
        div()
            .id("proxy-serve")
            .probe(
                "settings/proxy/serve",
                gpui::Role::CheckBox,
                msg::proxy_serve_name(cx),
            )
            // `aria_toggled`, not `aria_selected`: `accesskit_macos` reads a
            // checkbox's value from `toggled()`.
            .aria_toggled(enabled.into())
            .on_click(cx.listener(move |this, _, _, cx| this.set_enabled(next, cx)))
            .child(
                Switch::new("proxy-serve-switch")
                    .small()
                    .checked(enabled)
                    .on_click(cx.listener(move |this, checked: &bool, _, cx| {
                        this.set_enabled(*checked, cx);
                    })),
            )
    }

    /// The address row: the value, or the editor once "Change…" reveals it.
    fn binding_row(&self, settings: &ProxySettings, cx: &Context<Self>) -> gpui::AnyElement {
        let theme = cx.theme();
        match self.binding_edit.as_ref() {
            None => h_flex()
                .gap_2()
                .items_center()
                .child(
                    div()
                        .text_sm()
                        .text_color(theme.foreground)
                        .child(SharedString::from(format!(
                            "{}:{}",
                            settings.bind_address, settings.bind_port
                        ))),
                )
                .child(ghost_button(
                    SharedString::from("proxy-binding-change"),
                    SharedString::from("settings/proxy/binding/change"),
                    msg::proxy_binding_change(cx),
                    false,
                    cx,
                    cx.listener(|this, _, window, cx| this.begin_binding_edit(window, cx)),
                ))
                .into_any_element(),
            // The editor is what Save and Cancel replace, so it is the subtree
            // their handback asks about — the fields inside it are where the
            // keyboard is, and the pane around it survives either press.
            Some(edit) => h_flex()
                .id("proxy-binding-editor")
                .track_focus(&self.slot(BINDING_SLOT, cx))
                .gap_2()
                .items_center()
                .child(
                    div()
                        .id("proxy-address-wrap")
                        .probe_bounds(
                            "settings/proxy/binding/address",
                            gpui::Role::TextInput,
                            msg::proxy_address_name(cx),
                        )
                        .flex_1()
                        .min_w_0()
                        .child(Input::new(&edit.address).aria_label(msg::proxy_address_name(cx))),
                )
                .child(
                    div()
                        .id("proxy-port-wrap")
                        .probe_bounds(
                            "settings/proxy/binding/port",
                            gpui::Role::TextInput,
                            msg::proxy_port_name(cx),
                        )
                        .w(px(80.))
                        .flex_none()
                        .child(Input::new(&edit.port).aria_label(msg::proxy_port_name(cx))),
                )
                .child(ghost_button(
                    SharedString::from("proxy-binding-save"),
                    SharedString::from("settings/proxy/binding/save"),
                    msg::proxy_binding_save(cx),
                    true,
                    cx,
                    cx.listener(|this, _, window, cx| this.commit_binding_edit(window, cx)),
                ))
                .child(ghost_button(
                    SharedString::from("proxy-binding-cancel"),
                    SharedString::from("settings/proxy/binding/cancel"),
                    msg::proxy_binding_cancel(cx),
                    false,
                    cx,
                    cx.listener(|this, _, window, cx| this.cancel_binding_edit(window, cx)),
                ))
                .into_any_element(),
        }
    }

    /// One backend's exposure checkbox. **The exposure set is read from the
    /// stored rows** (`exposed_ids`), not from the live-and-enabled set the
    /// listing serves from, so a backend the reader disabled still shows the
    /// choice they made rather than silently losing it.
    fn backend_row(
        &self,
        backend: &BackendInfo,
        settings: &ProxySettings,
        cx: &Context<Self>,
    ) -> impl IntoElement + use<> {
        let theme = cx.theme();
        let exposed = settings.exposed_ids.contains(&backend.id);
        let id = backend.id.clone();
        let name = SharedString::from(backend.display_name.clone());
        h_flex()
            .gap_2()
            .items_center()
            .child(
                div()
                    .id(SharedString::from(format!("proxy-backend-{}", backend.id)))
                    .probe(
                        format!("settings/proxy/backends/{}", backend.id),
                        gpui::Role::CheckBox,
                        msg::proxy_backend_name(cx, name.to_string()),
                    )
                    .aria_toggled(exposed.into())
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.set_backend_exposed(id.clone(), !exposed, cx)
                    }))
                    .child(
                        Switch::new(SharedString::from(format!(
                            "proxy-backend-switch-{}",
                            backend.id
                        )))
                        .small()
                        .checked(exposed),
                    ),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(theme.foreground)
                    .child(name.clone()),
            )
    }

    /// One key's row: what it is called, enough of the key to tell rows apart,
    /// and what has become of it.
    fn key_row(
        &self,
        index: usize,
        key: &ProxyKeyInfo,
        cx: &Context<Self>,
    ) -> impl IntoElement + use<> {
        let theme = cx.theme();
        let label = SharedString::from(key.label.clone());
        let state = if key.revoked_at.is_some() {
            msg::proxy_key_revoked(cx)
        } else if key.last_used_at.is_some() {
            msg::proxy_key_used(cx)
        } else {
            msg::proxy_key_unused(cx)
        };
        let summary = SharedString::from(format!("{} · {}… · {}", key.label, key.prefix, state));
        let id = key.id.clone();
        let live = key.is_live();
        h_flex()
            .id(SharedString::from(format!("proxy-key-row-{index}")))
            .track_focus(&self.slot(&key_slot(&key.id), cx))
            .w_full()
            .gap_2()
            .items_center()
            .child(
                div()
                    .id(SharedString::from(format!("proxy-key-{index}")))
                    .probe_value(
                        format!("settings/proxy/keys/{index}"),
                        gpui::Role::ListItem,
                        summary.clone(),
                        summary.clone(),
                    )
                    .flex_1()
                    .min_w_0()
                    .text_sm()
                    .text_color(if live {
                        theme.foreground
                    } else {
                        theme.muted_foreground
                    })
                    .child(summary),
            )
            .when(live, |el| {
                el.child(ghost_button_labeled(
                    SharedString::from(format!("proxy-key-revoke-{index}")),
                    SharedString::from(format!("settings/proxy/keys/{index}/revoke")),
                    msg::proxy_key_revoke(cx),
                    msg::proxy_key_revoke_name(cx, label.to_string()),
                    false,
                    cx,
                    cx.listener(move |this, _, window, cx| this.revoke_key(id.clone(), window, cx)),
                ))
            })
    }

    /// The quiet strip over a cell whose *refresh* failed.
    ///
    /// The Library's line, not its panel: the values are still on screen and
    /// still worth reading, so what is owed is the fact that they are as of the
    /// last successful read, plus a way to ask again — nothing in this pane
    /// re-reads on its own between bus events.
    fn stale_strip(
        &self,
        probe_name: &'static str,
        cx: &Context<Self>,
    ) -> impl IntoElement + use<> {
        let theme = cx.theme();
        // The strip stands down the moment its Retry restarts the read, so the
        // press unmounts the surface it was made from. Keyed by the probe name,
        // which is already unique per painted element — the settings strip and
        // the keys strip can stand at once.
        h_flex()
            .id(probe_name)
            .track_focus(&self.slot(probe_name, cx))
            .gap_2()
            .items_center()
            .child(
                div()
                    .id(SharedString::from(format!("{probe_name}/line")))
                    .probe(probe_name, gpui::Role::Label, msg::proxy_stale(cx))
                    .text_xs()
                    .text_color(theme.muted_foreground)
                    .child(msg::proxy_stale(cx)),
            )
            .child(ghost_button(
                SharedString::from(format!("{probe_name}/retry")),
                SharedString::from(format!("{probe_name}-retry")),
                msg::proxy_retry(cx),
                false,
                cx,
                cx.listener(move |this, _, window, cx| this.refresh(probe_name, window, cx)),
            ))
    }

    /// The one moment a key exists in full.
    fn minted_banner(
        &self,
        key: &str,
        label: &str,
        cx: &Context<Self>,
    ) -> impl IntoElement + use<> {
        let theme = cx.theme();
        let key = SharedString::from(key.to_string());
        let copy = key.clone();
        v_flex()
            .id("proxy-key-minted")
            // Done takes this banner away, and its Copy verb is a tab stop in
            // it too — so the banner is the subtree the handback asks about.
            .track_focus(&self.slot(MINTED_SLOT, cx))
            // The key is what this surface is for, so it is the node's value —
            // a reader who cannot see it must still be able to hear it.
            .probe_value(
                "settings/proxy/keys/minted",
                gpui::Role::Alert,
                SharedString::from(label.to_string()),
                key.clone(),
            )
            .w_full()
            .gap_2()
            .px_3()
            .py_2()
            .rounded_md()
            .bg(theme.accent.opacity(0.10))
            .child(Label::new(msg::proxy_key_minted(cx)).text_xs())
            .child(
                div()
                    .font_family("Menlo")
                    .text_sm()
                    .text_color(theme.foreground)
                    .child(key),
            )
            .child(
                h_flex()
                    .gap_2()
                    .child(ghost_button(
                        SharedString::from("proxy-key-minted-copy"),
                        SharedString::from("settings/proxy/keys/minted/copy"),
                        msg::proxy_key_copy(cx),
                        true,
                        cx,
                        move |_, _, cx: &mut App| {
                            cx.write_to_clipboard(ClipboardItem::new_string(copy.to_string()));
                        },
                    ))
                    .child(ghost_button(
                        SharedString::from("proxy-key-minted-done"),
                        SharedString::from("settings/proxy/keys/minted/done"),
                        msg::proxy_key_done(cx),
                        false,
                        cx,
                        cx.listener(|this, _, window, cx| this.dismiss_minted(window, cx)),
                    )),
            )
    }
}

/// A read that has not answered, said plainly.
///
/// The Participants section's idiom: a cell in flight renders a quiet line, not
/// the empty state it will not necessarily land on. It carries its own `Label`
/// node, because the sentence *is* what the surface says here.
fn loading_line(probe_name: &'static str, message: SharedString, cx: &App) -> impl IntoElement {
    let theme = cx.theme();
    div()
        .id(probe_name)
        .probe(probe_name, gpui::Role::Label, message.clone())
        .text_xs()
        .text_color(theme.muted_foreground)
        .child(message)
}

/// A danger band whose message **wraps**.
///
/// `participants::error_banner` lays its label out in an `h_flex` with no
/// width discipline, which is right for the one-line refusals every other pane
/// puts in it and wrong for a sentence: this pane's two bands are a whole
/// explanation each, and an unwrapped one runs off the edge of the panel.
/// Written locally rather than by widening the shared helper, because the
/// helper's callers are sized around its current shape and a sentence is this
/// surface's problem.
fn notice_band(message: SharedString, cx: &App) -> impl IntoElement {
    let theme = cx.theme();
    div()
        .w_full()
        .max_w(px(520.))
        .px_3()
        .py_2()
        .rounded_md()
        .bg(theme.danger.opacity(0.08))
        .text_color(theme.danger)
        .text_xs()
        .child(message)
}

fn section_header(label: SharedString, cx: &App) -> impl IntoElement {
    let theme = cx.theme();
    div()
        .text_color(theme.muted_foreground)
        .text_sm()
        .font_medium()
        .child(label)
}

fn field_row<C: IntoElement>(label: SharedString, cx: &App, child: C) -> impl IntoElement {
    let theme = cx.theme();
    h_flex()
        .w_full()
        .gap_4()
        .py_1()
        .items_start()
        .child(
            div()
                .w(px(144.))
                .flex_none()
                .text_sm()
                .text_color(theme.muted_foreground)
                .child(label),
        )
        .child(div().flex_1().min_w_0().child(child))
}
