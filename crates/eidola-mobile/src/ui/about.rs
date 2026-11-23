//! About, and the licences.
//!
//! **About** says what is running, in the words of whoever knows it: the
//! app's own version (compiled in), and the shell's account of itself and
//! of the bundle it served — its version, digest and standing — stated as
//! what the shell reports. The shell shows the same digest in a sheet of its
//! own, drawn natively and out of this bundle's reach, and the screen says
//! how to open it: that sheet, not this screen, is the check.
//!
//! **The licences** are the bundle's own file, `licences.json`, generated
//! when the bundle is built (`crates/mobile-licences`) from the wasm build's
//! dependency graph and the assets it embeds, with every licence text in it:
//! read from this document's own origin, which the shell serves from the
//! bundle, so nothing is fetched from anywhere else.

use gpui::{
    Context, EventEmitter, InteractiveElement as _, IntoElement, ListAlignment, ListState,
    MouseButton, ParentElement as _, Render, Role, ScrollHandle, SharedString,
    StatefulInteractiveElement as _, Styled as _, Task, Window, div, list, px,
};
use gpui_component::ActiveTheme as _;
use serde::Deserialize;

use super::GUTTER;
use super::i18n::msg;
use super::nav::{Dest, NavEvent};
use super::probe::Probe as _;
use super::stores::Stores;
use super::widgets::{
    Pill, error_note, grouped_digest, header, keep_keyboard, nav_row, note, pill, section,
    value_row,
};
use eidola_app_core::updater::mobile::EMBEDDED;

/// The source repository.
pub const REPO_URL: &str = "https://github.com/eidola-ai/eidola";

pub struct AboutView {
    stores: Stores,
    scroll: ScrollHandle,
    open_task: Option<Task<()>>,
    open_error: Option<String>,
}

impl EventEmitter<NavEvent> for AboutView {}

impl AboutView {
    pub fn new(stores: Stores, cx: &mut Context<Self>) -> Self {
        cx.observe(&stores.updates, |_, _, cx| cx.notify()).detach();
        stores.updates.update(cx, |u, cx| u.refresh_report(cx));
        Self {
            stores,
            scroll: ScrollHandle::new(),
            open_task: None,
            open_error: None,
        }
    }

    fn open_source(&mut self, cx: &mut Context<Self>) {
        let open = self.stores.host.open_external(REPO_URL.to_string());
        self.open_error = None;
        self.open_task = Some(cx.spawn(async move |this, cx| {
            let result = open.await;
            this.update(cx, |this, cx| {
                this.open_task = None;
                this.open_error = result.err();
                cx.notify();
            })
            .ok();
        }));
    }
}

impl Render for AboutView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let hello = self.stores.host.hello().clone();
        let report = self.stores.updates.read(cx).report().cloned();
        let back = Box::new(cx.listener(|_, _, _, cx| cx.emit(NavEvent::Back)));

        let mut bundle = div().flex().flex_col();
        match report {
            Some(Ok(report)) => {
                let version: SharedString = if report.serving.version == EMBEDDED {
                    msg::updates_embedded(cx)
                } else {
                    report.serving.version.clone().into()
                };
                bundle = bundle
                    .child(value_row(
                        "about/bundle-version",
                        msg::about_bundle_version(cx),
                        version,
                        false,
                        cx,
                    ))
                    .child(value_row(
                        "about/bundle-digest",
                        msg::about_bundle_digest(cx),
                        match &report.serving.digest {
                            Some(d) => grouped_digest(d),
                            None => msg::about_digest_computing(cx),
                        },
                        true,
                        cx,
                    ))
                    .child(note(
                        if report.on_trial {
                            msg::updates_on_trial(cx)
                        } else {
                            msg::updates_confirmed(cx)
                        },
                        cx,
                    ));
            }
            Some(Err(e)) => {
                bundle = bundle.child(error_note(
                    "about/bundle-error",
                    msg::updates_report_failed(cx, e),
                    cx,
                ));
            }
            None => {}
        }

        div()
            .size_full()
            .flex()
            .flex_col()
            .child(header("about", msg::about_title(cx), Some(back), None, cx))
            .child(
                div()
                    .id("about-body")
                    .flex_1()
                    .min_h_0()
                    .overflow_y_scroll()
                    .track_scroll(&self.scroll)
                    .pb(px(32.))
                    .child(
                        div()
                            .px(GUTTER)
                            .pt(px(18.))
                            .font_family(eidola_theme::FONT_FAMILY)
                            .text_size(px(28.))
                            .child(msg::about_name(cx)),
                    )
                    .child(note(msg::about_purpose_lead(cx), cx))
                    .child(note(msg::about_purpose_attestation(cx), cx))
                    .child(value_row(
                        "about/version",
                        msg::about_app_version(cx),
                        format!("v{}", env!("CARGO_PKG_VERSION")).into(),
                        false,
                        cx,
                    ))
                    .child(section(msg::about_section_shell(cx), cx))
                    .child(note(msg::about_shell_reports(cx), cx))
                    .child(value_row(
                        "about/shell",
                        msg::about_shell(cx),
                        msg::about_shell_value(
                            cx,
                            hello.platform.clone(),
                            hello.shell.clone(),
                            hello.contract as i64,
                        ),
                        false,
                        cx,
                    ))
                    .child(bundle)
                    .child(note(msg::about_sheet_pointer(cx), cx))
                    .child(section(msg::about_section_source(cx), cx))
                    .child(nav_row(
                        "about/licences",
                        msg::about_licences(cx),
                        Some(msg::about_licences_detail(cx)),
                        None,
                        Box::new(cx.listener(|_, _, _, cx| cx.emit(NavEvent::To(Dest::Licences)))),
                        cx,
                    ))
                    .child(note(msg::about_source_note(cx), cx))
                    .child(div().px(GUTTER).child(pill(
                        "about/source",
                        msg::about_source(cx),
                        Pill::Secondary,
                        self.open_task.is_none(),
                        Box::new(cx.listener(|this, _, _, cx| this.open_source(cx))),
                        cx,
                    )))
                    .children(self.open_error.clone().map(|e| {
                        error_note("about/source-error", msg::account_door_failed(cx, e), cx)
                    })),
            )
    }
}

/// `licences.json`, as `crates/mobile-licences` writes it.
#[derive(Debug, Deserialize)]
pub struct Licences {
    /// Every distinct licence text, once.
    pub texts: Vec<String>,
    pub components: Vec<Component>,
}

#[derive(Debug, Deserialize)]
pub struct Component {
    pub name: String,
    #[serde(default)]
    pub version: Option<String>,
    /// The licence as the component declares it (an SPDX expression).
    pub licence: String,
    /// What it is: a crate in the wasm build, or an asset it embeds.
    pub kind: String,
    /// Indices into [`Licences::texts`].
    pub texts: Vec<usize>,
}

/// The bundle path of the licences file.
pub const LICENCES_PATH: &str = "licences.json";

pub struct LicencesView {
    licences: Option<Result<Licences, String>>,
    _load: Task<()>,
    list: ListState,
    /// The component whose texts are shown.
    open: Option<usize>,
}

impl EventEmitter<NavEvent> for LicencesView {}

impl LicencesView {
    pub fn new(stores: Stores, cx: &mut Context<Self>) -> Self {
        let read = stores.host.bundle_text(LICENCES_PATH);
        let load = cx.spawn(async move |this, cx| {
            let result = read.await.and_then(|text| {
                serde_json::from_str::<Licences>(&text).map_err(|e| e.to_string())
            });
            this.update(cx, |this, cx| {
                if let Ok(licences) = &result {
                    this.list.reset(licences.components.len() + 1);
                }
                this.licences = Some(result);
                cx.notify();
            })
            .ok();
        });
        Self {
            licences: None,
            _load: load,
            list: ListState::new(0, ListAlignment::Top, px(800.)),
            open: None,
        }
    }

    fn toggle(&mut self, ix: usize, cx: &mut Context<Self>) {
        let previous = self.open.replace(ix);
        if previous == Some(ix) {
            self.open = None;
        }
        // Rows are offset by the introduction.
        for row in [previous, Some(ix)].into_iter().flatten() {
            self.list.splice(row + 1..row + 2, 1);
        }
        cx.notify();
    }

    fn render_row(&mut self, ix: usize, cx: &mut Context<Self>) -> gpui::AnyElement {
        if ix == 0 {
            return note(msg::licences_intro(cx), cx).into_any_element();
        }
        let ix = ix - 1;
        let Some(Ok(licences)) = &self.licences else {
            return div().into_any_element();
        };
        let Some(component) = licences.components.get(ix) else {
            return div().into_any_element();
        };
        let theme = cx.theme().clone();
        let open = self.open == Some(ix);
        let title = match &component.version {
            Some(v) => format!("{} {v}", component.name),
            None => component.name.clone(),
        };
        let kind = match component.kind.as_str() {
            "crate" => msg::licences_kind_crate(cx),
            "font" => msg::licences_kind_font(cx),
            "icons" => msg::licences_kind_icons(cx),
            "shell" => msg::licences_kind_shell(cx),
            _ => msg::licences_kind_other(cx),
        };
        let mut row = div()
            .flex()
            .flex_col()
            .border_b_1()
            .border_color(theme.border)
            .child(
                div()
                    .id(("licence-row", ix))
                    .probe(
                        format!("licences/row/{ix}"),
                        Role::Button,
                        format!("{title}, {}", component.licence),
                    )
                    .px(GUTTER)
                    .py(px(10.))
                    .min_h(px(52.))
                    .flex()
                    .flex_col()
                    .gap(px(2.))
                    .active(|s| s.bg(theme.secondary))
                    .on_mouse_down(MouseButton::Left, keep_keyboard)
                    .on_click(cx.listener(move |this, _, _, cx| this.toggle(ix, cx)))
                    .child(div().text_size(px(16.)).child(title))
                    .child(
                        div()
                            .text_size(px(13.))
                            .text_color(theme.muted_foreground)
                            .child(format!("{} · {kind}", component.licence)),
                    ),
            );
        if open {
            for &text in &component.texts {
                if let Some(text) = licences.texts.get(text) {
                    row = row.child(
                        div()
                            .px(GUTTER)
                            .pb(px(12.))
                            .text_size(px(12.))
                            .line_height(px(17.))
                            .text_color(theme.muted_foreground)
                            .child(SharedString::from(text.clone())),
                    );
                }
            }
        }
        row.into_any_element()
    }
}

impl Render for LicencesView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let back = Box::new(cx.listener(|_, _, _, cx| cx.emit(NavEvent::Back)));
        let body = match &self.licences {
            None => note(msg::licences_loading(cx), cx).into_any_element(),
            Some(Err(e)) => error_note("licences/error", msg::licences_failed(cx, e.clone()), cx)
                .into_any_element(),
            Some(Ok(_)) => list(
                self.list.clone(),
                cx.processor(|this, ix, _, cx| this.render_row(ix, cx)),
            )
            .flex_1()
            .size_full()
            .into_any_element(),
        };
        div()
            .size_full()
            .flex()
            .flex_col()
            .child(header(
                "licences",
                msg::licences_title(cx),
                Some(back),
                None,
                cx,
            ))
            .child(div().flex_1().min_h_0().flex().flex_col().child(body))
    }
}
