//! Updates: the desktop's verified update flow (`crates/eidola-gui/src/updates.rs`)
//! on the phone, with the step the desktop leaves to its installer made here
//! and only by the reader: **Install update** approves the verified release,
//! which is downloaded, checked byte for byte, staged in the shell and
//! activated; the app then restarts into it, and the new bundle confirms
//! itself once it has drawn.
//!
//! Honest states, as on the desktop: "Checking…" only while a check runs; a
//! release that does not verify is a security warning and is never offered;
//! one whose claims changed shows the side-by-side and is offered only after
//! the reader's explicit "Treat as update". Below the check, what the shell
//! reports about the running bundle: its version and digest, whether it is on
//! trial, the last update that failed its trial (the app is then back on the
//! version it fell back to), and the way back to the bundle built into the
//! app.

use eidola_app_core::updater::mobile::EMBEDDED;
use gpui::{
    Context, EventEmitter, InteractiveElement as _, IntoElement, ParentElement as _, Render, Role,
    ScrollHandle, SharedString, StatefulInteractiveElement as _, Styled as _, Window, div, px,
};
use gpui_component::ActiveTheme as _;

use super::GUTTER;
use super::i18n::msg;
use super::nav::NavEvent;
use super::probe::Probe as _;
use super::stores::Stores;
use super::update_store::{Offer, UpdateDisplay};
use super::widgets::{Pill, error_note, grouped_digest, header, note, pill, section, value_row};

pub struct UpdatesView {
    stores: Stores,
    scroll: ScrollHandle,
}

impl EventEmitter<NavEvent> for UpdatesView {}

impl UpdatesView {
    pub fn new(stores: Stores, cx: &mut Context<Self>) -> Self {
        cx.observe(&stores.updates, |_, _, cx| cx.notify()).detach();
        // Opening the screen is the check, as opening the desktop's window is.
        stores.updates.update(cx, |u, cx| {
            u.refresh_report(cx);
            u.check_now(cx);
        });
        Self {
            stores,
            scroll: ScrollHandle::new(),
        }
    }
}

/// A line naming who attested a release and where it was logged.
fn attested_by(offer: &Offer, cx: &gpui::App) -> SharedString {
    let names: Vec<&str> = offer
        .summary
        .attestations
        .iter()
        .map(|a| a.attestant_name.as_str())
        .collect();
    if names.is_empty() {
        return msg::updates_attested_by_none(cx);
    }
    let logs: Vec<String> = offer
        .summary
        .attestations
        .iter()
        .map(|a| format!("#{}", a.rekor_log_index))
        .collect();
    msg::updates_attested_by(cx, names.join(", "), logs.join(", "))
}

fn status(name: &'static str, text: SharedString, cx: &gpui::App) -> impl IntoElement {
    div()
        .id(name)
        .probe(name, Role::Label, text.clone())
        .px(GUTTER)
        .pt(px(14.))
        .pb(px(4.))
        .font_family(eidola_theme::FONT_FAMILY)
        .text_size(px(20.))
        .line_height(px(26.))
        .text_color(cx.theme().foreground)
        .child(text)
}

impl UpdatesView {
    fn render_offer(
        &self,
        offer: &Offer,
        installing: bool,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let accepted = self.stores.updates.read(cx).claims_accepted_for(offer);
        let version = offer.summary.version.clone();
        let mut col = div().flex().flex_col();
        if offer.claims_changed() && !accepted {
            let matching = offer
                .claims
                .expected
                .len()
                .saturating_sub(offer.claims.deltas.len());
            col = col
                .child(status(
                    "updates/claims-changed",
                    msg::updates_claims_changed(cx, version),
                    cx,
                ))
                .child(note(msg::updates_claims_changed_note(cx), cx))
                .child(note(
                    msg::updates_claims_matching(
                        cx,
                        matching as i64,
                        offer.claims.expected.len() as i64,
                    ),
                    cx,
                ));
            let theme = cx.theme().clone();
            for (ix, delta) in offer.claims.deltas.iter().enumerate() {
                let expected = delta
                    .expected
                    .clone()
                    .unwrap_or_else(|| msg::updates_claim_absent(cx).to_string());
                let attested = delta
                    .attested
                    .clone()
                    .unwrap_or_else(|| msg::updates_claim_absent(cx).to_string());
                col = col.child(
                    div()
                        .id(("updates-delta", ix))
                        .probe(
                            format!("updates/delta/{ix}"),
                            Role::Label,
                            format!("{}: {expected} → {attested}", delta.key),
                        )
                        .mx(GUTTER)
                        .my(px(4.))
                        .p(px(10.))
                        .rounded_lg()
                        .border_1()
                        .border_color(theme.border)
                        .flex()
                        .flex_col()
                        .gap_1()
                        .child(div().text_size(px(14.)).child(delta.key.clone()))
                        .child(
                            div()
                                .text_size(px(13.))
                                .text_color(theme.muted_foreground)
                                .child(msg::updates_claim_expected(cx, expected)),
                        )
                        .child(
                            div()
                                .text_size(px(13.))
                                .text_color(theme.warning)
                                .child(msg::updates_claim_attested(cx, attested)),
                        ),
                );
            }
            return col
                .child(div().px(GUTTER).pt(px(10.)).child(pill(
                    "updates/accept-claims",
                    msg::updates_accept_claims(cx),
                    Pill::Secondary,
                    true,
                    Box::new(cx.listener(|this, _, _, cx| {
                        this.stores.updates.update(cx, |u, cx| u.accept_claims(cx))
                    })),
                    cx,
                )))
                .into_any_element();
        }
        col = col
            .child(status(
                "updates/available",
                msg::updates_available(cx, version),
                cx,
            ))
            .child(note(msg::updates_verified(cx), cx));
        if offer.claims_changed() {
            col = col.child(note(msg::updates_claims_accepted(cx), cx));
        }
        col = col
            .child(note(attested_by(offer, cx), cx))
            .child(value_row(
                "updates/released",
                msg::updates_released(cx),
                offer.summary.released_at.clone().into(),
                false,
                cx,
            ))
            .child(value_row(
                "updates/next-digest",
                msg::updates_next_digest(cx),
                grouped_digest(&offer.bundle_digest),
                true,
                cx,
            ))
            .child(note(msg::updates_install_note(cx), cx))
            .child(div().px(GUTTER).pt(px(8.)).child(pill(
                "updates/approve",
                if installing {
                    msg::updates_installing(cx)
                } else {
                    msg::updates_install(cx)
                },
                Pill::Primary,
                // Off while installing, and while a re-check is out whose
                // answer could replace this offer.
                !installing && self.stores.updates.read(cx).can_approve(),
                // The approval: the one call that stages anything.
                Box::new(cx.listener(|this, _, _, cx| {
                    this.stores.updates.update(cx, |u, cx| u.approve(cx))
                })),
                cx,
            )));
        col.into_any_element()
    }
}

impl Render for UpdatesView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let store = self.stores.updates.read(cx);
        let display = store.display().clone();
        let checking = store.checking();
        let busy = checking || store.installing() || store.reverting();
        let report = store.report().cloned();
        let can_revert = store.can_revert();
        let revert_error = store.revert_error().map(|e| e.to_string());
        let reverting = store.reverting();
        let back = Box::new(cx.listener(|_, _, _, cx| cx.emit(NavEvent::Back)));

        let check: gpui::AnyElement = match &display {
            UpdateDisplay::NotChecked | UpdateDisplay::Checking => {
                status("updates/checking", msg::updates_checking(cx), cx).into_any_element()
            }
            UpdateDisplay::UpToDate { latest_version } => div()
                .child(status(
                    "updates/up-to-date",
                    msg::updates_up_to_date(cx),
                    cx,
                ))
                .child(note(
                    match latest_version {
                        Some(v) => msg::updates_latest(cx, v.clone()),
                        None => msg::updates_no_latest(cx),
                    },
                    cx,
                ))
                .into_any_element(),
            UpdateDisplay::Unbundled { version } => div()
                .child(status(
                    "updates/unbundled",
                    msg::updates_unbundled(cx, version.clone()),
                    cx,
                ))
                .child(note(msg::updates_unbundled_note(cx), cx))
                .into_any_element(),
            UpdateDisplay::Available(offer) => self.render_offer(offer, false, cx),
            UpdateDisplay::Installing(offer) => self.render_offer(offer, true, cx),
            UpdateDisplay::Activated { version } => status(
                "updates/activated",
                msg::updates_restarting(cx, version.clone()),
                cx,
            )
            .into_any_element(),
            UpdateDisplay::InstallFailed { offer, error } => div()
                .child(error_note(
                    "updates/install-failed",
                    msg::updates_install_failed(
                        cx,
                        offer.summary.version.clone(),
                        error.to_string(),
                    ),
                    cx,
                ))
                .into_any_element(),
            UpdateDisplay::Unverifiable { reason } => div()
                .child(error_note(
                    "updates/security-warning",
                    msg::updates_unverifiable(cx, reason.clone()),
                    cx,
                ))
                .child(note(msg::updates_unverifiable_note(cx), cx))
                .into_any_element(),
            UpdateDisplay::CheckFailed { reason } => div()
                .child(status(
                    "updates/check-failed",
                    msg::updates_check_failed(cx),
                    cx,
                ))
                .child(note(reason.clone().into(), cx))
                .into_any_element(),
        };

        // The last update that did not start comes first: it is the news.
        let fell_back = match &report {
            Some(Ok(report)) => report.failed.clone().map(|failed| {
                // What the shell fell back to: an earlier update by its
                // release, or the built-in bundle by this app's version.
                let back_on = if report.serving.version == EMBEDDED {
                    env!("CARGO_PKG_VERSION").to_string()
                } else {
                    report.serving.version.clone()
                };
                error_note(
                    "updates/fell-back",
                    msg::updates_fell_back(cx, failed, back_on),
                    cx,
                )
            }),
            _ => None,
        };
        let mut running = div().flex().flex_col();
        match &report {
            Some(Ok(report)) => {
                let serving: SharedString = if report.serving.version == EMBEDDED {
                    msg::updates_embedded(cx)
                } else {
                    report.serving.version.clone().into()
                };
                // The probe names the version (data, not words), so a test
                // can tell which bundle the shell says it served.
                let serving_probe: &'static str = match report.serving.version.as_str() {
                    EMBEDDED => "updates/serving/embedded",
                    _ if report.serving.confirmed => "updates/serving/update-confirmed",
                    _ => "updates/serving/update-on-trial",
                };
                running = running
                    .child(value_row(
                        serving_probe,
                        msg::updates_serving(cx),
                        serving,
                        false,
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
                if let Some(digest) = &report.serving.digest {
                    running = running.child(value_row(
                        "updates/serving-digest",
                        msg::updates_serving_digest(cx),
                        grouped_digest(digest),
                        true,
                        cx,
                    ));
                }
                if let Some(next) = &report.awaiting_reload {
                    running =
                        running.child(note(msg::updates_awaiting_reload(cx, next.clone()), cx));
                }
            }
            Some(Err(e)) => {
                running = running.child(error_note(
                    "updates/report-error",
                    msg::updates_report_failed(cx, e.clone()),
                    cx,
                ));
            }
            None => {}
        }
        if can_revert {
            running = running.child(note(msg::updates_revert_note(cx), cx)).child(
                div().px(GUTTER).child(pill(
                    "updates/revert",
                    msg::updates_revert(cx),
                    Pill::Secondary,
                    !busy,
                    Box::new(cx.listener(|this, _, _, cx| {
                        this.stores.updates.update(cx, |u, cx| u.revert(cx))
                    })),
                    cx,
                )),
            );
        }
        if reverting {
            running = running.child(note(msg::updates_reverting(cx), cx));
        }
        if let Some(e) = revert_error {
            running = running.child(error_note(
                "updates/revert-error",
                msg::updates_revert_failed(cx, e),
                cx,
            ));
        }

        div()
            .size_full()
            .flex()
            .flex_col()
            .child(header(
                "updates",
                msg::updates_title(cx),
                Some(back),
                None,
                cx,
            ))
            .child(
                div()
                    .id("updates-body")
                    .flex_1()
                    .min_h_0()
                    .overflow_y_scroll()
                    .track_scroll(&self.scroll)
                    .pb(px(32.))
                    .children(fell_back.map(|note| div().pt(px(10.)).child(note)))
                    .child(check)
                    .child(div().px(GUTTER).pt(px(10.)).child(pill(
                        "updates/check",
                        if checking {
                            msg::updates_checking_short(cx)
                        } else {
                            msg::updates_check(cx)
                        },
                        Pill::Secondary,
                        !busy,
                        Box::new(cx.listener(|this, _, _, cx| {
                            this.stores.updates.update(cx, |u, cx| u.check_now(cx))
                        })),
                        cx,
                    )))
                    .child(section(msg::updates_section_running(cx), cx))
                    .child(value_row(
                        "updates/version",
                        msg::updates_app_version(cx),
                        format!("v{}", env!("CARGO_PKG_VERSION")).into(),
                        false,
                        cx,
                    ))
                    .child(running),
            )
    }
}
