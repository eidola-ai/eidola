//! Updates window — a small singleton (≈480×360, Eidola menu → "Check for
//! Updates…") that renders the live update-check state machine from
//! `eidola_app_core::updates`:
//!
//! checking / up-to-date (with last-checked time) / verified-update /
//! security-warning / claims-changed (side-by-side expected vs attested).
//!
//! The view reads `Core.update_check` + `Core.update_checking` reactively;
//! a background-poll result that landed while the window was closed is
//! pulled in by `Core::load_last_update_check` on construction. Honest
//! states only: "Checking…" renders solely while a real check is in
//! flight, the security warning never links to the artifact, and the
//! claims-changed state defaults to NOT trusted — "Treat as Update" is the
//! explicit recorded choice.

use eidola_app_core::updates::{ClaimsComparison, UpdateCheckResult, VerifiedRelease};
use gpui::{
    Context, Entity, FocusHandle, InteractiveElement, IntoElement, ParentElement, Render,
    SharedString, StatefulInteractiveElement, Styled, Subscription, Window, div,
};
use gpui_component::{
    ActiveTheme, Disableable, Sizable, StyledExt,
    button::{Button, ButtonVariants},
    h_flex, v_flex,
};

use crate::actions::CloseWindow;
use crate::stores::{Stores, UpdateStore};

/// Vertical reserve under the transparent titlebar so the traffic lights
/// don't land on content (same rationale as `chat::TITLE_BAR_RESERVE`).
#[cfg(target_os = "macos")]
const TITLE_BAR_RESERVE: gpui::Pixels = gpui::px(36.);
#[cfg(not(target_os = "macos"))]
const TITLE_BAR_RESERVE: gpui::Pixels = gpui::px(8.);

/// What the window shows right now — derived from `Core`'s cached update
/// state, never stored. Public so behavior tests assert the derivation
/// for every matrix row.
#[derive(Debug, Clone, PartialEq)]
pub enum UpdatesDisplay {
    /// A check is in flight and there's no completed result to show.
    Checking,
    /// No check has completed yet (and none is running).
    NoneYet,
    UpToDate {
        latest_version: Option<String>,
        checked_at_ms: i64,
    },
    UpdateAvailable {
        release: VerifiedRelease,
    },
    Unverifiable {
        version: String,
        tag: String,
        reason: String,
    },
    ClaimsChanged {
        release: VerifiedRelease,
        comparison: ClaimsComparison,
    },
    CheckFailed {
        message: String,
        checked_at_ms: i64,
    },
}

pub struct UpdatesView {
    update: Entity<UpdateStore>,
    focus_handle: FocusHandle,
    _subscriptions: Vec<Subscription>,
}

impl UpdatesView {
    pub fn new(stores: Stores, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let update = stores.update.clone();
        let focus_handle = cx.focus_handle();
        focus_handle.focus(window, cx);

        let _subscriptions = vec![cx.observe(&update, |_, _, cx| cx.notify())];

        // Reflect any background-poll result that landed while no window
        // was open, then run a fresh check — opening this window *is* the
        // manual check gesture. Both are no-ops on stub stores.
        update.update(cx, |s, cx| {
            s.load_last(cx);
            s.check_now(cx);
        });

        Self {
            update,
            focus_handle,
            _subscriptions,
        }
    }

    /// Derive the display state from the store's cached update state. An
    /// in-flight check only masks the page when there is nothing else to
    /// show; otherwise the last result stays up with a "Checking…" hint.
    pub fn derive_display(store: &UpdateStore) -> UpdatesDisplay {
        let Some(snapshot) = store.snapshot() else {
            return if store.checking() {
                UpdatesDisplay::Checking
            } else {
                UpdatesDisplay::NoneYet
            };
        };
        match &snapshot.result {
            UpdateCheckResult::UpToDate { latest_version } => UpdatesDisplay::UpToDate {
                latest_version: latest_version.clone(),
                checked_at_ms: snapshot.checked_at_ms,
            },
            UpdateCheckResult::UpdateAvailable { release } => UpdatesDisplay::UpdateAvailable {
                release: release.clone(),
            },
            UpdateCheckResult::Unverifiable {
                version,
                tag,
                reason,
            } => UpdatesDisplay::Unverifiable {
                version: version.clone(),
                tag: tag.clone(),
                reason: reason.clone(),
            },
            UpdateCheckResult::ClaimsChanged {
                release,
                comparison,
            } => UpdatesDisplay::ClaimsChanged {
                release: release.clone(),
                comparison: comparison.clone(),
            },
            UpdateCheckResult::CheckFailed { message } => UpdatesDisplay::CheckFailed {
                message: message.clone(),
                checked_at_ms: snapshot.checked_at_ms,
            },
        }
    }

    /// Current display state — what behavior tests assert against.
    pub fn display(&self, cx: &gpui::App) -> UpdatesDisplay {
        Self::derive_display(self.update.read(cx))
    }

    /// Run a manual check now. Public so tests drive it directly.
    pub fn check_now(&mut self, cx: &mut Context<Self>) {
        self.update.update(cx, |s, cx| s.check_now(cx));
    }

    /// The explicit "treat as update" action for a claims-changed release:
    /// records the user's choice (version + manifest hash) in app-core.
    /// Public so tests drive it directly.
    pub fn accept_claims(&mut self, cx: &mut Context<Self>) {
        let UpdatesDisplay::ClaimsChanged { release, .. } = self.display(cx) else {
            return;
        };
        self.update.update(cx, |s, cx| {
            s.accept_changed_claims(release.version.clone(), release.manifest_sha256.clone(), cx)
        });
    }

    fn open_release_page(&mut self, cx: &mut Context<Self>) {
        if let UpdatesDisplay::UpdateAvailable { release } = self.display(cx)
            && let Some(url) = release.release_url
        {
            cx.open_url(&url);
        }
    }
}

/// "just now" / "5m ago" / "3h ago" / "2d ago" — coarse on purpose.
pub fn relative_time(then_ms: i64, now_ms: i64) -> String {
    let delta_s = (now_ms - then_ms).max(0) / 1000;
    match delta_s {
        0..=59 => "just now".into(),
        60..=3599 => format!("{}m ago", delta_s / 60),
        3600..=86_399 => format!("{}h ago", delta_s / 3600),
        _ => format!("{}d ago", delta_s / 86_400),
    }
}

impl Render for UpdatesView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let display = self.display(cx);
        let checking = self.update.read(cx).checking();

        let body: gpui::AnyElement = match &display {
            UpdatesDisplay::Checking => render_checking(cx).into_any_element(),
            UpdatesDisplay::NoneYet => render_none_yet(cx).into_any_element(),
            UpdatesDisplay::UpToDate {
                latest_version,
                checked_at_ms,
            } => {
                render_up_to_date(latest_version.as_deref(), *checked_at_ms, cx).into_any_element()
            }
            UpdatesDisplay::UpdateAvailable { release } => {
                render_update_available(release, self, cx).into_any_element()
            }
            UpdatesDisplay::Unverifiable {
                version, reason, ..
            } => render_unverifiable(version, reason, cx).into_any_element(),
            UpdatesDisplay::ClaimsChanged {
                release,
                comparison,
            } => render_claims_changed(release, comparison, self, cx).into_any_element(),
            UpdatesDisplay::CheckFailed {
                message,
                checked_at_ms,
            } => render_check_failed(message, *checked_at_ms, cx).into_any_element(),
        };

        // Footer: re-check affordance + in-flight hint when a standing
        // result is shown while a re-check runs.
        let footer =
            h_flex()
                .w_full()
                .px_4()
                .py_3()
                .border_t_1()
                .border_color(theme.border)
                .gap_3()
                .child(
                    Button::new("check-now")
                        .label(if checking { "Checking…" } else { "Check Now" })
                        .small()
                        .disabled(checking)
                        .on_click(cx.listener(|this, _, _, cx| this.check_now(cx))),
                )
                .child(div().flex_1())
                .child(div().text_xs().text_color(theme.muted_foreground).child(
                    SharedString::from(format!("Eidola {}", env!("CARGO_PKG_VERSION"))),
                ));

        v_flex()
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(|_, _: &CloseWindow, window, _| {
                window.remove_window();
            }))
            .size_full()
            .bg(theme.background)
            .text_color(theme.foreground)
            .child(div().h(TITLE_BAR_RESERVE).w_full())
            .child(
                div()
                    .id("updates-body")
                    .flex_1()
                    .w_full()
                    .overflow_y_scroll()
                    .child(body),
            )
            .child(footer)
    }
}

fn render_checking(cx: &gpui::App) -> impl IntoElement {
    let theme = cx.theme();
    centered_col()
        .child(div().font_medium().child("Checking for updates…"))
        .child(
            div()
                .text_sm()
                .text_color(theme.muted_foreground)
                .child("Contacting the release feed."),
        )
}

fn render_none_yet(cx: &gpui::App) -> impl IntoElement {
    let theme = cx.theme();
    centered_col().child(
        div()
            .text_color(theme.muted_foreground)
            .child("No update check has completed yet."),
    )
}

fn render_up_to_date(
    latest_version: Option<&str>,
    checked_at_ms: i64,
    cx: &gpui::App,
) -> impl IntoElement {
    let theme = cx.theme();
    let latest_line = match latest_version {
        Some(v) => format!("Latest release: v{v}"),
        None => "No release is marked latest yet.".to_string(),
    };
    centered_col()
        .child(div().font_medium().child("Eidola is up to date."))
        .child(
            div()
                .text_sm()
                .text_color(theme.muted_foreground)
                .child(SharedString::from(latest_line)),
        )
        .child(
            div()
                .text_xs()
                .text_color(theme.muted_foreground)
                .child(SharedString::from(format!(
                    "Last checked {}",
                    relative_time(checked_at_ms, eidola_app_core::now_ms())
                ))),
        )
}

fn render_check_failed(message: &str, checked_at_ms: i64, cx: &gpui::App) -> impl IntoElement {
    // Quiet by design: an offline blip is not a security signal.
    let theme = cx.theme();
    centered_col()
        .child(
            div()
                .text_color(theme.muted_foreground)
                .child("Couldn't check for updates (offline?)."),
        )
        .child(
            div()
                .text_xs()
                .text_color(theme.muted_foreground)
                .child(SharedString::from(message.to_string())),
        )
        .child(
            div()
                .text_xs()
                .text_color(theme.muted_foreground)
                .child(SharedString::from(format!(
                    "Last attempt {}",
                    relative_time(checked_at_ms, eidola_app_core::now_ms())
                ))),
        )
}

fn render_update_available(
    release: &VerifiedRelease,
    view: &UpdatesView,
    cx: &Context<UpdatesView>,
) -> impl IntoElement {
    let theme = cx.theme();
    let _ = view;
    let mut col = centered_col()
        .child(div().font_semibold().child(SharedString::from(format!(
            "Eidola v{} is available.",
            release.version
        ))))
        .child(
            div()
                .text_sm()
                .text_color(theme.muted_foreground)
                .child("Cryptographically verified against this build's embedded trust root."),
        )
        .child(
            div()
                .text_xs()
                .text_color(theme.muted_foreground)
                .child(SharedString::from(format!(
                    "Signed by the pinned release workflow · Rekor #{}",
                    release.rekor_log_index
                ))),
        );
    if release.claims_accepted {
        col = col.child(div().text_xs().text_color(theme.muted_foreground).child(
            "Its claims differ from this build's expectations; you chose to treat it as an update.",
        ));
    }
    col.child(div().h_2()).child(
        Button::new("open-release")
            .primary()
            .label("View Release…")
            .on_click(cx.listener(|this, _, _, cx| this.open_release_page(cx))),
    )
}

fn render_unverifiable(version: &str, reason: &str, cx: &gpui::App) -> impl IntoElement {
    // Hard, visible security state. Never silent, never a link to the
    // artifact.
    let theme = cx.theme();
    v_flex()
        .px_5()
        .py_4()
        .gap_3()
        .w_full()
        .child(
            v_flex()
                .w_full()
                .px_3()
                .py_3()
                .gap_2()
                .rounded_md()
                .bg(theme.danger.opacity(0.08))
                .child(
                    div()
                        .font_semibold()
                        .text_color(theme.danger)
                        .child("Security warning"),
                )
                .child(
                    div()
                        .text_sm()
                        .text_color(theme.danger)
                        .child(SharedString::from(format!(
                            "The release marked latest (v{version}) could not be verified."
                        ))),
                )
                .child(
                    div()
                        .text_xs()
                        .text_color(theme.danger)
                        .child(SharedString::from(reason.to_string())),
                ),
        )
        .child(div().text_sm().text_color(theme.muted_foreground).child(
            "This may be a fake release or a compromised update channel. Eidola will \
                     not link to it. This warning stays until a check finds a release that \
                     verifies.",
        ))
}

fn render_claims_changed(
    release: &VerifiedRelease,
    comparison: &ClaimsComparison,
    _view: &UpdatesView,
    cx: &Context<UpdatesView>,
) -> impl IntoElement {
    let theme = cx.theme();
    let matching = comparison.expected.len().saturating_sub(
        comparison
            .deltas
            .iter()
            .filter(|d| d.expected.is_some())
            .count(),
    );

    let mut table = v_flex().w_full().text_xs().child(
        h_flex()
            .w_full()
            .gap_2()
            .pb_1()
            .border_b_1()
            .border_color(theme.border)
            .text_color(theme.muted_foreground)
            .child(div().flex_1().child("Claim"))
            .child(div().flex_1().child("Expected"))
            .child(div().flex_1().child("Attested")),
    );
    for delta in &comparison.deltas {
        table = table.child(
            h_flex()
                .w_full()
                .gap_2()
                .py_1()
                .border_b_1()
                .border_color(theme.border)
                .child(div().flex_1().child(SharedString::from(delta.key.clone())))
                .child(div().flex_1().child(SharedString::from(
                    delta.expected.clone().unwrap_or_else(|| "—".into()),
                )))
                .child(
                    div()
                        .flex_1()
                        .text_color(theme.danger)
                        .child(SharedString::from(
                            delta.attested.clone().unwrap_or_else(|| "absent".into()),
                        )),
                ),
        );
    }

    v_flex()
        .px_5()
        .py_4()
        .gap_3()
        .w_full()
        .child(div().font_semibold().child(SharedString::from(format!(
            "Release v{} verified — but its claims changed.",
            release.version
        ))))
        .child(div().text_sm().text_color(theme.muted_foreground).child(
            "The release is authentically signed, but what it attests differs from \
                     what this build expects. Review the differences; nothing is treated as \
                     an update unless you explicitly accept the change.",
        ))
        .child(table)
        .child(
            div()
                .text_xs()
                .text_color(theme.muted_foreground)
                .child(SharedString::from(format!(
                    "{matching} of {} expected claims match.",
                    comparison.expected.len()
                ))),
        )
        .child(
            h_flex().w_full().gap_2().child(
                Button::new("treat-as-update")
                    .outline()
                    .label("Treat as Update")
                    .small()
                    .on_click(cx.listener(|this, _, _, cx| this.accept_claims(cx))),
            ),
        )
}

fn centered_col() -> gpui::Div {
    v_flex()
        .px_5()
        .py_4()
        .gap_2()
        .w_full()
        .items_center()
        .pt_8()
}
