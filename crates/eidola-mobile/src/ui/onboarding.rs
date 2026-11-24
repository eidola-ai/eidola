//! The first run, on a phone: the desktop's onboarding (`crates/eidola-gui/src/onboarding/`)
//! as a short sequence of pages, one at a time, in the desktop's words.
//!
//! **Pages**: what Eidola is (the desktop's "Pause here" and "Your control"),
//! "Your responsibility", then the choice — a new account, an account the
//! reader already has, or going on without one. A new account passes through
//! the consent page and shows its credentials to save; an existing one is
//! checked against its balance; both end at adding credit (which leads to
//! the plans, checkout in the system browser) or "later".
//!
//! **The doctrine is the desktop's, rule for rule:**
//! - **Consent is bound to the documents rendered.** The consent page shows
//!   the snapshot of required documents `AccountStore::terms` holds (fetched
//!   when the page opens), and the agreement records the `(document,
//!   sha256)` pairs on screen when the box was ticked — never a flag.
//!   [`OnboardingView::consented_terms`] is the only way to the value
//!   creation submits; it answers only while the agreement matches the
//!   snapshot in hand, so a refresh that changed or added a document unticks
//!   the box by itself. The box is inert until documents are on screen.
//! - **An account created but not saved** (`AppError::AccountNotSaved`)
//!   shows its credential as copy rows, names the file it was also written to,
//!   and keeps the Create button off while those rows are on screen.
//! - **Linking checks the balance, and rolls the credentials back** if the
//!   server refuses them; "no such account" and "wrong secret" are one
//!   answer, by design of the server, so the words name both.
//! - **Going on without an account** turns the Eidola backend off, which is
//!   what keeps the first run from opening at the next launch; setting up an
//!   account later (Settings ▸ Account) turns it back on.
//! - **Failures are held as values** and worded at render.

use eidola_app_core::error::AppError;
use eidola_app_core::{TermsAcceptance, TermsDocument};
use gpui::{
    AnyElement, App, AppContext as _, Context, Entity, EventEmitter, InteractiveElement as _,
    IntoElement, MouseButton, ParentElement as _, Render, Role, ScrollHandle, SharedString,
    StatefulInteractiveElement as _, Styled as _, Task, Window, div, prelude::FluentBuilder as _,
    px,
};
use gpui_component::{ActiveTheme as _, Icon};
use gpui_kit_assets::IconName;

use super::account_view::format_credits;
use super::i18n::msg;
use super::nav::{Dest, NavEvent};
use super::probe::Probe as _;
use super::stores::Stores;
use super::widgets::{
    Pill, Prose, TextField, copy_row, error_note, header, keep_keyboard, note, pill, text_field,
};
use super::{GUTTER, TOUCH};

/// The published policies, linked when the server names no documents.
const TERMS_URL: &str = "https://www.eidola.ai/terms/";
const PRIVACY_URL: &str = "https://www.eidola.ai/privacy/";
/// The documentation behind the unlinkability claim.
const UNLINKABILITY_URL: &str = "https://www.eidola.ai/docs/privacy-guarantees/#2-unlinkability";

/// Where the first run begins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OnboardingStart {
    /// At launch: from what Eidola is.
    Welcome,
    /// From Settings ▸ Account: at the choice.
    Choose,
    /// From a checkout the server refused for want of the terms: at the
    /// agreement, for the account already configured.
    AcceptTerms,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Page {
    About,
    Responsibility,
    GetStarted,
    CreateAccount,
    NewAccount,
    ExistingAccount,
    Purchase,
    /// The agreement to the current documents, recorded for the account this
    /// profile already holds.
    AcceptTerms,
}

pub struct OnboardingView {
    stores: Stores,
    start: OnboardingStart,
    /// The pages walked, the current last.
    pages: Vec<Page>,
    prose: Entity<Prose>,
    scroll: ScrollHandle,
    /// The `(document, sha256)` pairs on screen when the box was ticked.
    agreed_to: Option<Vec<(String, String)>>,
    /// This screen asked for the creation the store is running, and moves
    /// to the new account's page when it lands.
    awaiting_create: bool,
    id_field: Entity<TextField>,
    secret_field: Entity<TextField>,
    /// A link was refused before any request: a field was empty.
    missing_credentials: bool,
    leaving: Option<Task<()>>,
    open_task: Option<Task<()>>,
    open_error: Option<String>,
}

impl EventEmitter<NavEvent> for OnboardingView {}

/// The agreement's key: the documents' names and text hashes, sorted.
pub fn consent_fingerprint(documents: &[TermsDocument]) -> Vec<(String, String)> {
    let mut fingerprint: Vec<(String, String)> = documents
        .iter()
        .map(|d| (d.document.clone(), d.sha256.clone()))
        .collect();
    fingerprint.sort();
    fingerprint
}

impl OnboardingView {
    pub fn new(
        stores: Stores,
        start: OnboardingStart,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        // The account operations are the store's (navigation cancels none of
        // them); this screen follows their outcomes.
        cx.observe(&stores.account, |this, account, cx| {
            if matches!(account.read(cx).link_outcome(), Some(Ok(_)))
                && !this.secret_field.read(cx).value(cx).is_empty()
            {
                this.secret_field.update(cx, |f, cx| f.clear(cx));
            }
            if this.awaiting_create && matches!(account.read(cx).create_outcome(), Some(Ok(_))) {
                this.awaiting_create = false;
                // The new account's credentials, wherever in the account
                // steps the reader is now: they are what to save.
                if this.page() != Page::NewAccount {
                    this.pages.push(Page::NewAccount);
                }
            }
            cx.notify();
        })
        .detach();
        stores.account.update(cx, |a, cx| {
            a.forget_settled_outcomes();
            if start == OnboardingStart::AcceptTerms {
                a.refresh_terms(cx);
            }
        });
        let creating = stores.account.read(cx).creating();
        let first = match start {
            OnboardingStart::Welcome => Page::About,
            OnboardingStart::Choose => Page::GetStarted,
            OnboardingStart::AcceptTerms => Page::AcceptTerms,
        };
        Self {
            prose: cx.new(|cx| Prose::new(window, cx)),
            id_field: cx.new(|cx| TextField::new(window, cx)),
            secret_field: cx.new(|cx| TextField::new(window, cx)),
            stores,
            start,
            pages: vec![first],
            scroll: ScrollHandle::new(),
            agreed_to: None,
            // A creation still running from an earlier visit lands here.
            awaiting_create: creating,
            missing_credentials: false,
            leaving: None,
            open_task: None,
            open_error: None,
        }
    }

    pub fn page(&self) -> Page {
        *self.pages.last().unwrap_or(&Page::About)
    }

    fn go(&mut self, page: Page, window: &mut Window, cx: &mut Context<Self>) {
        if matches!(page, Page::CreateAccount | Page::AcceptTerms) {
            self.agreed_to = None;
            self.stores.account.update(cx, |a, cx| {
                a.refresh_terms(cx);
                a.forget_accept_outcome();
            });
        }
        self.pages.push(page);
        self.open_error = None;
        self.scroll.set_offset(gpui::point(px(0.), px(0.)));
        window.dismiss_virtual_keyboard();
        cx.notify();
    }

    fn back(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.pages.len() > 1 {
            self.pages.pop();
            self.open_error = None;
            window.dismiss_virtual_keyboard();
            cx.notify();
        } else {
            cx.emit(NavEvent::Back);
        }
    }

    /// Whether there is anywhere to go back to: a page before this one, or
    /// (from Settings) the screen this was opened from.
    fn can_go_back(&self) -> bool {
        self.pages.len() > 1 || self.start != OnboardingStart::Welcome
    }

    /// The first run is over: to the Library at launch, back where it was
    /// opened from otherwise.
    fn finish(&mut self, cx: &mut Context<Self>) {
        cx.emit(match self.start {
            OnboardingStart::Welcome => NavEvent::Home,
            OnboardingStart::Choose | OnboardingStart::AcceptTerms => NavEvent::Back,
        });
    }

    /// Go on without an account: the Eidola backend off, then the Library.
    pub fn skip_account(&mut self, cx: &mut Context<Self>) {
        if self.leaving.is_some() {
            return;
        }
        let disable = self.stores.account.read(cx).set_eidola_enabled(false);
        self.leaving = Some(cx.spawn(async move |this, cx| {
            if let Err(e) = disable.await {
                log::error!("could not turn the Eidola backend off: {e}");
            }
            this.update(cx, |this, cx| {
                this.leaving = None;
                this.finish(cx);
            })
            .ok();
        }));
    }

    /// Record or withdraw agreement to the documents **currently on screen**.
    pub fn set_agreement(&mut self, checked: bool, cx: &mut Context<Self>) {
        self.agreed_to = if checked {
            self.stores
                .account
                .read(cx)
                .terms()
                .value()
                .map(|docs| consent_fingerprint(docs))
        } else {
            None
        };
        cx.notify();
    }

    /// The documents the reader agreed to **and that are still the ones in
    /// hand**: `None` when no snapshot is loaded, the box is unticked, or the
    /// snapshot moved since it was ticked. The Create button's state and the
    /// creation both read this, at the moment they need it.
    pub fn consented_terms(&self, cx: &App) -> Option<Vec<TermsDocument>> {
        let documents = self.stores.account.read(cx).terms().value()?.clone();
        let agreed_to = self.agreed_to.as_ref()?;
        (*agreed_to == consent_fingerprint(&documents)).then_some(documents)
    }

    fn unsaved_on_screen(&self, cx: &App) -> bool {
        matches!(
            self.stores.account.read(cx).create_outcome(),
            Some(Err(AppError::AccountNotSaved { .. }))
        )
    }

    fn begin_create(&mut self, cx: &mut Context<Self>) {
        // An account created but not saved is on screen with the only copy of
        // its secret; another attempt would replace it.
        if self.stores.account.read(cx).identity_busy() || self.unsaved_on_screen(cx) {
            return;
        }
        let Some(accepted) = self.consented_terms(cx) else {
            return;
        };
        if self
            .stores
            .account
            .update(cx, |a, cx| a.create(accepted, cx))
        {
            self.awaiting_create = true;
        }
    }

    fn begin_verify(&mut self, cx: &mut Context<Self>) {
        if self.stores.account.read(cx).identity_busy() {
            return;
        }
        self.id_field.update(cx, |f, cx| f.commit(cx));
        self.secret_field.update(cx, |f, cx| f.commit(cx));
        let id = self.id_field.read(cx).value(cx);
        let secret = self.secret_field.read(cx).value(cx);
        self.missing_credentials = id.is_empty() || secret.is_empty();
        if self.missing_credentials {
            cx.notify();
            return;
        }
        self.stores
            .account
            .update(cx, |a, cx| a.link(id, secret, cx));
        cx.notify();
    }

    /// Record the agreement to the documents on screen for the account this
    /// profile holds.
    fn begin_accept(&mut self, cx: &mut Context<Self>) {
        if self.stores.account.read(cx).accepting() {
            return;
        }
        let Some(presented) = self.consented_terms(cx) else {
            return;
        };
        self.stores
            .account
            .update(cx, |a, cx| a.accept_terms(presented, cx));
    }

    fn open(&mut self, url: String, cx: &mut Context<Self>) {
        let open = self.stores.host.open_external(url);
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

    fn body_text(&self, cx: &App) -> SharedString {
        match self.page() {
            Page::About => format!(
                "{}\n\n{}",
                msg::onboarding_pause_body(cx),
                msg::onboarding_control_body(cx)
            )
            .into(),
            Page::Responsibility => msg::onboarding_responsibility_body(cx),
            Page::GetStarted => msg::onboarding_get_started_body(cx),
            Page::CreateAccount => msg::onboarding_create_account_body(cx),
            Page::NewAccount => msg::onboarding_new_account_body(cx),
            Page::ExistingAccount => msg::onboarding_existing_account_body(cx),
            Page::Purchase => msg::onboarding_purchase_body(cx),
            Page::AcceptTerms => msg::onboarding_accept_terms_body(cx),
        }
    }
}

/// A link out of the app, through the system browser.
fn link_row(
    name: String,
    label: SharedString,
    url: String,
    cx: &mut Context<OnboardingView>,
) -> impl IntoElement {
    let text = msg::onboarding_link_external(cx, label.to_string());
    div()
        .id(SharedString::from(name.clone()))
        .probe(name, Role::Link, text.clone())
        .mx(GUTTER)
        .min_h(TOUCH)
        .flex()
        .items_center()
        .text_size(px(16.))
        .text_color(cx.theme().link)
        .on_mouse_down(MouseButton::Left, keep_keyboard)
        .on_click(cx.listener(move |this, _, _, cx| this.open(url.clone(), cx)))
        .child(text)
}

/// A document's link label: its published name and the version acceptance
/// will be recorded for.
fn document_link(doc: &TermsDocument, cx: &App) -> (String, SharedString) {
    let slug: String = doc
        .document
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    let name = match doc.document.as_str() {
        "terms_of_service" => msg::onboarding_link_terms_of_service(cx),
        "privacy_policy" => msg::onboarding_link_privacy_policy(cx),
        other => SharedString::from(other.replace('_', " ")),
    };
    (
        slug,
        msg::onboarding_document_version(cx, name.to_string(), doc.version),
    )
}

impl OnboardingView {
    fn cta(
        &self,
        name: &'static str,
        label: SharedString,
        kind: Pill,
        enabled: bool,
        on_click: impl Fn(&mut Self, &mut Window, &mut Context<Self>) + 'static,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        div().px(GUTTER).py(px(4.)).child(pill(
            name,
            label,
            kind,
            enabled,
            Box::new(cx.listener(move |this, _, window, cx| on_click(this, window, cx))),
            cx,
        ))
    }

    fn render_extras(&mut self, cx: &mut Context<Self>) -> Vec<AnyElement> {
        let mut out: Vec<AnyElement> = Vec::new();
        match self.page() {
            Page::About | Page::Responsibility => {
                out.push(
                    self.cta(
                        "onboarding/cta/understood",
                        msg::onboarding_cta_understood(cx),
                        Pill::Primary,
                        true,
                        |this, window, cx| {
                            let next = if this.page() == Page::About {
                                Page::Responsibility
                            } else {
                                Page::GetStarted
                            };
                            this.go(next, window, cx)
                        },
                        cx,
                    )
                    .into_any_element(),
                );
            }
            Page::GetStarted => {
                out.push(
                    link_row(
                        "onboarding/link/unlinkability".into(),
                        msg::onboarding_link_unlinkability(cx),
                        UNLINKABILITY_URL.into(),
                        cx,
                    )
                    .into_any_element(),
                );
                out.push(
                    self.cta(
                        "onboarding/cta/new-account",
                        msg::onboarding_cta_new_account(cx),
                        Pill::Primary,
                        true,
                        |this, window, cx| this.go(Page::CreateAccount, window, cx),
                        cx,
                    )
                    .into_any_element(),
                );
                out.push(
                    self.cta(
                        "onboarding/cta/existing-account",
                        msg::onboarding_cta_existing_account(cx),
                        Pill::Secondary,
                        true,
                        |this, window, cx| this.go(Page::ExistingAccount, window, cx),
                        cx,
                    )
                    .into_any_element(),
                );
                if !self.stores.account.read(cx).has_account() {
                    out.push(
                        div()
                            .id("onboarding-skip")
                            .probe(
                                "onboarding/cta/skip-account",
                                Role::Button,
                                msg::onboarding_cta_skip_account(cx),
                            )
                            .mx(GUTTER)
                            .mt(px(8.))
                            .min_h(TOUCH)
                            .flex()
                            .items_center()
                            .justify_center()
                            .text_center()
                            .text_size(px(15.))
                            .text_color(cx.theme().muted_foreground)
                            .on_mouse_down(MouseButton::Left, keep_keyboard)
                            .on_click(cx.listener(|this, _, _, cx| this.skip_account(cx)))
                            .child(msg::onboarding_cta_skip_account(cx))
                            .into_any_element(),
                    );
                }
            }
            Page::CreateAccount => out.extend(self.render_create(cx)),
            Page::NewAccount => {
                let created = match self.stores.account.read(cx).create_outcome() {
                    Some(Ok(created)) => Some(created.clone()),
                    _ => None,
                };
                if let Some(created) = &created {
                    let (id, secret): (SharedString, SharedString) =
                        (created.id.clone().into(), created.secret.clone().into());
                    out.push(
                        copy_row(
                            "onboarding/copy/account-id",
                            msg::onboarding_account_id(cx),
                            id.clone(),
                            id.to_string(),
                            cx,
                        )
                        .into_any_element(),
                    );
                    out.push(
                        copy_row(
                            "onboarding/copy/account-secret",
                            msg::onboarding_account_secret(cx),
                            secret.clone(),
                            secret.to_string(),
                            cx,
                        )
                        .into_any_element(),
                    );
                }
                out.push(note(msg::onboarding_new_account_note(cx), cx).into_any_element());
                // The account exists either way; whether its agreement to the
                // terms was recorded is what the server reported back. A
                // purchase is refused until it is, so say so here, and offer
                // the agreement again for the documents current now.
                if let Some(created) = &created
                    && let Some(line) = terms_standing(&created.terms, cx)
                {
                    out.push(error_note("onboarding/terms-standing", line, cx).into_any_element());
                    out.push(
                        self.cta(
                            "onboarding/cta/review-terms",
                            msg::onboarding_cta_review_terms(cx),
                            Pill::Primary,
                            true,
                            |this, window, cx| this.go(Page::AcceptTerms, window, cx),
                            cx,
                        )
                        .into_any_element(),
                    );
                }
                out.push(
                    self.cta(
                        "onboarding/cta/saved",
                        msg::onboarding_cta_saved(cx),
                        Pill::Primary,
                        true,
                        |this, window, cx| this.go(Page::Purchase, window, cx),
                        cx,
                    )
                    .into_any_element(),
                );
            }
            Page::ExistingAccount => out.extend(self.render_existing(cx)),
            Page::AcceptTerms => out.extend(self.render_accept(cx)),
            Page::Purchase => {
                out.push(note(msg::onboarding_purchase_checkout_note(cx), cx).into_any_element());
                out.push(
                    self.cta(
                        "onboarding/cta/plans",
                        msg::onboarding_cta_plans(cx),
                        Pill::Primary,
                        true,
                        |_, _, cx| cx.emit(NavEvent::To(Dest::Purchase)),
                        cx,
                    )
                    .into_any_element(),
                );
                out.push(
                    self.cta(
                        "onboarding/cta/purchase-later",
                        msg::onboarding_cta_purchase_later(cx),
                        Pill::Secondary,
                        true,
                        |this, _, cx| this.finish(cx),
                        cx,
                    )
                    .into_any_element(),
                );
            }
        }
        if let Some(e) = &self.open_error {
            out.push(
                error_note(
                    "onboarding/error/open",
                    msg::account_door_failed(cx, e.clone()),
                    cx,
                )
                .into_any_element(),
            );
        }
        out
    }

    /// The documents in hand, each linked, and the agreement bound to them:
    /// the consent the creation and a later agreement both submit.
    fn render_consent(&mut self, cx: &mut Context<Self>) -> Vec<AnyElement> {
        let mut out: Vec<AnyElement> = Vec::new();
        let terms = self.stores.account.read(cx).terms().clone();
        let documents = terms.value().cloned();
        match documents.as_deref() {
            // A server with no acceptance gate: nothing to record, but the
            // published policies still govern, so they are linked.
            Some([]) => {
                out.push(
                    link_row(
                        "onboarding/link/terms-of-service".into(),
                        msg::onboarding_link_terms_of_service(cx),
                        TERMS_URL.into(),
                        cx,
                    )
                    .into_any_element(),
                );
                out.push(
                    link_row(
                        "onboarding/link/privacy-policy".into(),
                        msg::onboarding_link_privacy_policy(cx),
                        PRIVACY_URL.into(),
                        cx,
                    )
                    .into_any_element(),
                );
            }
            Some(docs) => {
                for doc in docs {
                    let (slug, label) = document_link(doc, cx);
                    out.push(
                        link_row(
                            format!("onboarding/terms/{slug}"),
                            label,
                            doc.url.clone(),
                            cx,
                        )
                        .into_any_element(),
                    );
                }
            }
            None if terms.error().is_none() => out.push(
                div()
                    .id("onboarding-terms-loading")
                    .probe(
                        "onboarding/terms/loading",
                        Role::Label,
                        msg::onboarding_terms_loading(cx),
                    )
                    .child(note(msg::onboarding_terms_loading(cx), cx))
                    .into_any_element(),
            ),
            None => {}
        }
        if let Some(e) = terms.error() {
            out.push(
                error_note(
                    "onboarding/error/terms",
                    SharedString::from(e.to_string()),
                    cx,
                )
                .into_any_element(),
            );
            out.push(
                self.cta(
                    "onboarding/terms/retry",
                    msg::onboarding_terms_retry(cx),
                    Pill::Secondary,
                    true,
                    |this, _, cx| this.stores.account.update(cx, |a, cx| a.refresh_terms(cx)),
                    cx,
                )
                .into_any_element(),
            );
        }

        // The sentence the reader affirms: the box's label and its accessible
        // name, said once.
        let consent = msg::onboarding_consent_agree(cx);
        let have_documents = documents.is_some();
        let agreed = self.consented_terms(cx).is_some();
        let theme = cx.theme().clone();
        out.push(
            div()
                .id("onboarding-agree")
                .probe("onboarding/agree", Role::CheckBox, consent.clone())
                .aria_toggled(agreed.into())
                .mx(GUTTER)
                .mt(px(12.))
                .min_h(TOUCH)
                .flex()
                .items_center()
                .gap_3()
                .on_mouse_down(MouseButton::Left, keep_keyboard)
                .when(have_documents, |d| {
                    d.on_click(cx.listener(move |this, _, _, cx| this.set_agreement(!agreed, cx)))
                })
                .when(!have_documents, |d| d.opacity(0.5))
                .child(
                    div()
                        .flex_none()
                        .size(px(26.))
                        .rounded(px(6.))
                        .border_2()
                        .border_color(if agreed { theme.primary } else { theme.border })
                        .bg(if agreed {
                            theme.primary
                        } else {
                            theme.background
                        })
                        .flex()
                        .items_center()
                        .justify_center()
                        .when(agreed, |d| {
                            d.child(
                                Icon::new(IconName::Check)
                                    .size(px(18.))
                                    .text_color(theme.primary_foreground),
                            )
                        }),
                )
                .child(
                    div()
                        .flex_1()
                        .text_size(px(16.))
                        .line_height(px(22.))
                        .child(consent),
                )
                .into_any_element(),
        );
        out
    }

    fn render_create(&mut self, cx: &mut Context<Self>) -> Vec<AnyElement> {
        let mut out = self.render_consent(cx);
        let agreed = self.consented_terms(cx).is_some();
        let creating = self.stores.account.read(cx).creating();
        let unsaved = self.unsaved_on_screen(cx);
        out.push(
            self.cta(
                "onboarding/cta/create",
                if creating {
                    msg::onboarding_cta_create_pending(cx)
                } else {
                    msg::onboarding_cta_create(cx)
                },
                Pill::Primary,
                agreed && !self.stores.account.read(cx).identity_busy() && !unsaved,
                |this, _, cx| this.begin_create(cx),
                cx,
            )
            .into_any_element(),
        );

        let outcome = self.stores.account.read(cx).create_outcome().cloned();
        match outcome.as_ref().and_then(|o| o.as_ref().err()) {
            Some(AppError::AccountNotSaved { account, reason }) => {
                out.push(
                    error_note(
                        "onboarding/error/create",
                        msg::onboarding_create_not_saved(cx, reason.clone()),
                        cx,
                    )
                    .into_any_element(),
                );
                out.push(
                    copy_row(
                        "onboarding/copy/unsaved-account-id",
                        msg::onboarding_account_id(cx),
                        account.account_id.clone().into(),
                        account.account_id.clone(),
                        cx,
                    )
                    .into_any_element(),
                );
                out.push(
                    copy_row(
                        "onboarding/copy/unsaved-account-secret",
                        msg::onboarding_account_secret(cx),
                        account.secret.expose().to_string().into(),
                        account.secret.expose().to_string(),
                        cx,
                    )
                    .into_any_element(),
                );
                if let Some(path) = &account.recovery_file {
                    let line = msg::onboarding_create_recovery_file(cx, path.display().to_string());
                    out.push(
                        div()
                            .id("onboarding-recovery-file")
                            .probe("onboarding/create/recovery-file", Role::Label, line.clone())
                            .child(note(line, cx))
                            .into_any_element(),
                    );
                }
            }
            Some(e) => out.push(
                error_note(
                    "onboarding/error/create",
                    msg::onboarding_create_failed(cx, e.to_string()),
                    cx,
                )
                .into_any_element(),
            ),
            None => {}
        }
        out
    }

    fn render_existing(&mut self, cx: &mut Context<Self>) -> Vec<AnyElement> {
        let mut out: Vec<AnyElement> = Vec::new();
        let outcome = self.stores.account.read(cx).link_outcome().cloned();
        let linked = matches!(outcome, Some(Ok(_)));
        if !linked {
            out.push(
                text_field(
                    &self.id_field,
                    "onboarding/input/account-id",
                    msg::onboarding_account_id(cx),
                    msg::onboarding_account_id_placeholder(cx),
                    cx,
                )
                .into_any_element(),
            );
            out.push(
                text_field(
                    &self.secret_field,
                    "onboarding/input/account-secret",
                    msg::onboarding_account_secret(cx),
                    msg::onboarding_account_secret_placeholder(cx),
                    cx,
                )
                .into_any_element(),
            );
            let verifying = self.stores.account.read(cx).linking();
            out.push(
                self.cta(
                    "onboarding/cta/verify",
                    if verifying {
                        msg::onboarding_cta_verify_pending(cx)
                    } else {
                        msg::onboarding_cta_verify(cx)
                    },
                    Pill::Primary,
                    !self.stores.account.read(cx).identity_busy(),
                    |this, _, cx| this.begin_verify(cx),
                    cx,
                )
                .into_any_element(),
            );
        }
        if self.missing_credentials && !linked {
            out.push(
                error_note(
                    "onboarding/error/verify",
                    msg::onboarding_verify_missing_credentials(cx),
                    cx,
                )
                .into_any_element(),
            );
        }
        match &outcome {
            Some(Ok(available)) => {
                let line =
                    msg::onboarding_verified_balance(cx, *available, format_credits(*available));
                out.push(
                    div()
                        .id("onboarding-verified")
                        .probe("onboarding/verified", Role::Label, line.clone())
                        .px(GUTTER)
                        .py(px(8.))
                        .font_family(eidola_theme::FONT_FAMILY)
                        .text_size(px(18.))
                        .child(line)
                        .into_any_element(),
                );
                out.push(
                    self.cta(
                        "onboarding/cta/existing-purchase",
                        msg::onboarding_cta_existing_purchase(cx),
                        Pill::Secondary,
                        true,
                        |this, window, cx| this.go(Page::Purchase, window, cx),
                        cx,
                    )
                    .into_any_element(),
                );
                out.push(
                    self.cta(
                        "onboarding/cta/existing-done",
                        msg::onboarding_cta_existing_done(cx),
                        Pill::Primary,
                        true,
                        |this, _, cx| this.finish(cx),
                        cx,
                    )
                    .into_any_element(),
                );
            }
            Some(Err(e)) if !self.missing_credentials => {
                let text = match e {
                    // The server answers "no such account" and "wrong
                    // secret" alike, so the words name both.
                    AppError::Server {
                        status: 401 | 403, ..
                    } => msg::onboarding_verify_unverifiable(cx),
                    other => msg::onboarding_verify_failed(cx, other.to_string()),
                };
                out.push(error_note("onboarding/error/verify", text, cx).into_any_element());
            }
            _ => {}
        }
        out
    }

    /// The agreement again, for the account this profile holds: the documents
    /// current now, the box bound to them, and what the server then reports.
    fn render_accept(&mut self, cx: &mut Context<Self>) -> Vec<AnyElement> {
        let mut out = self.render_consent(cx);
        let agreed = self.consented_terms(cx).is_some();
        let account = self.stores.account.read(cx);
        let accepting = account.accepting();
        let outcome = account.accept_outcome().cloned();
        let complete = matches!(outcome, Some(Ok(TermsAcceptance::Complete)));
        if !complete {
            out.push(
                self.cta(
                    "onboarding/cta/record-agreement",
                    if accepting {
                        msg::onboarding_cta_record_agreement_pending(cx)
                    } else {
                        msg::onboarding_cta_record_agreement(cx)
                    },
                    Pill::Primary,
                    agreed && !accepting,
                    |this, _, cx| this.begin_accept(cx),
                    cx,
                )
                .into_any_element(),
            );
        }
        match &outcome {
            Some(Ok(TermsAcceptance::Complete)) => {
                let line = msg::onboarding_terms_recorded(cx);
                out.push(
                    div()
                        .id("onboarding-terms-recorded")
                        .probe("onboarding/terms-recorded", Role::Label, line.clone())
                        .child(note(line, cx))
                        .into_any_element(),
                );
                // Back to adding credit: in the first run, its page; from a
                // refused checkout, the plans it came from.
                let from_first_run = self.pages.len() > 1;
                out.push(
                    self.cta(
                        "onboarding/cta/continue",
                        msg::onboarding_cta_continue(cx),
                        Pill::Primary,
                        true,
                        move |this, window, cx| {
                            if from_first_run {
                                this.go(Page::Purchase, window, cx)
                            } else {
                                this.finish(cx)
                            }
                        },
                        cx,
                    )
                    .into_any_element(),
                );
            }
            Some(Ok(standing)) => {
                if let Some(line) = terms_standing(standing, cx) {
                    out.push(error_note("onboarding/terms-standing", line, cx).into_any_element());
                }
            }
            Some(Err(e)) => out.push(
                error_note(
                    "onboarding/error/accept",
                    msg::onboarding_accept_failed(cx, e.to_string()),
                    cx,
                )
                .into_any_element(),
            ),
            None => {}
        }
        out
    }
}

/// What a standing short of complete means for the reader: `None` when
/// everything required is accepted.
fn terms_standing(standing: &TermsAcceptance, cx: &App) -> Option<SharedString> {
    match standing {
        TermsAcceptance::Complete => None,
        TermsAcceptance::Outstanding { documents } => {
            let names: Vec<String> = documents
                .iter()
                .map(|d| match d.as_str() {
                    "terms_of_service" => msg::onboarding_link_terms_of_service(cx).to_string(),
                    "privacy_policy" => msg::onboarding_link_privacy_policy(cx).to_string(),
                    other => other.replace('_', " "),
                })
                .collect();
            Some(msg::onboarding_terms_outstanding(cx, names.join(", ")))
        }
        TermsAcceptance::Unknown { error } => {
            Some(msg::onboarding_terms_unknown(cx, error.to_string()))
        }
    }
}

impl Render for OnboardingView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let text = self.body_text(cx);
        self.prose.update(cx, |p, cx| p.sync(text, cx));
        let back = self.can_go_back().then(|| {
            Box::new(cx.listener(|this, _, window, cx| this.back(window, cx)))
                as super::widgets::OnClick
        });
        let mut extras = self.render_extras(cx);
        // The narrative pages and the choice keep their verbs at the foot of
        // the screen, under the thumb, while the words scroll above them.
        let footer: Vec<AnyElement> = match self.page() {
            Page::About | Page::Responsibility => std::mem::take(&mut extras),
            Page::GetStarted => extras.split_off(1.min(extras.len())),
            _ => Vec::new(),
        };
        let border = cx.theme().border;
        div()
            .size_full()
            .flex()
            .flex_col()
            .child(header(
                "onboarding",
                msg::onboarding_title(cx),
                back,
                None,
                cx,
            ))
            .child(
                div()
                    .id("onboarding-body")
                    .flex_1()
                    .min_h_0()
                    .overflow_y_scroll()
                    .track_scroll(&self.scroll)
                    .pt(px(8.))
                    .pb(px(40.))
                    .child(self.prose.clone())
                    .child(div().flex().flex_col().gap_1().pt(px(8.)).children(extras)),
            )
            .when(!footer.is_empty(), |d| {
                d.child(
                    div()
                        .flex_none()
                        .flex()
                        .flex_col()
                        .gap_1()
                        .pt(px(8.))
                        .pb(px(10.))
                        .border_t_1()
                        .border_color(border)
                        .children(footer),
                )
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(document: &str, sha256: &str, version: i64) -> TermsDocument {
        TermsDocument {
            document: document.into(),
            version,
            url: format!("https://www.eidola.ai/{document}/"),
            sha256: sha256.into(),
        }
    }

    /// The agreement's key is the documents' texts, as a set: a reordered
    /// listing of the same texts is the same agreement; a new text, a moved
    /// version of one, or an added document is not.
    #[test]
    fn consent_binds_to_the_texts_rendered() {
        let shown = vec![
            doc("terms_of_service", "aa", 1),
            doc("privacy_policy", "bb", 1),
        ];
        let agreed = consent_fingerprint(&shown);
        let reordered = vec![
            doc("privacy_policy", "bb", 1),
            doc("terms_of_service", "aa", 1),
        ];
        assert_eq!(agreed, consent_fingerprint(&reordered));
        let new_text = vec![
            doc("terms_of_service", "cc", 2),
            doc("privacy_policy", "bb", 1),
        ];
        assert_ne!(agreed, consent_fingerprint(&new_text));
        let mut added = shown.clone();
        added.push(doc("acceptable_use", "dd", 1));
        assert_ne!(agreed, consent_fingerprint(&added));
        assert_ne!(consent_fingerprint(&[]), agreed);
    }
}
