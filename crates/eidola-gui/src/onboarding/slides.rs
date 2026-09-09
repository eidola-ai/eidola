//! The onboarding slides — one self-contained component per slide.
//!
//! gpui's analogue of a React functional component is a props struct that
//! derives [`IntoElement`] and implements [`RenderOnce`]: the parent
//! constructs it each frame with plain data plus boxed callbacks (the
//! "props"), and `render` consumes it into elements. State lives lifted into
//! [`super::OnboardingView`]; the slides here are stateless presentation.
//!
//! Everything one slide shows — its prose markdown, its extra content
//! (links, inputs, credentials, plans), and its CTAs — lives on that slide's
//! component, so reading a slide top-to-bottom is one struct + one `render`.
//! The parent's `render_slide` is the router: it matches on [`Slide`] once,
//! wiring each component's callbacks back into the view (flow sequencing —
//! which slide a CTA reveals next — deliberately stays with the parent, so
//! the branch structure is visible in one place).

use gpui::{
    AnyElement, App, ClickEvent, ClipboardItem, Entity, InteractiveElement, IntoElement,
    ParentElement, Pixels, RenderOnce, Role, SharedString, StatefulInteractiveElement, Styled,
    Window, div, prelude::FluentBuilder, px,
};
use gpui_component::{
    ActiveTheme, Disableable, Icon, IconName, Sizable,
    button::{Button, ButtonVariants},
    checkbox::Checkbox,
    h_flex,
    input::{Input, InputState},
    v_flex,
};
use gpui_markdown_editor::{MarkdownEditor, MarkdownEditorState};

use eidola_app_core::error::AppError;
use eidola_app_core::{PriceInfo, TermsDocument};

use super::{CheckoutFailure, Slide, VerifyFailure};
use crate::plans::{self, format_credits};
use crate::probe::Probe as _;
use crate::space_view::{TITLE_BAR_RESERVE, prose_style};

/// External links referenced by the slides.
const REPO_URL: &str = "https://github.com/eidola-ai/eidola";
const TERMS_URL: &str = "https://www.eidola.ai/terms/";
const PRIVACY_URL: &str = "https://www.eidola.ai/privacy/";
/// The link inside the "Get started" prose. It lives here rather than in the
/// FTL body so a translation cannot change where the reader is sent for the
/// evidence behind an unlinkability claim.
const UNLINKABILITY_URL: &str = "https://www.eidola.ai/docs/privacy-guarantees/#2-unlinkability";

/// The reading column width for slide content (left-aligned prose, like a post).
const COLUMN_WIDTH: Pixels = px(560.);

/// A boxed click-callback prop.
pub(super) type OnClick = Box<dyn Fn(&ClickEvent, &mut Window, &mut App) + 'static>;
/// A boxed checkbox-toggle callback prop.
pub(super) type OnToggle = Box<dyn Fn(&bool, &mut Window, &mut App) + 'static>;

// -- Pause -----------------------------------------------------------------

/// "Pause here" — Eidola is not the same as the hosted assistants.
#[derive(IntoElement)]
pub(super) struct Pause {
    pub prose: Entity<MarkdownEditorState>,
    pub on_advance: OnClick,
}

impl RenderOnce for Pause {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let cta = cta_button(
            "pause",
            crate::i18n::msg::onboarding_cta_pause(cx),
            self.on_advance,
        )
        .into_any_element();
        slide_frame(&self.prose, None, cta, window, cx)
    }
}

// -- Tool ------------------------------------------------------------------

/// "Eidola is your tool" — the CD-era sovereignty analogy.
#[derive(IntoElement)]
pub(super) struct Tool {
    pub prose: Entity<MarkdownEditorState>,
    pub on_advance: OnClick,
}

impl RenderOnce for Tool {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let cta = cta_button(
            "tool",
            crate::i18n::msg::onboarding_cta_understood(cx),
            self.on_advance,
        )
        .into_any_element();
        slide_frame(&self.prose, None, cta, window, cx)
    }
}

// -- Control ---------------------------------------------------------------

/// "Your control" — no operator can read, retain, or change it. Links to the repo.
#[derive(IntoElement)]
pub(super) struct Control {
    pub prose: Entity<MarkdownEditorState>,
    pub on_advance: OnClick,
}

impl RenderOnce for Control {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let extras = link_row(
            "the-eidola-code-repository",
            crate::i18n::msg::onboarding_link_repository(cx),
            REPO_URL,
            cx,
        )
        .into_any_element();
        let cta = cta_button(
            "control",
            crate::i18n::msg::onboarding_cta_understood(cx),
            self.on_advance,
        )
        .into_any_element();
        slide_frame(&self.prose, Some(extras), cta, window, cx)
    }
}

// -- Responsibility ----------------------------------------------------------

/// "Your responsibility" — models are fallible; effects are yours.
#[derive(IntoElement)]
pub(super) struct Responsibility {
    pub prose: Entity<MarkdownEditorState>,
    pub on_advance: OnClick,
}

impl RenderOnce for Responsibility {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let cta = cta_button(
            "responsibility",
            crate::i18n::msg::onboarding_cta_understood(cx),
            self.on_advance,
        )
        .into_any_element();
        slide_frame(&self.prose, None, cta, window, cx)
    }
}

// -- GetStarted --------------------------------------------------------------

/// "Get started" — the branch point (new vs. existing account), plus the
/// quiet third way: no account at all (on-device inference only).
#[derive(IntoElement)]
pub(super) struct GetStarted {
    pub prose: Entity<MarkdownEditorState>,
    pub on_new_account: OnClick,
    pub on_existing_account: OnClick,
    pub on_skip_account: OnClick,
}

impl RenderOnce for GetStarted {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let fg = cx.theme().muted_foreground;
        let fg_hover = cx.theme().foreground;
        // The unlinkability link's target is the app's, not the translation's:
        // a URL living inside prose is a thing a translator can retarget, and
        // this one is a privacy claim's evidence.
        // One sentence, said once. The visible text and the accessible name
        // used to differ by a full stop, which is the failure the a11y-label
        // rule names — invisible in English, audible everywhere else — and it
        // was accidental rather than a deliberate spelling of the subject.
        let skip = crate::i18n::msg::onboarding_cta_skip_account(cx);
        let ctas = v_flex()
            .items_center()
            .gap_3()
            .child(cta_button(
                "new-account",
                crate::i18n::msg::onboarding_cta_new_account(cx),
                self.on_new_account,
            ))
            .child(cta_button(
                "existing-account",
                crate::i18n::msg::onboarding_cta_existing_account(cx),
                self.on_existing_account,
            ))
            // The account-free path: quiet by design — a real choice,
            // not a promoted one. It disables the Eidola backend, so
            // asks route only to on-device (and self-configured)
            // backends and onboarding stops auto-opening.
            .child(
                div()
                    .id("onboarding-skip-account")
                    .probe("onboarding/cta/skip-account", Role::Button, skip.clone())
                    .mt_2()
                    .cursor_pointer()
                    .text_sm()
                    .text_color(fg)
                    .hover(move |s| s.text_color(fg_hover))
                    .child(skip)
                    .on_click(self.on_skip_account),
            )
            .into_any_element();
        slide_frame(&self.prose, None, ctas, window, cx)
    }
}

// -- CreateAccount -----------------------------------------------------------

/// New-account branch: agree to terms, create an anonymous account. The
/// create button stays disabled until the agreement checkbox is checked — an
/// explicit, required consent step separate from the action.
///
/// **What this slide links to is what account creation submits.** The
/// documents come from `AccountStore::terms` (the server's current snapshot)
/// and travel back into `AppCore::account_create` unchanged, so agreement is
/// recorded for the versions named here or not at all. Until that snapshot
/// has arrived there is nothing to agree to, and the CTA stays disabled.
#[derive(IntoElement)]
pub(super) struct CreateAccount {
    pub prose: Entity<MarkdownEditorState>,
    /// The documents whose acceptance creating an account will record, or
    /// `None` while the snapshot has not arrived. An empty list is a loaded
    /// answer — a server running no acceptance gate — and the published
    /// policies are linked instead.
    pub documents: Option<Vec<TermsDocument>>,
    /// Set while the snapshot is being fetched (the initial load only).
    pub loading_documents: bool,
    /// Why the snapshot could not be read, if it could not be — the typed
    /// error, so the line is worded where it is drawn.
    pub documents_error: Option<AppError>,
    /// Whether the agreement box reads as checked — derived by the parent
    /// from whether the reader's consent still covers `documents`, never a
    /// stored flag. False therefore also covers "agreed to an earlier
    /// snapshot", which is why it alone gates the CTA.
    pub agreed: bool,
    /// Whether an account-create request is in flight.
    pub creating: bool,
    /// Why creation was refused, typed for the same reason.
    pub error: Option<AppError>,
    pub on_toggle_agree: OnToggle,
    pub on_create: OnClick,
    pub on_retry_documents: OnClick,
}

impl RenderOnce for CreateAccount {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let theme = cx.theme();
        let have_documents = self.documents.is_some();
        let mut extras = v_flex().gap_3();
        match self.documents.as_deref() {
            // A server with no acceptance gate: nothing to record, but the
            // published policies still govern, so they are still linked.
            Some([]) => {
                extras = extras
                    .child(link_row(
                        "terms-of-service",
                        crate::i18n::msg::onboarding_link_terms_of_service(cx),
                        TERMS_URL,
                        cx,
                    ))
                    .child(link_row(
                        "privacy-policy",
                        crate::i18n::msg::onboarding_link_privacy_policy(cx),
                        PRIVACY_URL,
                        cx,
                    ));
            }
            Some(docs) => {
                for doc in docs {
                    let (slug, label) = document_link(doc, cx);
                    extras = extras.child(link_row(&slug, label, doc.url.clone(), cx));
                }
            }
            None if self.loading_documents => {
                let loading = crate::i18n::msg::onboarding_terms_loading(cx);
                extras = extras.child(
                    div()
                        .id("onboarding-terms-loading")
                        .probe("onboarding/terms/loading", Role::Label, loading.clone())
                        .text_color(theme.muted_foreground)
                        .child(loading),
                );
            }
            None => {}
        }
        if let Some(err) = self.documents_error {
            let retry = crate::i18n::msg::onboarding_terms_retry(cx);
            extras = extras
                .child(error_line("terms", err.to_string(), cx))
                .child(
                    div()
                        .id("onboarding-terms-retry")
                        .probe("onboarding/terms/retry", Role::Button, retry.clone())
                        .cursor_pointer()
                        .text_color(theme.link)
                        .hover(|s| s.underline())
                        .child(retry)
                        .on_click(self.on_retry_documents),
                );
        }
        // The sentence a reader is asked to affirm, read once and spent twice:
        // it is the checkbox's visible label *and* its accessible name, so the
        // two can never say different things.
        let consent = crate::i18n::msg::onboarding_consent_agree(cx);
        let extras = extras
            .child(
                // Required consent — the "Create a new account." button
                // stays disabled until this is checked. **Inert until the
                // documents are on screen**: consent binds to a particular
                // snapshot (`OnboardingView::set_agreement`), so a click with
                // nothing loaded would have no text to be agreement to, and a
                // box that ticked anyway would be claiming one.
                div()
                    .id("onboarding-agree")
                    .pt_10()
                    .probe("onboarding/agree", Role::CheckBox, consent.clone())
                    // A checkbox's state is `toggled`, not `selected` — the
                    // macOS adapter reads `accessibilityValue` off `toggled()`
                    // and consults `is_selected()` only for `Role::Tab`.
                    .aria_toggled(self.agreed.into())
                    .map(|d| {
                        if have_documents {
                            let toggle = self.on_toggle_agree;
                            let next = !self.agreed;
                            d.on_click(move |_, window, cx| toggle(&next, window, cx))
                        } else {
                            d.tab_stop(false)
                        }
                    })
                    .child(
                        Checkbox::new("onboarding-agree-box")
                            .role(None)
                            .label(consent)
                            .checked(self.agreed)
                            .disabled(!have_documents)
                            .tab_stop(false)
                            .p_1(),
                    ),
            )
            .when_some(self.error, |el, err| {
                el.child(error_line("create", err.to_string(), cx))
            });

        let label = if self.creating {
            crate::i18n::msg::onboarding_cta_create_pending(cx)
        } else {
            crate::i18n::msg::onboarding_cta_create(cx)
        };
        // `agreed` is derived from the consent binding, not a stored flag:
        // it is true only while the reader's agreement covers the snapshot
        // being rendered, so "no documents" and "documents the agreement no
        // longer covers" are the same disabled state, not two conditions.
        let enabled = self.agreed && !self.creating;
        let cta = div()
            .id("onboarding-cta-create")
            .probe("onboarding/cta/create", Role::Button, label.clone())
            // One predicate decides both the tab stop and the activation, so a
            // wrapper focused when the CTA disables itself cannot re-invoke on
            // a second Enter (`Button` stops propagation on mouse-down while
            // disabled, so the wrapper never arms a click either).
            .map(|d| {
                if enabled {
                    d.on_click(self.on_create)
                } else {
                    d.tab_stop(false)
                }
            })
            .child(
                Button::new("onboarding-btn-create")
                    .role(None)
                    .ghost()
                    .label(label)
                    .disabled(!enabled)
                    .tab_stop(false),
            );

        slide_frame(
            &self.prose,
            Some(extras.into_any_element()),
            cta.into_any_element(),
            window,
            cx,
        )
    }
}

// -- NewAccount --------------------------------------------------------------

/// New-account branch: the freshly-minted id + secret to save.
#[derive(IntoElement)]
pub(super) struct NewAccount {
    pub prose: Entity<MarkdownEditorState>,
    pub id: SharedString,
    pub secret: SharedString,
    pub on_saved: OnClick,
}

impl RenderOnce for NewAccount {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let extras = v_flex()
            .gap_4()
            .child(credential_row(
                "account-id",
                crate::i18n::msg::onboarding_account_id(cx),
                self.id,
                cx,
            ))
            .child(credential_row(
                "account-secret",
                crate::i18n::msg::onboarding_account_secret(cx),
                self.secret,
                cx,
            ))
            .child(
                div()
                    .text_sm()
                    .text_color(cx.theme().muted_foreground)
                    .child(crate::i18n::msg::onboarding_new_account_note(cx)),
            );

        let cta = cta_button(
            "saved",
            crate::i18n::msg::onboarding_cta_saved(cx),
            self.on_saved,
        )
        .into_any_element();
        slide_frame(
            &self.prose,
            Some(extras.into_any_element()),
            cta,
            window,
            cx,
        )
    }
}

// -- ExistingAccount -----------------------------------------------------------

/// Existing-account branch: enter id + secret, check the balance. Once the
/// account verifies, the verify CTA is replaced by purchase/done choices.
#[derive(IntoElement)]
pub(super) struct ExistingAccount {
    pub prose: Entity<MarkdownEditorState>,
    pub id_input: Entity<InputState>,
    pub secret_input: Entity<InputState>,
    /// Whether a verification request is in flight.
    pub verifying: bool,
    /// `Ok(available_credits)` once verified, or why it could not be — the
    /// typed failure, worded at render.
    pub verify_result: Option<Result<i64, VerifyFailure>>,
    pub on_verify: OnClick,
    pub on_purchase: OnClick,
    pub on_done: OnClick,
}

impl RenderOnce for ExistingAccount {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let theme = cx.theme();

        let mut extras = v_flex()
            .gap_3()
            .child(labeled_input(
                crate::i18n::msg::onboarding_account_id(cx),
                "onboarding/input/account-id",
                &self.id_input,
            ))
            .child(labeled_input(
                crate::i18n::msg::onboarding_account_secret(cx),
                "onboarding/input/account-secret",
                &self.secret_input,
            ));
        extras = match &self.verify_result {
            // The balance is one sentence with the noun inside it rather than
            // a number with "credits" appended: "1 credits" is a bug in
            // English and the agreement rule differs in every other language.
            // `$credits` is grouped for reading; the raw count rides beside it
            // only so the select can choose.
            Some(Ok(available)) => extras.child(div().text_color(theme.foreground).child(
                crate::i18n::msg::onboarding_verified_balance(
                    cx,
                    *available,
                    format_credits(*available),
                ),
            )),
            Some(Err(failure)) => extras.child(error_line(
                "verify",
                super::verify_error_copy(failure, cx).to_string(),
                cx,
            )),
            None => extras,
        };

        let ctas = match &self.verify_result {
            Some(Ok(_)) => v_flex()
                .items_center()
                .gap_3()
                .child(cta_button(
                    "existing-purchase",
                    crate::i18n::msg::onboarding_cta_existing_purchase(cx),
                    self.on_purchase,
                ))
                .child(cta_button(
                    "existing-done",
                    crate::i18n::msg::onboarding_cta_existing_done(cx),
                    self.on_done,
                ))
                .into_any_element(),
            _ => cta_button(
                "verify",
                if self.verifying {
                    crate::i18n::msg::onboarding_cta_verify_pending(cx)
                } else {
                    crate::i18n::msg::onboarding_cta_verify(cx)
                },
                self.on_verify,
            )
            .into_any_element(),
        };

        slide_frame(
            &self.prose,
            Some(extras.into_any_element()),
            ctas,
            window,
            cx,
        )
    }
}

// -- Purchase ----------------------------------------------------------------

/// Either branch: choose a plan / add credit via Stripe checkout.
#[derive(IntoElement)]
pub(super) struct Purchase {
    pub prose: Entity<MarkdownEditorState>,
    /// The plans this account may actually buy — already narrowed to
    /// one-time top-ups when `subscribed`.
    pub prices: Vec<PriceInfo>,
    /// Whether a subscription is already in force. Changes what the empty
    /// and populated states say, and points at where it is managed.
    pub subscribed: bool,
    /// Whether the price list is still loading (empty-state copy).
    pub loading: bool,
    /// The plan whose checkout request is currently in flight, if any.
    pub checkout_pending: Option<String>,
    /// Why the last checkout link was not opened, typed for the same reason as
    /// every other slot here.
    pub checkout_error: Option<CheckoutFailure>,
    pub on_select: plans::PlanSelectHandler,
    pub on_later: OnClick,
}

impl RenderOnce for Purchase {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let theme = cx.theme();

        // With a subscription already in force the server refuses a second,
        // so the recurring plans are gone from `prices` and this says why —
        // and where the existing one is managed.
        let subscribed_note = self.subscribed.then(|| {
            div()
                .text_sm()
                .text_color(theme.muted_foreground)
                .child(crate::i18n::msg::onboarding_purchase_subscribed_note(cx))
                .into_any_element()
        });

        let extras = if self.prices.is_empty() {
            v_flex()
                .gap_2()
                .when_some(subscribed_note, |el, note| el.child(note))
                .child(
                    div()
                        .text_color(theme.muted_foreground)
                        .child(if self.loading {
                            crate::i18n::msg::onboarding_purchase_loading(cx)
                        } else if self.subscribed {
                            crate::i18n::msg::onboarding_purchase_none_topups(cx)
                        } else {
                            crate::i18n::msg::onboarding_purchase_none_plans(cx)
                        }),
                )
                .into_any_element()
        } else {
            v_flex()
                .gap_2()
                .when_some(subscribed_note, |el, note| el.child(note))
                .child(
                    div()
                        .text_sm()
                        .text_color(theme.muted_foreground)
                        .child(crate::i18n::msg::onboarding_purchase_checkout_note(cx)),
                )
                // The whole slide around these rows is localized, so the rows
                // are too — a shared component says nothing on its own.
                .child(plans::plan_rows(
                    &self.prices,
                    self.checkout_pending.as_deref(),
                    self.on_select,
                    "onboarding",
                    plans::PlanLabels::localized(),
                    cx,
                ))
                .when_some(self.checkout_error, |el, err| {
                    el.child(error_line(
                        "checkout",
                        super::checkout_error_copy(&err, cx).to_string(),
                        cx,
                    ))
                })
                .into_any_element()
        };

        let cta = cta_button(
            "purchase-later",
            crate::i18n::msg::onboarding_cta_purchase_later(cx),
            self.on_later,
        )
        .into_any_element();
        slide_frame(&self.prose, Some(extras), cta, window, cx)
    }
}

/// The prose a slide shows, in the reader's active locale.
///
/// **One function, so the render and any test agree about which message a slide
/// carries.** It also keeps the one argument-bearing body honest: the
/// unlinkability link's target is supplied here rather than written into the
/// translation, so no locale can send a reader somewhere else for the evidence
/// behind a privacy claim.
pub(super) fn slide_body(slide: Slide, cx: &App) -> SharedString {
    match slide {
        Slide::Pause => crate::i18n::msg::onboarding_pause_body(cx),
        Slide::Tool => crate::i18n::msg::onboarding_tool_body(cx),
        Slide::Control => crate::i18n::msg::onboarding_control_body(cx),
        Slide::Responsibility => crate::i18n::msg::onboarding_responsibility_body(cx),
        Slide::GetStarted => crate::i18n::msg::onboarding_get_started_body(cx, UNLINKABILITY_URL),
        Slide::CreateAccount => crate::i18n::msg::onboarding_create_account_body(cx),
        Slide::NewAccount => crate::i18n::msg::onboarding_new_account_body(cx),
        Slide::ExistingAccount => crate::i18n::msg::onboarding_existing_account_body(cx),
        Slide::Purchase => crate::i18n::msg::onboarding_purchase_body(cx),
    }
}

// -- Shared layout + primitives ------------------------------------------------

/// Breathing room below a slide's content block (and, symmetrically-ish, the
/// gap the top title-bar reserve leaves above it) so a short slide's centered
/// block never crowds the window edges.
const SLIDE_BOTTOM_PAD: Pixels = px(56.);

/// The shared slide layout: a left-aligned reading column (prose + any
/// `extras`) with its call-to-action group in **normal flow beneath it**, the
/// whole block centered — horizontally always, and vertically when the slide is
/// shorter than the window.
///
/// The slide's height is its **content**, floored at one window
/// (`min_h(content_size)`), never fixed to the window: a short slide reads as a
/// full page and centers, while a long one grows past the window and the outer
/// page simply scrolls to it — so content is never clipped and the CTAs, being
/// after the prose in flow, can never overlap it. (The old layout fixed each
/// slide to exactly the window height and vertically-centered an overflowing
/// prose region, which spilled onto the CTAs and — under mandatory whole-window
/// snapping — left the spill unreachable.)
fn slide_frame(
    prose_state: &Entity<MarkdownEditorState>,
    extras: Option<AnyElement>,
    ctas: AnyElement,
    window: &mut Window,
    cx: &mut App,
) -> AnyElement {
    // **The prose editor is the view's, not the element's.** It used to be
    // element-owned state (`use_keyed_state`, keyed per slide, initialized
    // once and evicted by the framework a frame after the slide stopped
    // rendering), which was right while the body was a `const`. It is not
    // right for a localized one: `use_keyed_state` seeds on the first frame
    // and `i18n::apply` replaces no state — it only refreshes windows — so
    // every slide would go on reading in the language the window opened in.
    // The rule this module already stated covers it: state the view must
    // *write* is lifted. `OnboardingView::sync_prose` owns the minting, the
    // push and the pruning; this only renders what it is handed.
    let prose = MarkdownEditor::new(prose_state)
        .style(prose_style(cx))
        .disabled(true)
        .into_any_element();

    // The narrative reading column: prose then extras, left-aligned, capped to
    // a comfortable measure and centered as a unit; the CTA group sits below it
    // in flow, centered on its own axis.
    let content = v_flex()
        .w_full()
        .items_center()
        .gap_8()
        .child(
            v_flex()
                .w(COLUMN_WIDTH)
                .max_w_full()
                .gap_4()
                .child(prose)
                .child(extras.unwrap_or_else(|| div().into_any_element())),
        )
        .child(v_flex().items_center().gap_3().child(ctas));

    v_flex()
        .w_full()
        .flex_none()
        // At least one screenful — the content box, not the raw surface: on
        // Linux CSD `viewport_size` includes the shadow padding. A short slide
        // fills exactly this and centers; a tall one grows past it (min, not
        // fixed) and the page scrolls.
        .min_h(crate::chrome::content_size(window).height)
        // Clear the titlebar drag band when this slide sits at the viewport top,
        // and leave matching room at the bottom.
        .pt(TITLE_BAR_RESERVE)
        .pb(SLIDE_BOTTOM_PAD)
        .px_8()
        // Center the content block vertically while there is slack; once the
        // content exceeds the min height the block simply flows from the top.
        .justify_center()
        .child(content)
        .into_any_element()
}

/// A ghost-button CTA, wrapped in a probed div for accessibility + the driver.
fn cta_button(
    key: &'static str,
    label: impl Into<SharedString>,
    on_click: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
) -> impl IntoElement {
    let label = label.into();
    div()
        .id(SharedString::from(format!("onboarding-cta-{key}")))
        .probe(
            SharedString::from(format!("onboarding/cta/{key}")),
            Role::Button,
            label.clone(),
        )
        .on_click(on_click)
        .child(
            Button::new(SharedString::from(format!("onboarding-btn-{key}")))
                .role(None)
                .ghost()
                .label(label)
                .tab_stop(false),
        )
}

/// The vertical inset of the back affordance from a slide's top — clear of the
/// titlebar drag band (which owns the top [`TITLE_BAR_RESERVE`] and paints last)
/// so the window-move gesture keeps the very top edge.
const BACK_BUTTON_TOP: Pixels = px(46.);

/// An up-chevron "back" affordance pinned near the top of a slide, shown on
/// every slide after the first. It's a *visible* alternative to the scroll-back
/// gesture — clicking it glides to the previous slide — since the gesture is a
/// less obvious way to go back for many people. `key` scopes the a11y/driver
/// name so the per-slide buttons (all painted at once) don't collide.
pub(super) fn back_button(
    key: impl std::fmt::Display,
    on_back: OnClick,
    cx: &App,
) -> impl IntoElement {
    let theme = cx.theme();
    let fg = theme.muted_foreground;
    let fg_hover = theme.foreground;
    let hover_bg = theme.muted.opacity(0.5);
    div()
        .absolute()
        .top(BACK_BUTTON_TOP)
        .left_0()
        .right_0()
        .flex()
        .justify_center()
        .child(
            div()
                .id(SharedString::from(format!("onboarding-back-{key}")))
                .probe(
                    SharedString::from(format!("onboarding/back/{key}")),
                    Role::Button,
                    crate::i18n::msg::onboarding_back(cx),
                )
                .flex()
                .items_center()
                .justify_center()
                .size(px(28.))
                .rounded_full()
                .cursor_pointer()
                .text_color(fg)
                .hover(move |s| s.text_color(fg_hover).bg(hover_bg))
                .child(Icon::new(IconName::ChevronUp).small())
                .on_click(on_back),
        )
}

/// A standalone clickable external link line ("Label ↗"). `slug` names the
/// probe and is **stable by contract** — it never carries anything that moves
/// (a document version, a fetched title), because a probe name is a selector.
fn link_row(
    slug: &str,
    label: impl Into<SharedString>,
    url: impl Into<SharedString>,
    cx: &App,
) -> impl IntoElement {
    let theme = cx.theme();
    let label: SharedString = label.into();
    let url: SharedString = url.into();
    div()
        .id(SharedString::from(format!("onboarding-link-{slug}")))
        .probe(
            SharedString::from(format!("onboarding/link/{slug}")),
            Role::Link,
            label.clone(),
        )
        .w_full()
        .cursor_pointer()
        .text_color(theme.link)
        .hover(|s| s.underline())
        .child(crate::i18n::msg::onboarding_link_external(
            cx,
            label.to_string(),
        ))
        .on_click(move |_, _, cx| cx.open_url(url.as_ref()))
}

/// Slug and display label for a required document. The **slug is derived from
/// the document key**, which is the wire identifier and does not move; the
/// label carries the version, which does.
///
/// The name is the document's **published title** and stays English in every
/// locale — it is the text acceptance is recorded against and the heading the
/// reader will find at the other end of the link. Only the wrapper around it
/// localizes, and it does so as one message rather than a name with a version
/// appended, so a locale can order the two as its own grammar wants.
fn document_link(doc: &TermsDocument, cx: &App) -> (String, SharedString) {
    let slug: String = doc
        .document
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    let name = match doc.document.as_str() {
        "terms_of_service" => crate::i18n::msg::onboarding_link_terms_of_service(cx),
        "privacy_policy" => crate::i18n::msg::onboarding_link_privacy_policy(cx),
        other => SharedString::from(other.replace('_', " ")),
    };
    let label = crate::i18n::msg::onboarding_document_version(cx, name.to_string(), doc.version);
    (slug, label)
}

/// A labeled single-line credential value in mono, with a Copy affordance.
///
/// `key` is the row's **stable** identity — its probe name and its element id
/// both derive from it. They used to derive from the *label*, which localizing
/// would have translated: a probe name is a selector and an element id keys
/// per-element state and the row's accessibility node, so either moving with
/// the reader's language is a defect the moment the label does.
fn credential_row(
    key: &'static str,
    label: SharedString,
    value: SharedString,
    cx: &App,
) -> impl IntoElement {
    let theme = cx.theme();
    let to_copy = value.clone();
    v_flex()
        .gap_1()
        .child(
            div()
                .text_sm()
                .text_color(theme.muted_foreground)
                .child(label.clone()),
        )
        .child(
            h_flex()
                .w_full()
                .items_center()
                .justify_between()
                .gap_2()
                .px_3()
                .py_2()
                .rounded_md()
                .bg(theme.muted.opacity(0.5))
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .font_family("Menlo")
                        .text_sm()
                        .child(value.clone()),
                )
                .child(
                    div()
                        .id(SharedString::from(format!("onboarding-copy-{key}")))
                        // The verb is repeated on both rows, so its accessible
                        // name takes the subject its row supplies — the
                        // `ghost_button_labeled` rule.
                        .probe(
                            SharedString::from(format!("onboarding/copy/{key}")),
                            Role::Button,
                            crate::i18n::msg::onboarding_copy_label(cx, label.to_string()),
                        )
                        .flex_none()
                        .cursor_pointer()
                        .text_sm()
                        .text_color(theme.muted_foreground)
                        .hover(|s| s.text_color(theme.foreground))
                        .child(crate::i18n::msg::onboarding_copy(cx))
                        .on_click(move |_, _, cx| {
                            cx.write_to_clipboard(ClipboardItem::new_string(to_copy.to_string()))
                        }),
                ),
        )
}

/// A labeled text input (existing-account credentials), probed for a11y.
///
/// The element id derives from the **probe name**, which is already required to
/// be unique per painted element and does not move with the reader's language —
/// deriving it from the label (as it did) would remint the field's per-element
/// state and its accessibility node on a locale change.
fn labeled_input(
    label: SharedString,
    probe_name: &'static str,
    state: &Entity<InputState>,
) -> impl IntoElement {
    v_flex()
        .gap_1()
        .child(div().text_sm().child(label.clone()))
        .child(
            div()
                .id(SharedString::from(format!("{probe_name}-wrap")))
                .probe_bounds(probe_name, Role::TextInput, label.clone())
                .w_full()
                .child(Input::new(state).aria_label(label)),
        )
}

/// A danger-colored inline error line. `key` scopes the a11y/driver name so
/// concurrently-revealed slides (create / verify / checkout) don't collide;
/// the `Alert` role makes assistive technology announce the failure.
fn error_line(key: &'static str, message: String, cx: &App) -> impl IntoElement {
    let theme = cx.theme();
    div()
        .id(SharedString::from(format!("onboarding-error-{key}")))
        .probe(
            SharedString::from(format!("onboarding/error/{key}")),
            Role::Alert,
            SharedString::from(message.clone()),
        )
        .text_sm()
        .text_color(theme.danger)
        .child(SharedString::from(message))
}
