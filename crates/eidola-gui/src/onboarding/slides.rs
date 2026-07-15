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

use eidola_app_core::PriceInfo;

use crate::plans::{self, format_credits};
use crate::probe::Probe as _;
use crate::space_view::{TITLE_BAR_RESERVE, prose_style};

/// External links referenced by the slides.
const REPO_URL: &str = "https://github.com/eidola-ai/eidola";
const TERMS_URL: &str = "https://www.eidola.ai/terms/";
const PRIVACY_URL: &str = "https://www.eidola.ai/privacy/";

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
    pub on_advance: OnClick,
}

impl Pause {
    const MARKDOWN: &'static str =
        "## *Pause here*\n\nEidola is *not* the same as ChatGPT, Claude or Gemini.";
}

impl RenderOnce for Pause {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        slide_frame(
            "pause",
            Self::MARKDOWN,
            None,
            cta_button("pause", "OK, you have my attention.", self.on_advance).into_any_element(),
            window,
            cx,
        )
    }
}

// -- Tool ------------------------------------------------------------------

/// "Eidola is your tool" — the CD-era sovereignty analogy.
#[derive(IntoElement)]
pub(super) struct Tool {
    pub on_advance: OnClick,
}

impl Tool {
    const MARKDOWN: &'static str = "## Eidola is *your* tool\n\nIn years past, an application was delivered to your \
         computer via a CD:\n\n- Its behavior *couldn't* spontaneously change without your \
         involvement.\n- Your files, plans, usage patterns, and insights were *yours alone*, \
         undiscoverable by any third party.\n- The **structure** of the technology — *not* some \
         company's promises — enforced these properties.\n\nEidola approximates this \
         approach as closely as possible, structurally maximizing end-user sovereignty even \
         for workloads that are best run in a data center.";
}

impl RenderOnce for Tool {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        slide_frame(
            "tool",
            Self::MARKDOWN,
            None,
            cta_button("tool", "I understand.", self.on_advance).into_any_element(),
            window,
            cx,
        )
    }
}

// -- Control ---------------------------------------------------------------

/// "Your control" — no operator can read, retain, or change it. Links to the repo.
#[derive(IntoElement)]
pub(super) struct Control {
    pub on_advance: OnClick,
}

impl Control {
    const MARKDOWN: &'static str = "## *Your* control\n\nYou and only you are in control — not us, not the operators who run the \
         hardware.:\n\n- **Only you can read, retain, or profile your interactions.** Your data is \
         decrypted only inside sealed, hardware-attested enclaves that keep nothing, and \
         the side of Eidola that handles payment is cryptographically separated from the \
         side that serves your requests.\n- **Only you can update Eidola — on your \
         device the server.** Nothing changes until your \
         client verifies a new version and you decide to trust it.\n\nDon't blindly trust our claims; verify them. If you don't know how \
         to evaluate our code and architecture, **request the opinion of the most technical \
         person you already trust**.";
}

impl RenderOnce for Control {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        slide_frame(
            "control",
            Self::MARKDOWN,
            Some(link_row("The Eidola code repository", REPO_URL, cx).into_any_element()),
            cta_button("control", "I understand.", self.on_advance).into_any_element(),
            window,
            cx,
        )
    }
}

// -- Responsibility ----------------------------------------------------------

/// "Your responsibility" — models are fallible; effects are yours.
#[derive(IntoElement)]
pub(super) struct Responsibility {
    pub on_advance: OnClick,
}

impl Responsibility {
    const MARKDOWN: &'static str = "## *Your* responsibility\n\nEidola is a tool that makes it easier for *you* to run \
         AI models without highly-specialized technical skills or expensive hardware. Nobody \
         is looking over your shoulder, so it's critical that you understand:\n\n- You will \
         be running AI models, which are at their core collections of probabilities. The \
         models that run in Eidola are freely available to download and use with or without \
         Eidola.\n- Even the very best models are fallible. They can be amazingly useful, but \
         do make mistakes and can exhibit unexpected behavior. It is mathematically \
         impossible to evaluate how a large model will behave in every possible scenario.\n- \
         The models themselves have no intrinsic memory or ability to cause external effects; \
         they can only evaluate data and take actions that you make available to them. Eidola \
         makes it easy to understand and configure access, but the results — both good and \
         bad — are ultimately your responsibility.";
}

impl RenderOnce for Responsibility {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        slide_frame(
            "responsibility",
            Self::MARKDOWN,
            None,
            cta_button("responsibility", "I understand.", self.on_advance).into_any_element(),
            window,
            cx,
        )
    }
}

// -- GetStarted --------------------------------------------------------------

/// "Get started" — the branch point (new vs. existing account), plus the
/// quiet third way: no account at all (on-device inference only).
#[derive(IntoElement)]
pub(super) struct GetStarted {
    pub on_new_account: OnClick,
    pub on_existing_account: OnClick,
    pub on_skip_account: OnClick,
}

impl GetStarted {
    const MARKDOWN: &'static str = "## Get started\n\nYou'll need some credits to run models.\n\nYour account is just a \
         random id — buying credit is the only step that touches a payment method, and even \
         then [we are structurally unable to link your requests back to it](https://www.eidola.ai/docs/privacy-guarantees/#2-unlinkability).";
}

impl RenderOnce for GetStarted {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let fg = cx.theme().muted_foreground;
        let fg_hover = cx.theme().foreground;
        slide_frame(
            "get-started",
            Self::MARKDOWN,
            None,
            v_flex()
                .items_center()
                .gap_3()
                .child(cta_button(
                    "new-account",
                    "I need a new account.",
                    self.on_new_account,
                ))
                .child(cta_button(
                    "existing-account",
                    "I already have an account.",
                    self.on_existing_account,
                ))
                // The account-free path: quiet by design — a real choice,
                // not a promoted one. It disables the Eidola backend, so
                // asks route only to on-device (and self-configured)
                // backends and onboarding stops auto-opening.
                .child(
                    div()
                        .id("onboarding-skip-account")
                        .probe(
                            "onboarding/cta/skip-account",
                            Role::Button,
                            "Continue without an account — on-device models only",
                        )
                        .mt_2()
                        .cursor_pointer()
                        .text_sm()
                        .text_color(fg)
                        .hover(move |s| s.text_color(fg_hover))
                        .child("Continue without an account — on-device models only.")
                        .on_click(self.on_skip_account),
                )
                .into_any_element(),
            window,
            cx,
        )
    }
}

// -- CreateAccount -----------------------------------------------------------

/// New-account branch: agree to terms, create an anonymous account. The
/// create button stays disabled until the agreement checkbox is checked — an
/// explicit, required consent step separate from the action.
#[derive(IntoElement)]
pub(super) struct CreateAccount {
    /// Whether the terms/privacy agreement checkbox is checked.
    pub agreed: bool,
    /// Whether an account-create request is in flight.
    pub creating: bool,
    pub error: Option<String>,
    pub on_toggle_agree: OnToggle,
    pub on_create: OnClick,
}

impl CreateAccount {
    const MARKDOWN: &'static str = "## Create an account\n\nPlease read and understand our Terms of Service and Privacy \
         Policy.";
}

impl RenderOnce for CreateAccount {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let extras = v_flex()
            .gap_3()
            .child(link_row("Terms of Service", TERMS_URL, cx))
            .child(link_row("Privacy Policy", PRIVACY_URL, cx))
            .child(
                // Required consent — the "Create a new account." button
                // stays disabled until this is checked.
                div()
                    .id("onboarding-agree")
                    .pt_10()
                    .probe(
                        "onboarding/agree",
                        Role::CheckBox,
                        "I agree to the Terms of Service and Privacy Policy.",
                    )
                    .child(
                        Checkbox::new("onboarding-agree-box")
                            .label("I agree to the Terms of Service and Privacy Policy.")
                            .checked(self.agreed)
                            .on_click(self.on_toggle_agree)
                            .p_1(),
                    ),
            )
            .when_some(self.error, |el, err| {
                el.child(error_line("create", err, cx))
            });

        let label = if self.creating {
            "Creating your anonymous account…"
        } else {
            "Create a new account."
        };
        let enabled = self.agreed && !self.creating;
        let cta = div()
            .id("onboarding-cta-create")
            .probe("onboarding/cta/create", Role::Button, label)
            .child(
                Button::new("onboarding-btn-create")
                    .ghost()
                    .label(label)
                    .disabled(!enabled)
                    .on_click(self.on_create),
            );

        slide_frame(
            "create-account",
            Self::MARKDOWN,
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
    pub id: SharedString,
    pub secret: SharedString,
    pub on_saved: OnClick,
}

impl NewAccount {
    const MARKDOWN: &'static str = "## Your new account\n\nYour new account has been created:";
}

impl RenderOnce for NewAccount {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let extras = v_flex()
            .gap_4()
            .child(credential_row("Account ID", self.id, cx))
            .child(credential_row("Account Secret", self.secret, cx))
            .child(
                div()
                    .text_sm()
                    .text_color(cx.theme().muted_foreground)
                    .child(
                        "This is used only to add and consume credits. If the account \
                         secret is lost, it cannot be recovered, and you will need to \
                         create a new account.",
                    ),
            );

        slide_frame(
            "new-account",
            Self::MARKDOWN,
            Some(extras.into_any_element()),
            cta_button("saved", "I've saved this somewhere.", self.on_saved).into_any_element(),
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
    pub id_input: Entity<InputState>,
    pub secret_input: Entity<InputState>,
    /// Whether a verification request is in flight.
    pub verifying: bool,
    /// `Ok(available_credits)` once verified, or `Err(message)` on failure.
    pub verify_result: Option<Result<i64, String>>,
    pub on_verify: OnClick,
    pub on_purchase: OnClick,
    pub on_done: OnClick,
}

impl ExistingAccount {
    const MARKDOWN: &'static str = "## Your existing account\n\nEnter your account details:";
}

impl RenderOnce for ExistingAccount {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let theme = cx.theme();

        let mut extras = v_flex()
            .gap_3()
            .child(labeled_input(
                "Account ID",
                "onboarding/input/account-id",
                &self.id_input,
            ))
            .child(labeled_input(
                "Account Secret",
                "onboarding/input/account-secret",
                &self.secret_input,
            ));
        extras = match &self.verify_result {
            Some(Ok(available)) => extras.child(div().text_color(theme.foreground).child(
                SharedString::from(format!(
                    "This account is valid and has a balance of {} credits.",
                    format_credits(*available)
                )),
            )),
            Some(Err(msg)) => extras.child(error_line("verify", msg.clone(), cx)),
            None => extras,
        };

        let ctas = match &self.verify_result {
            Some(Ok(_)) => v_flex()
                .items_center()
                .gap_3()
                .child(cta_button(
                    "existing-purchase",
                    "I want to purchase more credits.",
                    self.on_purchase,
                ))
                .child(cta_button(
                    "existing-done",
                    "This looks good.",
                    self.on_done,
                ))
                .into_any_element(),
            _ => cta_button(
                "verify",
                if self.verifying {
                    "Checking…"
                } else {
                    "Check account balance."
                },
                self.on_verify,
            )
            .into_any_element(),
        };

        slide_frame(
            "existing-account",
            Self::MARKDOWN,
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
    pub prices: Vec<PriceInfo>,
    /// Whether the price list is still loading (empty-state copy).
    pub loading: bool,
    /// The plan whose checkout request is currently in flight, if any.
    pub checkout_pending: Option<String>,
    pub checkout_error: Option<String>,
    pub on_select: plans::PlanSelectHandler,
    pub on_later: OnClick,
}

impl Purchase {
    const MARKDOWN: &'static str = "## Add credit\n\nChoose a plan or purchase credits directly. We use Stripe to \
         process payments.";
}

impl RenderOnce for Purchase {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let theme = cx.theme();

        let extras = if self.prices.is_empty() {
            div()
                .text_color(theme.muted_foreground)
                .child(if self.loading {
                    "Loading plans…"
                } else {
                    "No plans are available right now."
                })
                .into_any_element()
        } else {
            v_flex()
                .gap_2()
                .child(
                    div()
                        .text_sm()
                        .text_color(theme.muted_foreground)
                        .child("Checkout opens in your browser; credit lands on this account."),
                )
                .child(plans::plan_rows(
                    &self.prices,
                    self.checkout_pending.as_deref(),
                    self.on_select,
                    "onboarding",
                    cx,
                ))
                .when_some(self.checkout_error, |el, err| {
                    el.child(error_line("checkout", err, cx))
                })
                .into_any_element()
        };

        slide_frame(
            "purchase",
            Self::MARKDOWN,
            Some(extras),
            cta_button(
                "purchase-later",
                "I will purchase credits later.",
                self.on_later,
            )
            .into_any_element(),
            window,
            cx,
        )
    }
}

// -- Shared layout + primitives ------------------------------------------------

/// The shared full-window slide layout: the prose body (with any `extras`
/// below it) vertically centered in a left-aligned reading column, and the
/// CTA group centered on the window at the bottom.
fn slide_frame(
    key: &'static str,
    markdown: &'static str,
    extras: Option<AnyElement>,
    ctas: AnyElement,
    window: &mut Window,
    cx: &mut App,
) -> AnyElement {
    // The prose editor is element-owned state (the `useState` analogue):
    // keyed per slide, initialized once, and evicted by the framework the
    // frame after the slide stops rendering (a branch truncation) — the
    // lifecycle the view's `bodies` map used to hand-roll. This is safe
    // *only because* every revealed slide paints every frame (a plain
    // v_flex stack, no virtualization); if the slides ever render
    // conditionally or through `list()`, lift this state back onto the view
    // (the retired chat view's `text_states` map, which died with an
    // unmounted `list()` item, was the lesson here).
    let prose_state = window.use_keyed_state(
        SharedString::from(format!("onboarding-prose-{key}")),
        cx,
        |window, cx| {
            let mut s = MarkdownEditorState::new(window, cx);
            s.set_value(markdown.to_string(), cx);
            s
        },
    );
    let prose = MarkdownEditor::new(&prose_state)
        .style(prose_style(cx))
        .disabled(true)
        .into_any_element();
    let column = v_flex()
        .w(COLUMN_WIDTH)
        .max_w_full()
        .gap_4()
        .child(prose)
        .child(extras.unwrap_or_else(|| div().into_any_element()));

    v_flex()
        .w_full()
        // The content box, not the raw surface: on Linux CSD `viewport_size`
        // includes the shadow padding, and the reveal/snap stride is measured
        // as `chrome::content_size().height` (see `glide_to_index`), so each
        // slide must be exactly that tall or snaps land between slides. Equal
        // to `viewport_size` off Linux CSD, so macOS/tests are unchanged.
        .h(crate::chrome::content_size(window).height)
        .child(
            // The prose region: fills the space above the CTAs, centering
            // the reading column vertically; horizontally centered as a unit.
            v_flex()
                .flex_1()
                .min_h_0()
                .justify_center()
                .pt(TITLE_BAR_RESERVE)
                .child(h_flex().w_full().justify_center().px_8().child(column)),
        )
        .child(
            // The CTAs are centered on the window (unlike the left-aligned
            // narrative), stacked and centered as a group.
            h_flex()
                .w_full()
                .justify_center()
                .px_8()
                .pb(px(72.))
                .child(v_flex().items_center().gap_3().child(ctas)),
        )
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
        .child(
            Button::new(SharedString::from(format!("onboarding-btn-{key}")))
                .ghost()
                .label(label)
                .on_click(on_click),
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
                    "Go to the previous slide",
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

/// A standalone clickable external link line ("Label ↗").
fn link_row(label: &'static str, url: &'static str, cx: &App) -> impl IntoElement {
    let theme = cx.theme();
    let slug: String = label
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect();
    div()
        .id(SharedString::from(format!("onboarding-link-{label}")))
        .probe(
            SharedString::from(format!("onboarding/link/{slug}")),
            Role::Link,
            label,
        )
        .w_full()
        .cursor_pointer()
        .text_color(theme.link)
        .hover(|s| s.underline())
        .child(SharedString::from(format!("{label} ↗")))
        .on_click(move |_, _, cx| cx.open_url(url))
}

/// A labeled single-line credential value in mono, with a Copy affordance.
fn credential_row(label: &'static str, value: SharedString, cx: &App) -> impl IntoElement {
    let theme = cx.theme();
    let to_copy = value.clone();
    v_flex()
        .gap_1()
        .child(
            div()
                .text_sm()
                .text_color(theme.muted_foreground)
                .child(label),
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
                        .id(SharedString::from(format!("onboarding-copy-{label}")))
                        .probe(
                            SharedString::from(format!("onboarding/copy/{label}")),
                            Role::Button,
                            SharedString::from(format!("Copy {label}")),
                        )
                        .flex_none()
                        .cursor_pointer()
                        .text_sm()
                        .text_color(theme.muted_foreground)
                        .hover(|s| s.text_color(theme.foreground))
                        .child("Copy")
                        .on_click(move |_, _, cx| {
                            cx.write_to_clipboard(ClipboardItem::new_string(to_copy.to_string()))
                        }),
                ),
        )
}

/// A labeled text input (existing-account credentials), probed for a11y.
fn labeled_input(
    label: &'static str,
    probe_name: &'static str,
    state: &Entity<InputState>,
) -> impl IntoElement {
    v_flex().gap_1().child(div().text_sm().child(label)).child(
        div()
            .id(SharedString::from(format!("onboarding-input-{label}")))
            .probe(probe_name, Role::TextInput, label)
            .w_full()
            .child(Input::new(state)),
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
