//! The onboarding window — a standalone singleton that walks a new user from
//! zero to a usable, funded account.
//!
//! Unlike the retired chat-window empty states (which this replaces), onboarding
//! is its own window: a vertical stack of **full-window slides**. Only the first
//! slide is present initially; each slide's call-to-action reveals the next slide
//! and animate-scrolls to it. Already-revealed slides can be scrolled back and
//! forth, and the page **snaps** to whole slides. While the finger is down the
//! page scrolls freely; at finger-lift we take over with our **own** decaying
//! glide (an ease-out from the release position, reusing the space view's easing
//! from [`crate::space_view::nav`]) to the target slide, and **suppress** the OS
//! momentum that follows so the whole motion is one continuous curve that lands
//! exactly on a slide. (This mirrors the space view's horizontal branch snap; in
//! this gpui pin macOS momentum arrives as `Moved` events with no end signal, so
//! we re-assert our glide position on each trailing step rather than letting it
//! drift the page — the same `reassert` trick.)
//!
//! The flow **branches** at "Get started": a new-account path (create → show the
//! id/secret → add credit) or an existing-account path (enter id/secret → verify
//! → add credit). Choosing a different branch upstream truncates and replaces the
//! downstream slides, so the visible flow always reflects the current choice.
//!
//! Styling borrows heavily from the space view: the same Newsreader prose column
//! ([`crate::space_view::prose_style`]) rendered through a disabled
//! `MarkdownEditor`, left-aligned in a centered reading column, with ghost-button
//! CTAs at the bottom of each slide.
//!
//! Structure: this module owns the *state* — the reveal machine, the scroll/snap
//! physics, and the async account operations. Each slide's *presentation* (its
//! prose, extras, and CTAs) is a self-contained [`RenderOnce`](gpui::RenderOnce)
//! component in [`slides`], constructed each frame by the [`render_slide`]
//! router with props + callbacks — the gpui analogue of React functional
//! components under a stateful parent.
//!
//! [`render_slide`]: OnboardingView::render_slide

use std::rc::Rc;
use std::time::{Duration, Instant};

use eidola_app_core::error::AppError;
use gpui::{
    AnyElement, AppContext, Context, Entity, FocusHandle, Focusable, InteractiveElement,
    IntoElement, IsZero, ParentElement, Pixels, Render, ScrollHandle, SharedString,
    StatefulInteractiveElement, Styled, Subscription, Task, TouchPhase, Window, div, point, px,
};
use gpui_component::{
    ActiveTheme,
    input::InputState,
    scroll::{Scrollbar, ScrollbarShow},
    v_flex,
};
use gpui_markdown_editor::MarkdownEditorState;

use crate::actions::CloseWindow;
use crate::plans;
use crate::space_view::TITLE_BAR_RESERVE;
use crate::space_view::nav::{ease_out_cubic, snap_duration, snap_target_index};
use crate::stores::Stores;
use crate::titlebar::{self, DragArm};

mod slides;

/// One page of the onboarding flow. The set that is *visible* is
/// [`OnboardingView::revealed`]; conditional slides only appear once the
/// upstream choice that leads to them is made.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Slide {
    /// "Pause here" — Eidola is not the same as the hosted assistants.
    Pause,
    /// "Eidola is your tool" — the CD-era sovereignty analogy.
    Tool,
    /// "Your control" — nobody but you can see or update it.
    Control,
    /// "Your responsibility" — models are fallible; effects are yours.
    Responsibility,
    /// "Get started" — the branch point (new vs. existing account).
    GetStarted,
    /// New-account branch: agree to terms, create an anonymous account.
    CreateAccount,
    /// New-account branch: the freshly-minted id + secret to save.
    NewAccount,
    /// Existing-account branch: enter id + secret, check the balance.
    ExistingAccount,
    /// Either branch: choose a plan / add credit.
    Purchase,
}

/// An in-flight vertical snap glide of the page toward a slide boundary.
#[derive(Clone, Debug)]
struct VSnap {
    from_y: f32,
    to_y: f32,
    start: Instant,
    duration: Duration,
}

pub struct OnboardingView {
    stores: Stores,
    focus_handle: FocusHandle,
    _subs: Vec<Subscription>,

    /// Revealed slides, in order. Always starts as `[Slide::Pause]`.
    revealed: Vec<Slide>,
    /// One disabled prose editor per revealed slide (its heading + intro),
    /// created lazily in render and pruned when a slide is truncated away.
    bodies: std::collections::HashMap<Slide, Entity<MarkdownEditorState>>,

    // -- Account creation (new-account branch) ----------------------------
    /// Whether the terms/privacy agreement checkbox is checked (gates the
    /// "Create a new account." button on [`Slide::CreateAccount`]).
    agreed: bool,
    creating: bool,
    /// The freshly created (id, secret) to present on [`Slide::NewAccount`].
    created: Option<(SharedString, SharedString)>,
    create_error: Option<String>,
    create_task: Option<Task<()>>,

    // -- Existing-account verification ------------------------------------
    id_input: Entity<InputState>,
    secret_input: Entity<InputState>,
    verifying: bool,
    /// `Ok(available_credits)` once verified, or `Err(message)` on failure.
    verify_result: Option<Result<i64, String>>,
    verify_task: Option<Task<()>>,

    // -- Checkout (purchase slide) ----------------------------------------
    checkout_pending: Option<String>,
    checkout_error: Option<String>,
    checkout_task: Option<Task<()>>,

    // -- Scroll + snap -----------------------------------------------------
    page_scroll: ScrollHandle,
    /// The in-flight decaying glide to a slide boundary — our *own* momentum
    /// curve (an ease-out from the release position), started at finger-lift or
    /// on a reveal. `None` while the page is free under the user's finger.
    snap: Option<VSnap>,
    /// True from finger-lift (`TouchPhase::Ended`) until the next gesture starts:
    /// we own the scroll. In this gpui pin the OS delivers post-lift **momentum**
    /// as `Moved` events with no end signal, and would otherwise drift the page
    /// off the target — so while owning we **suppress** it, re-asserting our
    /// glide/pin position on every trailing momentum step (mirrors the space
    /// view's `reassert_horizontal`). A fresh `Started` releases ownership.
    owning: bool,
    /// The resting offset the current glide lands on, held after the glide
    /// completes so trailing momentum can't nudge the page off the slide.
    pinned_y: Option<f32>,
    /// The last finger-drag step (px) — the release velocity for the flick
    /// decision at lift.
    last_dy: Pixels,
    /// A slide index to animate-scroll to on the next render (set when a slide
    /// is revealed; consumed in render where `window` is available).
    pending_scroll: Option<usize>,

    /// Titlebar drag arming (see [`crate::titlebar`]).
    drag_arm: DragArm,
}

impl OnboardingView {
    pub fn new(stores: Stores, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let focus_handle = cx.focus_handle();
        let id_input =
            cx.new(|cx| InputState::new(window, cx).placeholder("00000000-0000-0000-0000-…"));
        let secret_input = cx.new(|cx| InputState::new(window, cx).placeholder("account secret"));

        let _subs = vec![
            cx.observe(&stores.account, |_, _, cx| cx.notify()),
            cx.observe(&stores.config, |_, _, cx| cx.notify()),
        ];

        window.focus(&focus_handle, cx);

        Self {
            stores,
            focus_handle,
            _subs,
            revealed: vec![Slide::Pause],
            bodies: std::collections::HashMap::new(),
            agreed: false,
            creating: false,
            created: None,
            create_error: None,
            create_task: None,
            id_input,
            secret_input,
            verifying: false,
            verify_result: None,
            verify_task: None,
            checkout_pending: None,
            checkout_error: None,
            checkout_task: None,
            page_scroll: ScrollHandle::new(),
            snap: None,
            owning: false,
            pinned_y: None,
            last_dy: px(0.),
            pending_scroll: None,
            drag_arm: titlebar::drag_arm(),
        }
    }

    // -- Test seams --------------------------------------------------------

    /// The view's focus handle (behavior tests dispatch actions through it).
    pub fn focus_handle(&self) -> FocusHandle {
        self.focus_handle.clone()
    }

    /// The currently revealed slides, in order.
    pub fn revealed(&self) -> Vec<Slide> {
        self.revealed.clone()
    }

    /// The freshly-created account id + secret, if account creation succeeded.
    #[doc(hidden)]
    pub fn created_for_test(&self) -> Option<(String, String)> {
        self.created
            .as_ref()
            .map(|(id, secret)| (id.to_string(), secret.to_string()))
    }

    /// The last verification result (Ok(balance) / Err(message)), if any.
    #[doc(hidden)]
    pub fn verify_result_for_test(&self) -> Option<Result<i64, String>> {
        self.verify_result.clone()
    }

    /// The existing-account input editors (tests set their values directly).
    #[doc(hidden)]
    pub fn existing_inputs_for_test(&self) -> (Entity<InputState>, Entity<InputState>) {
        (self.id_input.clone(), self.secret_input.clone())
    }

    // -- Reveal / snap state machine --------------------------------------

    /// Reveal `next` immediately after `after`, truncating any slides that were
    /// revealed past `after` for a *different* choice — so re-choosing a branch
    /// upstream replaces the stale downstream slides. Idempotent when `next`
    /// already follows `after` (just re-scrolls to it). Animate-scrolls to the
    /// revealed slide.
    pub fn reveal(&mut self, after: Slide, next: Slide, cx: &mut Context<Self>) {
        let Some(pos) = self.revealed.iter().position(|s| *s == after) else {
            return;
        };
        let target = pos + 1;
        if self.revealed.get(target) != Some(&next) {
            self.revealed.truncate(target);
            self.revealed.push(next);
        }
        self.pending_scroll = Some(target);
        cx.notify();
    }

    /// Take over the scroll and drive our own decaying glide to slide `index`:
    /// mark ownership (so trailing OS momentum is suppressed), pin the resting
    /// offset, and start the ease-out. Used both for a reveal and for the
    /// finger-lift settle.
    fn glide_to_index(&mut self, index: usize, window: &mut Window, cx: &mut Context<Self>) {
        let stride = window.viewport_size().height.as_f32();
        if stride <= 0.0 {
            return;
        }
        let to_y = -(index as f32) * stride;
        let from_y = self.page_scroll.offset().y.as_f32();
        self.owning = true;
        self.pinned_y = Some(to_y);
        let dist = (to_y - from_y).abs();
        if dist < 0.5 {
            let off = self.page_scroll.offset();
            self.page_scroll.set_offset(point(off.x, px(to_y)));
            self.snap = None;
            cx.notify();
            return;
        }
        self.snap = Some(VSnap {
            from_y,
            to_y,
            start: Instant::now(),
            duration: snap_duration(dist, stride),
        });
        self.drive_snap(window, cx);
    }

    /// One frame of the active glide: ease the offset toward the target and, if
    /// not yet arrived, schedule the next frame.
    fn drive_snap(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(a) = self.snap.clone() else { return };
        let t = (a.start.elapsed().as_secs_f32() / a.duration.as_secs_f32().max(f32::EPSILON))
            .clamp(0.0, 1.0);
        let y = a.from_y + (a.to_y - a.from_y) * ease_out_cubic(t);
        let off = self.page_scroll.offset();
        self.page_scroll.set_offset(point(off.x, px(y)));
        if t >= 1.0 {
            self.snap = None;
        } else {
            let entity = cx.entity();
            window.on_next_frame(move |window, cx| {
                entity.update(cx, |this, cx| this.drive_snap(window, cx));
            });
        }
        cx.notify();
    }

    /// While we own the scroll, re-assert our authoritative position (the live
    /// glide point, or the settled pin) over whatever the OS momentum just
    /// applied to the handle — so the trailing momentum can't drift the page off
    /// the target. Mirrors the space view's `reassert_horizontal`.
    fn reassert_scroll(&self) {
        let y = if let Some(a) = self.snap.as_ref() {
            let t = (a.start.elapsed().as_secs_f32() / a.duration.as_secs_f32().max(f32::EPSILON))
                .clamp(0.0, 1.0);
            a.from_y + (a.to_y - a.from_y) * ease_out_cubic(t)
        } else if let Some(p) = self.pinned_y {
            p
        } else {
            return;
        };
        let off = self.page_scroll.offset();
        self.page_scroll.set_offset(point(off.x, px(y)));
    }

    // -- Account operations ------------------------------------------------

    /// Create an anonymous account (new-account branch). On success, store the
    /// id/secret, refresh config + balances, and reveal the "Your new account"
    /// slide; on failure, surface the error inline.
    pub fn begin_create(&mut self, cx: &mut Context<Self>) {
        if self.creating {
            return;
        }
        self.creating = true;
        self.create_error = None;
        cx.notify();

        let Some(rx) = self.stores.account.read(cx).request_account_create() else {
            // Stub / no backend: the in-flight marker is the observable state.
            return;
        };
        self.create_task = Some(cx.spawn(async move |this, cx| {
            let res = rx.await.unwrap_or_else(|_| {
                Err(AppError::Internal {
                    message: "account creation task cancelled".into(),
                })
            });
            let _ = this.update(cx, |this, cx| {
                this.creating = false;
                this.create_task = None;
                match res {
                    Ok(created) => {
                        this.created = Some((created.id.into(), created.secret.into()));
                        this.stores.config.update(cx, |c, cx| c.refresh(cx));
                        this.stores.account.update(cx, |s, cx| {
                            s.refresh_prices(cx);
                            s.refresh_balances(cx);
                        });
                        this.reveal(Slide::CreateAccount, Slide::NewAccount, cx);
                    }
                    Err(e) => this.create_error = Some(e.to_string()),
                }
                cx.notify();
            });
        }));
    }

    /// Verify the entered existing-account credentials (existing-account branch).
    /// On success the credentials are linked and the balance is shown; on failure
    /// the config is left untouched (the store rolls the write back) and an
    /// honest message is shown.
    pub fn begin_verify(&mut self, cx: &mut Context<Self>) {
        if self.verifying {
            return;
        }
        let id = self.id_input.read(cx).value().trim().to_string();
        let secret = self.secret_input.read(cx).value().trim().to_string();
        if id.is_empty() || secret.is_empty() {
            self.verify_result = Some(Err("Enter both an account ID and secret.".into()));
            cx.notify();
            return;
        }
        self.verifying = true;
        self.verify_result = None;
        cx.notify();

        let Some(rx) = self
            .stores
            .account
            .read(cx)
            .request_verify_account(id, secret)
        else {
            return;
        };
        self.verify_task = Some(cx.spawn(async move |this, cx| {
            let res = rx.await.unwrap_or_else(|_| {
                Err(AppError::Internal {
                    message: "verification task cancelled".into(),
                })
            });
            let _ = this.update(cx, |this, cx| {
                this.verifying = false;
                this.verify_task = None;
                match res {
                    Ok(balances) => {
                        let available = balances.available;
                        this.stores.config.update(cx, |c, cx| c.refresh(cx));
                        this.stores.account.update(cx, |s, cx| {
                            s.set_balances(balances, cx);
                            s.refresh_prices(cx);
                        });
                        this.verify_result = Some(Ok(available));
                    }
                    Err(e) => this.verify_result = Some(Err(verify_error_copy(&e))),
                }
                cx.notify();
            });
        }));
    }

    /// Open a Stripe checkout for `price_id` in the browser (purchase slide).
    pub fn begin_checkout(&mut self, price_id: String, cx: &mut Context<Self>) {
        if self.checkout_pending.is_some() {
            return;
        }
        self.checkout_pending = Some(price_id.clone());
        self.checkout_error = None;
        cx.notify();

        let Some(rx) = self.stores.account.read(cx).request_checkout(price_id) else {
            return;
        };
        self.checkout_task = Some(cx.spawn(async move |this, cx| {
            let res = rx.await.unwrap_or_else(|_| {
                Err(AppError::Internal {
                    message: "checkout task cancelled".into(),
                })
            });
            let _ = this.update(cx, |this, cx| {
                this.checkout_pending = None;
                this.checkout_task = None;
                match res {
                    Ok(url) => cx.open_url(&url),
                    Err(e) => this.checkout_error = Some(e.to_string()),
                }
                cx.notify();
            });
        }));
    }

    /// Ensure a prose editor exists for every revealed slide, and prune editors
    /// for slides that have been truncated away.
    fn sync_bodies(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        for &slide in &self.revealed.clone() {
            self.bodies.entry(slide).or_insert_with(|| {
                cx.new(|cx| {
                    let mut s = MarkdownEditorState::new(window, cx);
                    s.set_value(slides::markdown(slide).to_string(), cx);
                    s
                })
            });
        }
        let live: std::collections::HashSet<Slide> = self.revealed.iter().copied().collect();
        self.bodies.retain(|slide, _| live.contains(slide));
    }
}

impl Focusable for OnboardingView {
    fn focus_handle(&self, _cx: &gpui::App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for OnboardingView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let bg = theme.background;
        let fg = theme.foreground;
        let font_family = theme.font_family.clone();

        self.sync_bodies(window, cx);

        // Kick off the animate-scroll to a freshly-revealed slide now that we
        // have `window` (the reveal itself just recorded the target index).
        if let Some(idx) = self.pending_scroll.take() {
            self.glide_to_index(idx, window, cx);
        }

        let revealed = self.revealed.clone();
        let body = v_flex().w_full().children(
            revealed
                .into_iter()
                .map(|slide| self.render_slide(slide, cx)),
        );

        div()
            .track_focus(&self.focus_handle)
            .key_context("OnboardingView")
            .on_action(cx.listener(|_, _: &CloseWindow, window, _| {
                window.remove_window();
            }))
            .relative()
            .size_full()
            .bg(bg)
            .font_family(font_family)
            .text_color(fg)
            .child(
                div()
                    .id("onboarding-scroll")
                    .size_full()
                    .overflow_y_scroll()
                    .track_scroll(&self.page_scroll)
                    // While the finger is down the page scrolls natively; at
                    // lift we take over with our own decaying glide (a computed
                    // momentum curve) to the target slide, and *suppress* the OS
                    // momentum that follows — each trailing momentum `Moved` is
                    // re-asserted back onto our glide so it can't drift the page
                    // off the slide. A fresh `Started` releases ownership.
                    .on_scroll_wheel(cx.listener(
                        |this, ev: &gpui::ScrollWheelEvent, window, cx| match ev.touch_phase {
                            TouchPhase::Started => {
                                this.owning = false;
                                this.pinned_y = None;
                                this.snap = None;
                                this.last_dy = px(0.);
                            }
                            TouchPhase::Moved => {
                                if this.owning {
                                    // Post-lift momentum — keep the page on our
                                    // glide/pin instead of letting it drift.
                                    cx.stop_propagation();
                                    this.reassert_scroll();
                                } else {
                                    // Active finger drag — remember the release
                                    // velocity for the flick decision.
                                    let dy = ev.delta.pixel_delta(window.line_height()).y;
                                    if !dy.is_zero() {
                                        this.last_dy = dy;
                                    }
                                }
                            }
                            TouchPhase::Ended => {
                                // Finger lifted: pick the target slide from the
                                // release velocity (nearest, or ±1 on a flick)
                                // and glide there, owning the scroll so the OS
                                // momentum that follows is suppressed.
                                let stride = window.viewport_size().height.as_f32();
                                if stride > 0.0 {
                                    let from_y = this.page_scroll.offset().y.as_f32();
                                    let target = snap_target_index(
                                        from_y,
                                        stride,
                                        this.last_dy.as_f32(),
                                        this.revealed.len(),
                                    );
                                    this.glide_to_index(target, window, cx);
                                }
                            }
                        },
                    ))
                    .child(body),
            )
            // A right-edge scroll indicator that appears only while scrolling
            // (`ScrollbarShow::Scrolling`) — the built-in gpui-component
            // overlay bound to the same page scroll handle.
            .child(
                div()
                    .absolute()
                    .top_0()
                    .right_0()
                    .bottom_0()
                    .w(px(14.))
                    .child(
                        Scrollbar::vertical(&self.page_scroll)
                            .scrollbar_show(ScrollbarShow::Scrolling),
                    ),
            )
            // The titlebar drag band paints last so it wins hit-testing over the
            // slide content beneath the traffic lights.
            .child(titlebar::drag_band(
                "onboarding-titlebar",
                TITLE_BAR_RESERVE,
                self.drag_arm.clone(),
            ))
    }
}

impl OnboardingView {
    /// The slide router: construct the matching [`slides`] component with its
    /// props — state snapshots plus callbacks back into this view. Flow
    /// sequencing (which slide a CTA reveals next) lives here so the branch
    /// structure is visible in one place; everything a slide *shows* lives on
    /// its component.
    fn render_slide(&self, slide: Slide, cx: &Context<Self>) -> AnyElement {
        let prose = self.bodies.get(&slide).cloned();
        match slide {
            Slide::Pause => slides::Pause {
                prose,
                on_advance: Box::new(
                    cx.listener(|this, _, _, cx| this.reveal(Slide::Pause, Slide::Tool, cx)),
                ),
            }
            .into_any_element(),
            Slide::Tool => slides::Tool {
                prose,
                on_advance: Box::new(
                    cx.listener(|this, _, _, cx| this.reveal(Slide::Tool, Slide::Control, cx)),
                ),
            }
            .into_any_element(),
            Slide::Control => slides::Control {
                prose,
                on_advance: Box::new(cx.listener(|this, _, _, cx| {
                    this.reveal(Slide::Control, Slide::Responsibility, cx)
                })),
            }
            .into_any_element(),
            Slide::Responsibility => slides::Responsibility {
                prose,
                on_advance: Box::new(cx.listener(|this, _, _, cx| {
                    this.reveal(Slide::Responsibility, Slide::GetStarted, cx)
                })),
            }
            .into_any_element(),
            Slide::GetStarted => slides::GetStarted {
                prose,
                on_new_account: Box::new(cx.listener(|this, _, _, cx| {
                    this.reveal(Slide::GetStarted, Slide::CreateAccount, cx)
                })),
                on_existing_account: Box::new(cx.listener(|this, _, _, cx| {
                    this.reveal(Slide::GetStarted, Slide::ExistingAccount, cx)
                })),
            }
            .into_any_element(),
            Slide::CreateAccount => slides::CreateAccount {
                prose,
                agreed: self.agreed,
                creating: self.creating,
                error: self.create_error.clone(),
                on_toggle_agree: Box::new(cx.listener(|this, checked: &bool, _, cx| {
                    this.agreed = *checked;
                    cx.notify();
                })),
                on_create: Box::new(cx.listener(|this, _, _, cx| this.begin_create(cx))),
            }
            .into_any_element(),
            Slide::NewAccount => {
                let (id, secret) = self.created.clone().unwrap_or_default();
                slides::NewAccount {
                    prose,
                    id,
                    secret,
                    on_saved: Box::new(cx.listener(|this, _, _, cx| {
                        this.reveal(Slide::NewAccount, Slide::Purchase, cx)
                    })),
                }
                .into_any_element()
            }
            Slide::ExistingAccount => slides::ExistingAccount {
                prose,
                id_input: self.id_input.clone(),
                secret_input: self.secret_input.clone(),
                verifying: self.verifying,
                verify_result: self.verify_result.clone(),
                on_verify: Box::new(cx.listener(|this, _, _, cx| this.begin_verify(cx))),
                on_purchase: Box::new(cx.listener(|this, _, _, cx| {
                    this.reveal(Slide::ExistingAccount, Slide::Purchase, cx)
                })),
                on_done: Box::new(cx.listener(|_, _, window, _| window.remove_window())),
            }
            .into_any_element(),
            Slide::Purchase => {
                let account = self.stores.account.read(cx);
                let weak = cx.entity().downgrade();
                let on_select: plans::PlanSelectHandler = Rc::new(move |price_id, _window, app| {
                    let _ = weak.update(app, |this, cx| this.begin_checkout(price_id, cx));
                });
                slides::Purchase {
                    prose,
                    prices: account.prices().value().cloned().unwrap_or_default(),
                    loading: account.is_loading(),
                    checkout_pending: self.checkout_pending.clone(),
                    checkout_error: self.checkout_error.clone(),
                    on_select,
                    on_later: Box::new(cx.listener(|_, _, window, _| window.remove_window())),
                }
                .into_any_element()
            }
        }
    }
}

/// Map a verification failure to honest user-facing copy. The server collapses
/// "no such account" and "wrong secret" into one `401` (to avoid account
/// enumeration), so those are indistinguishable and share one message; other
/// failures (network, attestation) surface their real reason.
fn verify_error_copy(e: &AppError) -> String {
    match e {
        AppError::Server {
            status: 401 | 403, ..
        } => "We couldn't verify that account. Check the ID and secret, or create a new account \
              instead."
            .to_string(),
        other => other.to_string(),
    }
}
