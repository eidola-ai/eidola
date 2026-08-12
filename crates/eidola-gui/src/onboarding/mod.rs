//! The onboarding window — a standalone singleton that walks a new user from
//! zero to a usable, funded account.
//!
//! Unlike the retired chat-window empty states (which this replaces), onboarding
//! is its own window: **one continuous scrolling page** of stacked slides. Only
//! the first slide is present initially; each slide's call-to-action reveals the
//! next slide and animate-scrolls to it. Already-revealed slides can be scrolled
//! back and forth freely.
//!
//! Each slide is sized **by its content**, with a `min_h` of one window so a
//! short slide still reads as a full page and a long one grows and scrolls
//! rather than clipping — its call-to-action sits in normal flow *after* the
//! prose, so it can never overlap the narrative on a small window. (This
//! replaced the previous fixed-window-height slides, whose vertically-centered
//! prose overflowed onto the CTAs and whose mandatory whole-window snap made the
//! overflow unreachable.)
//!
//! **Snapping is proximity, not mandatory** — the behavior of CSS
//! `scroll-snap-type: y proximity`. A *user* gesture that comes to rest **near**
//! a slide boundary glides onto it; one that ends **mid-content** (reading a
//! long slide) stays exactly where the reader left it. The decision is made at
//! finger-lift by [`crate::space_view::nav::proximity_snap_target`] over the
//! live slide-top offsets: a "near" (or a flick beside a boundary) result is
//! driven home by our **own** decaying ease-out glide with the trailing OS
//! momentum **suppressed** (in this gpui pin macOS momentum arrives as `Moved`
//! events with no end signal, so — as in the space view's horizontal branch
//! snap — we re-assert our glide position on each trailing step); a "far" result
//! simply **releases** to the native momentum so the page rests in the content.
//! Programmatic **reveal** and **back** glides always land exactly on their
//! target slide (proximity governs only user gestures).
//!
//! The flow **branches** at "Get started": a new-account path (create → show the
//! id/secret → add credit) or an existing-account path (enter id/secret → verify
//! → add credit). Choosing a different branch upstream truncates and replaces the
//! downstream slides, so the visible flow always reflects the current choice.
//!
//! Styling borrows heavily from the space view: the same Newsreader prose column
//! ([`crate::space_view::prose_style`]) rendered through a disabled
//! `MarkdownEditor`, left-aligned in a centered reading column, with ghost-button
//! CTAs in normal flow beneath the prose.
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

use eidola_app_core::SubscriptionState;
use eidola_app_core::error::AppError;
use gpui::{
    AnyElement, App, AppContext, Context, Entity, FocusHandle, Focusable, InteractiveElement,
    IntoElement, IsZero, ParentElement, Pixels, Render, ScrollHandle, SharedString,
    StatefulInteractiveElement, Styled, Subscription, Task, TouchPhase, Window, div, point, px,
};
use gpui_component::{
    ActiveTheme,
    input::InputState,
    scroll::{Scrollbar, ScrollbarShow},
};

use crate::actions::CloseWindow;
use crate::plans;
use crate::space_view::TITLE_BAR_RESERVE;
use crate::space_view::nav::{ease_out_cubic, proximity_snap_target, snap_duration};
use crate::stores::Stores;
use crate::titlebar;

mod slides;

/// Fraction of the window's content height within which a *released* scroll
/// gesture counts as "near" a slide boundary and snaps onto it (proximity
/// snapping). Beyond this band the page stays where the reader left it, so a
/// long slide can be read without being yanked to the next one. Scales with the
/// window so the feel is consistent across sizes.
const SNAP_PROXIMITY_FRACTION: f32 = 0.25;

/// One page of the onboarding flow. The set that is *visible* is
/// [`OnboardingView::revealed`]; conditional slides only appear once the
/// upstream choice that leads to them is made.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Slide {
    /// "Pause here" — Eidola is not the same as the hosted assistants.
    Pause,
    /// "Eidola is your tool" — the CD-era sovereignty analogy.
    Tool,
    /// "Your control" — no operator can read, retain, or change it.
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

    /// Revealed slides, in order. Always starts as `[Slide::Pause]`. (Each
    /// slide's prose editor is element-owned state inside its component —
    /// see [`slides`] — so revealing/truncating slides needs no bookkeeping
    /// here beyond this list.)
    revealed: Vec<Slide>,

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

    /// Glide the page to a revealed slide by index — the back arrow's action.
    /// Public so behavior tests can drive the same path the arrow's click takes.
    #[doc(hidden)]
    pub fn scroll_to_slide(&mut self, index: usize, window: &mut Window, cx: &mut Context<Self>) {
        self.glide_to_index(index, window, cx);
    }

    /// The resting offset the current glide lands on (test seam for the snap).
    #[doc(hidden)]
    pub fn pinned_y_for_test(&self) -> Option<f32> {
        self.pinned_y
    }

    /// The measured content-space top of each revealed slide (test seam for the
    /// size-to-content / min-height-and-grow contract). Requires a prior paint
    /// so the child bounds are populated.
    #[doc(hidden)]
    pub fn slide_tops_for_test(&self, window: &Window) -> Vec<f32> {
        self.slide_tops(window)
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
        // Reaching the plans is the moment this window needs to know whether
        // the account already subscribes — the "existing account" branch can
        // arrive here with one, and the server refuses a second. Nothing on
        // the bus carries this (it is a live read that persists nothing), so
        // the reveal is where it is asked for.
        if next == Slide::Purchase {
            self.stores
                .account
                .update(cx, |s, cx| s.refresh_subscription(cx));
        }
        self.pending_scroll = Some(target);
        cx.notify();
    }

    /// The content-space y of each revealed slide's top (ascending, `[0] == 0`),
    /// read from the live child bounds of the scroll container — so it copes
    /// with slides taller than the window and with variable heights. Falls back
    /// per-slide to the previous slide's bottom (a just-revealed slide whose own
    /// bounds haven't painted yet inherits its predecessor's measured bottom),
    /// and finally to a one-window stride before the first paint.
    fn slide_tops(&self, window: &Window) -> Vec<f32> {
        let content_h = crate::chrome::content_size(window).height.as_f32();
        let base = self.page_scroll.bounds().top().as_f32();
        let mut tops = Vec::with_capacity(self.revealed.len());
        for i in 0..self.revealed.len() {
            let t = if let Some(b) = self.page_scroll.bounds_for_item(i) {
                b.top().as_f32() - base
            } else if i == 0 {
                0.0
            } else {
                // Not yet painted (freshly revealed): its top is the previous
                // slide's bottom, or one stride past the previous top.
                self.page_scroll
                    .bounds_for_item(i - 1)
                    .map(|b| b.bottom().as_f32() - base)
                    .unwrap_or_else(|| tops[i - 1] + content_h)
            };
            tops.push(t);
        }
        tops
    }

    /// Take over the scroll and drive our own decaying glide to slide `index`:
    /// mark ownership (so trailing OS momentum is suppressed), pin the resting
    /// offset, and start the ease-out. Used for a reveal and for the back arrow
    /// (programmatic, always exact), and for a proximity finger-lift settle.
    fn glide_to_index(&mut self, index: usize, window: &mut Window, cx: &mut Context<Self>) {
        let content_h = crate::chrome::content_size(window).height.as_f32();
        if content_h <= 0.0 {
            return;
        }
        let tops = self.slide_tops(window);
        let to_y = -tops.get(index).copied().unwrap_or(index as f32 * content_h);
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
            duration: snap_duration(dist, content_h),
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

    /// Leave onboarding through one of its deliberate exits (skip / done /
    /// later), landing the user in the app.
    ///
    /// At launch onboarding opens *instead of* a blank space (see `run`), so
    /// the exit has to open the space it stood in for — otherwise a finished
    /// flow leaves no window at all: macOS would linger dock-only, and Linux
    /// quits outright the moment the last window closes (gpui's own
    /// `QuitMode::Default`, plus `run`'s `on_window_closed`). Only when this
    /// *is* the last window, though — opened from the Eidola menu there's
    /// already a window behind it, and a surprise second blank space is not
    /// what "Later" means.
    ///
    /// Order is load-bearing: open first, close second. `remove_window` only
    /// flags the window; gpui tears it down and runs the quit-on-empty check
    /// as this update unwinds, so the replacement must already be in
    /// `cx.windows()` by then. Deferring the open would land it after the
    /// app had quit.
    fn leave(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if cx.windows().len() <= 1 {
            crate::open_blank_space_window(cx, self.stores.clone());
        }
        window.remove_window();
    }

    /// The account-free path: disable the `eidola` backend (recorded in the
    /// DB, so launch stops auto-opening onboarding) and leave. Asks then
    /// route only to on-device / self-configured backends; the choice
    /// reverses any time in Settings → Backends (or by walking this flow
    /// again from the Eidola menu).
    pub fn skip_account(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.stores
            .backends
            .update(cx, |s, cx| s.set_enabled("eidola".into(), false, cx));
        self.leave(window, cx);
    }

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
                        this.forget_account_scoped_view_state();
                        this.stores.account.update(cx, |s, cx| {
                            s.refresh_prices(cx);
                            s.refresh_balances(cx);
                            // The account this machine speaks for has just
                            // changed — drop what the last one owned and read
                            // the new one's standing.
                            s.account_identity_changed(cx);
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
            let _ = this.update(cx, |this, cx| this.finish_verify(res, cx));
        }));
    }

    /// Land a credential check. Public so behavior tests drive the same path
    /// the request's own task does.
    pub fn finish_verify(
        &mut self,
        res: Result<eidola_app_core::BalancesResult, AppError>,
        cx: &mut Context<Self>,
    ) {
        self.verifying = false;
        self.verify_task = None;
        match res {
            Ok(balances) => {
                let available = balances.available;
                self.stores.config.update(cx, |c, cx| c.refresh(cx));
                self.forget_account_scoped_view_state();
                self.stores.account.update(cx, |s, cx| {
                    s.set_balances(balances, cx);
                    s.refresh_prices(cx);
                    // Linking a *different* account changes whose
                    // subscription this is, and an Account pane already on
                    // screen has no other trigger to re-read: nothing on the
                    // bus invalidates that cell and activation only fires on
                    // a pane change.
                    s.account_identity_changed(cx);
                });
                self.verify_result = Some(Ok(available));
            }
            Err(e) => self.verify_result = Some(Err(verify_error_copy(&e))),
        }
        cx.notify();
    }

    /// Open a Stripe checkout for `price_id` in the browser (purchase slide).
    pub fn begin_checkout(&mut self, price_id: String, cx: &mut Context<Self>) {
        if self.checkout_pending.is_some() {
            return;
        }
        self.checkout_pending = Some(price_id.clone());
        self.checkout_error = None;
        cx.notify();

        // Which account this checkout would fund. The reader can go back from
        // the Purchase slide and link a different account while the request is
        // in flight, so the answer is re-asked when it lands.
        let minted_for = self.account_identity(cx);
        let Some(rx) = self.stores.account.read(cx).request_checkout(price_id) else {
            return;
        };
        self.checkout_task = Some(cx.spawn(async move |this, cx| {
            let res = rx.await.unwrap_or_else(|_| {
                Err(AppError::Internal {
                    message: "checkout task cancelled".into(),
                })
            });
            let _ = this.update(cx, |this, cx| this.finish_checkout(minted_for, res, cx));
        }));
    }

    /// Land a checkout mint. Public so behavior tests drive the same path the
    /// request's own task does.
    pub fn finish_checkout(
        &mut self,
        minted_for: Option<String>,
        res: Result<String, AppError>,
        cx: &mut Context<Self>,
    ) {
        self.checkout_pending = None;
        self.checkout_task = None;
        match res {
            Ok(url) => {
                if self.mint_is_current(&minted_for, cx) {
                    cx.open_url(&url);
                } else {
                    self.checkout_error = Some(crate::account::STALE_MINT.to_string());
                }
            }
            Err(e) => self.checkout_error = Some(e.to_string()),
        }
        cx.notify();
    }

    pub fn checkout_error(&self) -> Option<&str> {
        self.checkout_error.as_deref()
    }

    pub fn checkout_pending(&self) -> Option<&str> {
        self.checkout_pending.as_deref()
    }

    /// Drop the checkout state started for the account being replaced — the
    /// Account pane's rule (see `AccountView::forget_account_scoped_view_state`)
    /// on the surface that owns the other half of it.
    ///
    /// Here the **event** hook is enough where the pane's could not be: this
    /// view is the sole author of an identity change during onboarding (there
    /// is no reset slide), so creating and linking are the whole set. A reader
    /// who pressed a plan, went back, and linked a different account must not
    /// return to a Purchase slide whose row still reads as in flight for the
    /// account they walked away from.
    fn forget_account_scoped_view_state(&mut self) {
        self.checkout_pending = None;
        self.checkout_task = None;
        self.checkout_error = None;
    }

    /// The account a checkout started now would fund. Read past the
    /// `ConfigStore` cache — see [`crate::stores::AccountStore::account_identity`].
    fn account_identity(&self, cx: &App) -> Option<String> {
        let fallback = self.cached_account_id(cx);
        self.stores
            .account
            .read(cx)
            .account_identity(fallback.as_deref())
    }

    fn cached_account_id(&self, cx: &App) -> Option<String> {
        self.stores
            .config
            .read(cx)
            .state()
            .and_then(|s| s.account_id.clone())
    }

    fn mint_is_current(&self, minted_for: &Option<String>, cx: &App) -> bool {
        let fallback = self.cached_account_id(cx);
        self.stores
            .account
            .read(cx)
            .mint_is_current(minted_for.as_deref(), fallback.as_deref())
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

        // Kick off the animate-scroll to a freshly-revealed slide now that we
        // have `window` (the reveal itself just recorded the target index).
        if let Some(idx) = self.pending_scroll.take() {
            self.glide_to_index(idx, window, cx);
        }

        let revealed = self.revealed.clone();
        let slides = revealed.into_iter().enumerate().map(|(idx, slide)| {
            let inner = self.render_slide(slide, cx);
            // Every slide past the first carries a visible up-chevron "back"
            // affordance (a discoverable alternative to the scroll-back
            // gesture) that glides to the previous slide. Flow sequencing —
            // forward *and* back — lives with the parent, so the back wiring
            // is here rather than on the stateless slide components.
            if idx == 0 {
                inner
            } else {
                let on_back: slides::OnClick = Box::new(cx.listener(move |this, _, window, cx| {
                    this.glide_to_index(idx - 1, window, cx);
                }));
                div()
                    .relative()
                    .w_full()
                    .flex_none()
                    .child(inner)
                    .child(slides::back_button(idx, on_back, cx))
                    .into_any_element()
            }
        });

        // Round all four corners to the CSD frame on Linux (no-op on
        // macOS/SSD) — the same treatment every other window root gets.
        crate::chrome::round_client_corners(div(), window)
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
                // The scroll container *is* the slide column: the slides are its
                // direct children so `page_scroll` tracks a per-slide child
                // bound (the boundary offsets the proximity snap reads).
                div()
                    .id("onboarding-scroll")
                    .size_full()
                    .flex()
                    .flex_col()
                    .items_stretch()
                    .overflow_y_scroll()
                    .track_scroll(&self.page_scroll)
                    // While the finger is down the page scrolls natively; at
                    // lift we make a *proximity* decision: a rest near a slide
                    // boundary (or a flick beside one) is driven home by our own
                    // decaying glide with the trailing OS momentum suppressed
                    // (each momentum `Moved` re-asserted back onto the glide so
                    // it can't drift the page); a rest deep in a slide's content
                    // simply releases to the native momentum so the reader keeps
                    // their place. A fresh `Started` releases ownership.
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
                                    // Post-lift momentum after a *snap* — keep the
                                    // page on our glide/pin instead of drifting.
                                    cx.stop_propagation();
                                    this.reassert_scroll();
                                } else {
                                    // Active finger drag (or momentum after a
                                    // *release*): let it scroll, and remember the
                                    // release velocity for the flick decision.
                                    let dy = ev.delta.pixel_delta(window.line_height()).y;
                                    if !dy.is_zero() {
                                        this.last_dy = dy;
                                    }
                                }
                            }
                            TouchPhase::Ended => {
                                // Finger lifted: proximity decision. Near a
                                // boundary (or a flick beside one) → glide onto
                                // it, owning the scroll to suppress OS momentum;
                                // deep in content → release, leaving the page to
                                // rest under native momentum where the reader is.
                                let content_h = crate::chrome::content_size(window).height.as_f32();
                                if content_h > 0.0 {
                                    let tops = this.slide_tops(window);
                                    let viewport_top = -this.page_scroll.offset().y.as_f32();
                                    let proximity = content_h * SNAP_PROXIMITY_FRACTION;
                                    match proximity_snap_target(
                                        &tops,
                                        viewport_top,
                                        this.last_dy.as_f32(),
                                        proximity,
                                    ) {
                                        Some(idx) => this.glide_to_index(idx, window, cx),
                                        None => {
                                            // Stay put — release to native momentum.
                                            this.owning = false;
                                            this.pinned_y = None;
                                            this.snap = None;
                                            cx.notify();
                                        }
                                    }
                                }
                            }
                            TouchPhase::Cancelled => {
                                // The system took the gesture; it never
                                // committed, so unwind rather than snap —
                                // the same release the "stay put" decision
                                // above performs, minus the proximity choice.
                                this.owning = false;
                                this.pinned_y = None;
                                this.snap = None;
                                this.last_dy = px(0.);
                                cx.notify();
                            }
                        },
                    ))
                    .children(slides),
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
                window,
                cx,
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
        match slide {
            Slide::Pause => slides::Pause {
                on_advance: Box::new(
                    cx.listener(|this, _, _, cx| this.reveal(Slide::Pause, Slide::Tool, cx)),
                ),
            }
            .into_any_element(),
            Slide::Tool => slides::Tool {
                on_advance: Box::new(
                    cx.listener(|this, _, _, cx| this.reveal(Slide::Tool, Slide::Control, cx)),
                ),
            }
            .into_any_element(),
            Slide::Control => slides::Control {
                on_advance: Box::new(cx.listener(|this, _, _, cx| {
                    this.reveal(Slide::Control, Slide::Responsibility, cx)
                })),
            }
            .into_any_element(),
            Slide::Responsibility => slides::Responsibility {
                on_advance: Box::new(cx.listener(|this, _, _, cx| {
                    this.reveal(Slide::Responsibility, Slide::GetStarted, cx)
                })),
            }
            .into_any_element(),
            Slide::GetStarted => slides::GetStarted {
                on_new_account: Box::new(cx.listener(|this, _, _, cx| {
                    this.reveal(Slide::GetStarted, Slide::CreateAccount, cx)
                })),
                on_existing_account: Box::new(cx.listener(|this, _, _, cx| {
                    this.reveal(Slide::GetStarted, Slide::ExistingAccount, cx)
                })),
                on_skip_account: Box::new(
                    cx.listener(|this, _, window, cx| this.skip_account(window, cx)),
                ),
            }
            .into_any_element(),
            Slide::CreateAccount => slides::CreateAccount {
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
                    id,
                    secret,
                    on_saved: Box::new(cx.listener(|this, _, _, cx| {
                        this.reveal(Slide::NewAccount, Slide::Purchase, cx)
                    })),
                }
                .into_any_element()
            }
            Slide::ExistingAccount => slides::ExistingAccount {
                id_input: self.id_input.clone(),
                secret_input: self.secret_input.clone(),
                verifying: self.verifying,
                verify_result: self.verify_result.clone(),
                on_verify: Box::new(cx.listener(|this, _, _, cx| this.begin_verify(cx))),
                on_purchase: Box::new(cx.listener(|this, _, _, cx| {
                    this.reveal(Slide::ExistingAccount, Slide::Purchase, cx)
                })),
                on_done: Box::new(cx.listener(|this, _, window, cx| this.leave(window, cx))),
            }
            .into_any_element(),
            Slide::Purchase => {
                let account = self.stores.account.read(cx);
                let weak = cx.entity().downgrade();
                let on_select: plans::PlanSelectHandler = Rc::new(move |price_id, _window, app| {
                    let _ = weak.update(app, |this, cx| this.begin_checkout(price_id, cx));
                });
                // Only an answered, affirmative read narrows the list; an
                // unanswered or failed one leaves every plan offered and
                // lets the server refuse, which it does honestly. Managing
                // an existing subscription is Settings' job, so this slide
                // says where rather than growing a second door.
                let subscribed = account
                    .subscription()
                    .value()
                    .is_some_and(|s| s.state == SubscriptionState::Active);
                let prices = account.prices().value().cloned().unwrap_or_default();
                slides::Purchase {
                    prices: plans::offered_plans(&prices, subscribed),
                    subscribed,
                    loading: account.is_loading(),
                    checkout_pending: self.checkout_pending.clone(),
                    checkout_error: self.checkout_error.clone(),
                    on_select,
                    on_later: Box::new(cx.listener(|this, _, window, cx| this.leave(window, cx))),
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
